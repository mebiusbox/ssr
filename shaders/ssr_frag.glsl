#define MAX_ITERATIONS 300
#define MAX_BINARY_SEARCH_ITERATIONS 64
uniform sampler2D ColorSampler;
uniform sampler2D NormalSampler;
uniform sampler2D DepthSampler;
uniform sampler2D BackDepthSampler;
uniform sampler2D MetalRoughSampler;
uniform samplerCube EnvSampler;
uniform vec4 Resolution;    // xy: BufferSize, zw: InvBufferSize
uniform vec4 _ProjectionParams; // x: 1.0, y: near, z: far, w: 1/far
uniform vec4 _ScreenParams;     // x: targetWidth, y: targetHeight, z: 1.0+1.0/w, w: 1.0+1.0/h
uniform float CameraNear;
uniform float CameraFar;
uniform mat4 CameraProjectionMatrix;        // projection matrix that maps to screen pixels (not NDC)
uniform mat4 CameraInverseProjectionMatrix; // inverse projection matrix (NDC to camera space)
uniform mat4 ViewInverseMatrix;
uniform float Iterations;                   // maximum ray iterations
uniform float BinarySearchIterations;       // maximum binary search refinement iterations
uniform float PixelZSize;                   // Z size in camera space of a pixel in the depth buffer = Thickness
uniform float PixelStride;                  // number of pixels per ray step close to camera
uniform float PixelStrideZCutoff;           // ray origin Z at this distance will have a pixel stride of 1.0
uniform float MaxRayDistance;               // maximum distance of a ray
uniform float ScreenEdgeFadeStart;          // distance to screen edge that ray hits will start to fade [0,1]
uniform float EyeFadeStart;                 // ray direction's Z that ray hits will start to fade [0,1]
uniform float EyeFadeEnd;                   // ray direction's Z that ray hits will be cut [0,1]
uniform float EdgeDistance;
uniform float EdgeExponent;
uniform float FadeDistance;
uniform float FadeExponent;
varying vec2 vUv;

vec3 transformDirection(in vec3 dir, in mat4 matrix) {
  return normalize((matrix * vec4(dir, 0.0)).xyz);
}
vec3 inverseTransformDirection(in vec3 dir, in mat4 matrix) {
  return normalize((vec4(dir, 0.0) * matrix).xyz);
}

float viewZToOrthographicDepth(float viewZ) {
    return (viewZ + CameraNear) / (CameraNear - CameraFar);
}

float perspectiveDepthToViewZ(float invClipZ) {
    return (CameraNear*CameraFar) / ((CameraFar-CameraNear)*invClipZ-CameraFar);
}

float getDepth(vec2 screenPosition) {
    return texture2D(DepthSampler, screenPosition).x;
}

float getBackDepth(vec2 screenPosition) {
    return texture2D(BackDepthSampler, screenPosition).x;
}

float getLinearDepth(vec2 screenPosition) {
    float fragCoordZ = texture2D(DepthSampler, screenPosition).x;
    float nz = CameraNear * fragCoordZ;
    return -nz / (CameraFar * (fragCoordZ-1.0) - nz);
}

float getViewZ(float depth) {
    return (CameraNear*CameraFar) / ((CameraFar-CameraNear)*depth-CameraFar);
}

vec3 getViewPosition(vec2 screenPosition, float depth, float viewZ) {
    float clipW = CameraProjectionMatrix[2][3] * viewZ + CameraProjectionMatrix[3][3];
    vec4 clipPosition = vec4((vec3(screenPosition, depth) - 0.5)*2.0, 1.0);
    clipPosition *= clipW; // unprojection
    return (CameraInverseProjectionMatrix * clipPosition).xyz;
}

vec3 getViewNormal(vec2 screenPosition) {
    vec3 rgb = texture2D(NormalSampler, screenPosition).xyz;
    return 2.0*rgb.xyz - 1.0;
}

//----------------------------------------------------------------------------
void swapGEQ(inout float aa, inout float bb)
{
    if (aa > bb)
    {
        float tmp = aa;
        aa = bb;
        bb = tmp;
    }
}

float distanceSquared(vec2 a, vec2 b) { a -= b; return dot(a,a); }

bool rayIntersectsDepthBF(float zA, float zB, vec2 uv)
{
    float cameraZ = getLinearDepth(uv) * -CameraFar;
    // float backZ = getViewZ(getBackDepth(uv)) * -CameraFar;
    float backZ = getLinearDepth(uv) * -CameraFar;
    return zB <= cameraZ && zA >= backZ - PixelZSize;
}

bool rayIntersectsDepth(float zA, float zB, vec2 uv)
{
    float sceneZMax = getViewZ(getDepth(uv));
    float dist = zB - sceneZMax;
    return dist < 0.0 && dist > -PixelZSize;
}

// Trace a ray in screenspace from rayOrg (in camera space) pointing in rayDir (in camera space)
// using jitter to offset the ray based on (jitter * PixelStride).
// 
// Returns true if the ray hits a pixel in the depth buffer
// and outputs the hitPixel (in UV space), the hitPoint (in camera space) and the number
// of iterations it took to get there.
//
// Based on Morgan McGuire & Mike's GLSL implementation:
// http://casual-effects.blogspot.com/2014/08/screen-space-ray-tracing.html
bool traceScreenSpaceRay(
    vec3 rayOrg,
    vec3 rayDir,
    float jitter,
    out vec2 hitPixel,
    out vec3 hitPoint,
    out float iterationCount)
{
    // Clip to the near plane
    float rayLength = ((rayOrg.z + rayDir.z * MaxRayDistance) > -CameraNear) ?
        (-CameraNear - rayOrg.z) / rayDir.z : MaxRayDistance;
    vec3 rayEnd = rayOrg + rayDir * rayLength;

    // Project into homogeneous clip space
    vec4 H0 = CameraProjectionMatrix * vec4(rayOrg, 1.0);
    vec4 H1 = CameraProjectionMatrix * vec4(rayEnd, 1.0);

    float k0 = 1.0 / H0.w, k1 = 1.0 / H1.w;

    // The interpolated homogeneous version of the camera-space points
    vec3 Q0 = rayOrg * k0, Q1 = rayEnd * k1;

    // Screen-space endpoints
    vec2 P0 = H0.xy * k0, P1 = H1.xy * k1;

    // If the line is degenerate, make it cover at least one pixel
    // to avoid handling zero-pixel extent as a special case later
    P1 += (distanceSquared(P0,P1) < 0.0001) ? 0.01 : 0.0;

    vec2 delta = P1 - P0;

    // Permute so that the primary iteration is in x to collapse
    // all quadrant-specific DDA class later
    bool permute = false;
    if (abs(delta.x) < abs(delta.y)) {
        // This is a more-vertical line
        permute = true;
        delta = delta.yx; P0 = P0.yx; P1 = P1.yx;
    }

    float stepDir = sign(delta.x);
    float invdx = stepDir / delta.x;

    // Track the derivatives of Q and K
    vec3  dQ = (Q1 - Q0) * invdx;
    float dk = (k1 - k0) * invdx;
    vec2  dP = vec2(stepDir, delta.y * invdx);

    // Calculate pixel stride based on distance of ray origin from camera.
    // Since perspective means distant objects will be smaller in screen space
    // we can use this to have higher quality reflections for far away objects
    // while still using a large pixel stride for near objects (and increase performance)
    // this also helps mitigate artifacts on distance reflections when we use a large
    // pixel stride.
    float strideScalar = 1.0 - min(1.0, rayOrg.z / PixelStrideZCutoff);
    float pixelStride = 1.0 + strideScalar * PixelStride;
    // pixelStride = 0.0;

    // Scale derivatives by the desired pixel stride and then
    // offset the starting values by the jitter fraction
    dP *= pixelStride; dQ *= pixelStride; dk *= pixelStride;
    P0 += dP * jitter; Q0 += dQ * jitter; k0 += dk * jitter;

    float zA = 0.0, zB = 0.0;
    
    // Track ray step and derivatives in a vec4 to parallelize
    vec4 pqk = vec4(P0, Q0.z, k0);
    vec4 dPQK = vec4(dP, dQ.z, dk);
    bool intersect = false;
    float count = 0.0;

    zB = rayOrg.z;

    for (int i=0; i<MAX_ITERATIONS; i++)
    {
        if (float(i) >= Iterations) break;
        if (intersect) break;

        pqk += dPQK;

        zA = zB;
        zB = (dPQK.z * 0.5 + pqk.z) / (dPQK.w * 0.5 + pqk.w);
        swapGEQ(zB, zA);

        hitPixel = permute ? pqk.yx : pqk.xy;
        hitPixel = hitPixel.xy*0.5+0.5;

        intersect = rayIntersectsDepthBF(zA, zB, hitPixel);
        // intersect = rayIntersectsDepthBF(zA, zB, hitPixel);

        count = float(i);
    }

    // hitPixel = H0.xy / H0.w + dk*jitter;
    // hitPixel = permute ? P0.yx : P0.xy;
    // hitPixel = hitPixel.xy*0.5+0.5;

    // Binary search refinement
    if (pixelStride > 1.0 && intersect)
    {
        pqk -= dPQK;
        dPQK /= pixelStride;

        float originalStride = pixelStride * 0.5;
        float stride = originalStride;

        zA = pqk.z / pqk.w;
        zB = zA;

        for (int j=0; j<MAX_BINARY_SEARCH_ITERATIONS; j++)
        {
            if (float(j) >= BinarySearchIterations) break;

            pqk += dPQK * stride;

            zA = zB;
            zB = (dPQK.z * -0.5 + pqk.z) / (dPQK.w * -0.5 + pqk.w);
            swapGEQ(zB, zA);

            hitPixel = permute ? pqk.yx : pqk.xy;
            hitPixel = hitPixel.xy*0.5+0.5;
            // hitPixel *= Resolution.zw;

            originalStride *= 0.5;
            stride = rayIntersectsDepthBF(zA, zB, hitPixel) ? -originalStride : originalStride;
        }
    }

    Q0.xy += dQ.xy * count;
    Q0.z = pqk.z;
    hitPoint = Q0 / pqk.w;
    iterationCount = count;

    return intersect;
}

bool traceCameraSpaceRay(
    vec3 rayOrg,
    vec3 rayDir,
    float dotNV,
    float jitter,
    out vec2 hitPixel,
    out vec3 hitPoint,
    out float iterationCount)
{
    float rayDist = MaxRayDistance * (1.0 - saturate(dotNV)*step(0.6,dotNV));
    // float rayDist = MaxRayDistance;

    // Clip to the near plane
    float rayLength = ((rayOrg.z + rayDir.z * rayDist) > -CameraNear) ?
        (-CameraNear - rayOrg.z) / rayDir.z : rayDist;
    vec3 rayEnd = rayOrg + rayDir * rayLength;

    vec3 Q0 = rayOrg;
    vec3 Q1 = rayEnd;
    vec3 delta = Q1 - Q0;
    vec3 deltaStep = delta / Iterations;

    vec3 Q = Q0;
    vec2 P;
    float zA, zB = Q0.z;
    
    // Track ray step and derivatives in a vec4 to parallelize
    bool intersect = false;
    float count = 0.0;
    for (int i=0; i<MAX_ITERATIONS; i++)
    {
        if (float(i) >= Iterations) break;
        if (intersect) break;

        Q += deltaStep;
        vec4 clip = CameraProjectionMatrix * vec4(Q,1.0);
        P = clip.xy / clip.w;

        zA = zB;
        zB = Q.z;
        swapGEQ(zB, zA);

        hitPixel = P.xy*0.5+0.5;
        intersect = rayIntersectsDepth(zA, zB, hitPixel);

        count = float(i);
    }

    // hitPixel = H0.xy / H0.w + dk*jitter;
    // hitPixel = permute ? P0.yx : P0.xy;
    // hitPixel = hitPixel.xy*0.5+0.5;
    
    // Binary search refinement
    float pixelStride = PixelStride;
    if (pixelStride > 1.0 && intersect)
    {
        Q -= deltaStep;
        deltaStep /= pixelStride;

        float originalStride = pixelStride * 0.5;
        float stride = originalStride;

        zA = Q.z;
        zB = zA;

        for (int j=0; j<MAX_BINARY_SEARCH_ITERATIONS; j++)
        {
            if (float(j) >= BinarySearchIterations) break;

            Q += deltaStep * stride;
            vec4 clip = CameraProjectionMatrix * vec4(Q,1.0);
            P = clip.xy / clip.w;

            zA = zB;
            zB = Q.z;
            swapGEQ(zB, zA);

            hitPixel = P.xy*0.5+0.5;

            originalStride *= 0.5;
            // stride = rayIntersectsDepthBF(zA, zB, hitPixel) ? -originalStride : originalStride;
            stride = rayIntersectsDepth(zA, zB, hitPixel) ? -originalStride : originalStride;
        }
    }

    hitPoint = Q;
    iterationCount = count;

    return intersect;
}

float calculateAlphaForIntersection(
    bool intersect,
    float iterationCount,
    float specularStrength,
    vec2 hitPixel, 
    vec3 hitPoint,
    vec3 rayOrg,
    vec3 rayDir)
{
    float alpha = min(1.0, specularStrength + 1.0);

    // Fade ray hits test approach the maximum iterations
    alpha *= 1.0 - (iterationCount / Iterations);

    // Fade ray hits that approach the screen edge
    float screenFade = ScreenEdgeFadeStart;
    vec2 hitPixelNDC = (hitPixel * 2.0 - 1.0);
    float maxDistance = min(1.0, max(abs(hitPixelNDC.x), abs(hitPixelNDC.y)));
    alpha *= 1.0 - (max(0.0, maxDistance - screenFade) / (1.0 - screenFade));

    // Fade ray hits base on how much they face the camera
    float eyeFadeStart = EyeFadeStart;
    float eyeFadeEnd = EyeFadeEnd;
    swapGEQ(eyeFadeStart, eyeFadeEnd);

    float eyeDir = clamp(rayDir.z, eyeFadeStart, eyeFadeEnd);
    alpha *= 1.0 - ((eyeDir - eyeFadeStart) / (eyeFadeEnd - eyeFadeStart));

    // Fade ray hits based on distance from ray origin
    alpha *= 1.0 - clamp(distance(rayOrg, hitPoint) / MaxRayDistance, 0.0, 1.0);

    alpha *= intersect ? 1.0 : 0.0;

    return alpha;
}

void main() {
    float depth = getDepth(vUv);
    float viewZ = getViewZ(depth);

    vec3 viewPosition = getViewPosition(vUv, depth, viewZ);
    vec3 viewNormal   = getViewNormal(vUv);

    vec4 metalRoughness = texture2D(MetalRoughSampler, vUv);
    float specularStrength = metalRoughness.y;
    vec4 specRoughPixel = vec4(1.0,1.0,1.0,1.0);

    vec3 rayOrg = viewPosition;
    vec3 rayDir = normalize(reflect(normalize(rayOrg), viewNormal));

    vec2 hitPixel;
    vec3 hitPoint;
    float iterationCount;

    vec2 uv2 = vUv * Resolution.xy;
    float c = (uv2.x + uv2.y) * 0.25;
    float jitter = mod(c, 1.0);
    jitter = 0.0;

    float dotNV = dot(normalize(-rayOrg), viewNormal);

    // bool intersect = traceScreenSpaceRay(rayOrg, rayDir, jitter, hitPixel, hitPoint, iterationCount);
    bool intersect = traceCameraSpaceRay(rayOrg, rayDir, dotNV, jitter, hitPixel, hitPoint, iterationCount);
    // float alpha = calculateAlphaForIntersection(intersect, iterationCount, specularStrength, hitPixel, hitPoint, rayOrg, rayDir);
    hitPixel = mix(vUv, hitPixel, intersect ? 1.0 : 0.0);

    // Comment out the line below to get faked specular,
    // in no way physically correct but will tint based
    // on spec. Physically correct handling of spec is coming...
    specRoughPixel = vec4(1.0,1.0,1.0,1.0);

    vec3 color = texture2D(ColorSampler, hitPixel).xyz;

    // vec3 reflectVec = inverseTransformDirection(rayDir, viewMatrix);
    // vec3 reflectVec = transformDirection(rayDir, ViewInverseMatrix);
    // vec3 query = vec3(-reflectVec.x, reflectVec.yz);
    // vec3 color = textureCube(EnvSampler, query).rgb;

    color = mix(vec3(.3), color, depth < 1.0 ? 1.0 : 0.0);
    // vec3 color = mix(vec3(.3), vec3(intersect ? 1.0 : 0.0), depth < 1.0 ? 1.0 : 0.0);

    float h = max(0.0, hitPoint.y - viewPosition.y);
    float alpha = saturate(1.0 - pow(h / FadeDistance, FadeExponent));
    // float d = length(rayOrg - hitPoint);
    // float alpha = saturate(1.0 - pow(d / FadeDistance, FadeExponent));

    vec2 edgeuv = vUv * (1.0 - vUv.yx);
    float edge = edgeuv.x * edgeuv.y * EdgeDistance;
    edge = saturate(pow(abs(edge), EdgeExponent));
    alpha *= edge;

    alpha *= specularStrength;

    // float uvFactor = 2.0 * length(hitPixel - vec2(0.5,0.5));
    // uvFactor *= uvFactor;
    // float edge = max(0.0, 1.0 - uvFactor);

    gl_FragColor = vec4(color.xyz * specRoughPixel.rgb, alpha);
    // gl_FragColor = vec4(color.xyz * specRoughPixel.rgb, 1.0);
    // gl_FragColor = vec4(rayDir*0.5+0.5, 1.0);
    // gl_FragColor = vec4(hitPixel, 0.0, 1.0);
    // gl_FragColor = vec4(vec3(intersect ? 1.0 : 0.0), 1.0);
    // gl_FragColor = vec4(vec3(jitter), 1.0);
    // gl_FragColor = vec4(vec3(1.0 - -viewZ/(CameraFar-CameraNear)), 1.0);
    // vec4 worldPos = ViewInverseMatrix * vec4(rayOrg, 1.0);
    // gl_FragColor = vec4(worldPos.xyz / (worldPos.w*10.0), 1.0);
    // gl_FragColor = vec4(viewNormal*0.5+0.5, 1.0);
    // gl_FragColor = vec4(vec3(edge), 1.0);
    // gl_FragColor = vec4(vec3(alpha), 1.0);
}
