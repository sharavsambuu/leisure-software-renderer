#version 460
#extension GL_EXT_ray_query : enable

layout(location = 0) in vec3 inWorldPos;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 lightPos;
} pcs;

void main() {
    vec3 lightDir = normalize(pcs.lightPos - inWorldPos);
    float tMin = 0.001;
    float tMax = distance(pcs.lightPos, inWorldPos);

    // Dynamic ray query
    rayQueryEXT query;
    rayQueryInitializeEXT(query, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, inWorldPos, tMin, lightDir, tMax);
    
    while(rayQueryProceedEXT(query)) {
        // We could handle opacity or other states here
    }

    float shadow = 1.0;
    if (rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
        shadow = 0.2; // In shadow
    }

    vec3 color = vec3(0.8, 0.7, 0.6) * shadow;
    outColor = vec4(color, 1.0);
}
