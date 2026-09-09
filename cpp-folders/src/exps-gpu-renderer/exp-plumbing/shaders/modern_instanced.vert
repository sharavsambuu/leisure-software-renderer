#version 450

layout(location = 0) out vec2 outUV;
layout(location = 1) flat out uint outTexIndex;

struct InstanceData {
    mat4 model;
    uint textureIndex;
    float padding[3];
};

layout(std430, binding = 0) readonly buffer InstanceBuffer {
    InstanceData instances[];
};

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
} pc;

vec3 pos[3] = vec3[](
    vec3(0.0, -0.5, 0.0),
    vec3(0.5, 0.5, 0.0),
    vec3(-0.5, 0.5, 0.0)
);

vec2 uvs[3] = vec2[](
    vec2(0.5, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0)
);

void main() {
    InstanceData data = instances[gl_InstanceIndex];
    gl_Position = pc.viewProj * data.model * vec4(pos[gl_VertexIndex], 1.0);
    outUV = uvs[gl_VertexIndex];
    outTexIndex = data.textureIndex;
}
