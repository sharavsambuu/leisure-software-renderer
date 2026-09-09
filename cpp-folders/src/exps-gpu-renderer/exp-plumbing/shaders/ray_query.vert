#version 460
layout(location = 0) in vec3 inPosition;
layout(location = 0) out vec3 outWorldPos;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 view;
    mat4 proj;
} pcs;

void main() {
    vec4 worldPos = pcs.model * vec4(inPosition, 1.0);
    outWorldPos = worldPos.xyz;
    gl_Position = pcs.proj * pcs.view * worldPos;
}
