#version 450

layout(set = 0, binding = 0) uniform CameraUBO
{
    mat4 view_proj;
    vec4 camera_pos;
    vec4 light_dir_ws;
} ubo;

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal_ws;
layout(location = 2) in vec3 v_base_color;
layout(location = 3) flat in uint v_lit_mode;

layout(location = 0) out vec4 out_color;

void main()
{
    vec3 base = clamp(v_base_color, vec3(0.0), vec3(1.0));
    if (v_lit_mode == 0u)
    {
        out_color = vec4(base, 1.0);
        return;
    }

    vec3 N = normalize(v_normal_ws);
    vec3 L = normalize(-ubo.light_dir_ws.xyz);
    vec3 V = normalize(ubo.camera_pos.xyz - v_world_pos);
    vec3 H = normalize(L + V);

    float ndotl = max(dot(N, L), 0.0);
    float ndoth = max(dot(N, H), 0.0);

    float ambient = 0.18;
    float diffuse = 0.72 * ndotl;
    float specular = (ndotl > 0.0) ? (0.35 * pow(ndoth, 32.0)) : 0.0;

    vec3 lit = clamp(base * (ambient + diffuse) + vec3(specular), vec3(0.0), vec3(1.0));
    out_color = vec4(lit, 1.0);
}
