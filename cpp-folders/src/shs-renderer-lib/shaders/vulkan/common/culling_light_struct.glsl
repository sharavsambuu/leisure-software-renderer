#ifndef SHS_COMMON_CULLING_LIGHT_STRUCT_GLSL
#define SHS_COMMON_CULLING_LIGHT_STRUCT_GLSL

struct CullingLightGPU
{
    vec4 position_range;
    vec4 color_intensity;
    vec4 direction_spot;
    vec4 axis_spot_outer;
    vec4 up_shape_x;
    vec4 shape_attenuation;
    uvec4 type_shape_flags;
    vec4 cull_sphere;
    vec4 cull_aabb_min;
    vec4 cull_aabb_max;
};

#endif // SHS_COMMON_CULLING_LIGHT_STRUCT_GLSL
