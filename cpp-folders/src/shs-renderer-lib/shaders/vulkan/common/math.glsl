#ifndef SHS_COMMON_MATH_GLSL
#define SHS_COMMON_MATH_GLSL

vec3 shs_normalize_or(vec3 v, vec3 fallback)
{
    float n2 = dot(v, v);
    if (n2 <= 1e-10)
    {
        return fallback;
    }
    return v * inversesqrt(n2);
}

#endif // SHS_COMMON_MATH_GLSL
