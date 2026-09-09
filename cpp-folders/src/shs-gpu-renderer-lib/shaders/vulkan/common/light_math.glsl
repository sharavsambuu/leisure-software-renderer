#ifndef SHS_COMMON_LIGHT_MATH_GLSL
#define SHS_COMMON_LIGHT_MATH_GLSL

float shs_eval_light_attenuation_smoothstep(
    float distance,
    float range,
    uint attenuation_model,
    float attenuation_power,
    float attenuation_bias,
    float attenuation_cutoff)
{
    float safe_range = max(range, 1e-4);
    if (distance >= safe_range)
    {
        return 0.0;
    }

    float norm = clamp(1.0 - distance / safe_range, 0.0, 1.0);
    float falloff = 0.0;
    if (attenuation_model == SHS_LIGHT_ATTEN_LINEAR)
    {
        falloff = norm;
    }
    else if (attenuation_model == SHS_LIGHT_ATTEN_INVERSE_SQUARE)
    {
        float denom = max(distance * distance, attenuation_bias);
        float inv = 1.0 / denom;
        float range_norm = safe_range * safe_range;
        falloff = min(1.0, inv * range_norm) * (norm * norm);
    }
    else
    {
        falloff = norm * norm * (3.0 - 2.0 * norm);
    }

    falloff = pow(max(falloff, 0.0), max(attenuation_power, 0.001));
    if (attenuation_cutoff > 0.0 && falloff < attenuation_cutoff)
    {
        return 0.0;
    }
    return max(falloff, 0.0);
}

float shs_eval_light_attenuation_quadratic(
    float distance,
    float range,
    uint attenuation_model,
    float attenuation_power,
    float attenuation_bias,
    float attenuation_cutoff)
{
    float safe_range = max(range, 1e-4);
    float t = clamp(distance / safe_range, 0.0, 1.0);
    float edge = 1.0 - t;
    float falloff = 0.0;

    if (attenuation_model == SHS_LIGHT_ATTEN_LINEAR)
    {
        falloff = edge;
    }
    else if (attenuation_model == SHS_LIGHT_ATTEN_INVERSE_SQUARE)
    {
        float denom = max(distance * distance, max(attenuation_bias, 1e-5));
        float inv = (safe_range * safe_range) / denom;
        falloff = inv * edge * edge;
    }
    else
    {
        falloff = edge * edge;
    }

    falloff = pow(max(falloff, 0.0), max(attenuation_power, 0.001));
    if (falloff <= max(attenuation_cutoff, 0.0))
    {
        return 0.0;
    }
    return falloff;
}

#endif // SHS_COMMON_LIGHT_MATH_GLSL
