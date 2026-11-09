import bevy_render::{
    view::View,
    maths::powsafe,
};
import package::{
    fullscreen_vertex_shader::FullscreenVertexOutput,
    tonemapping::{tone_mapping, screen_space_dither},
};

@group(0) @binding(0) var<uniform> view: View;

@group(0) @binding(1) var hdr_texture: texture_2d<f32>;
@group(0) @binding(2) var hdr_sampler: sampler;
@group(0) @binding(3) var dt_lut_texture: texture_3d<f32>;
@group(0) @binding(4) var dt_lut_sampler: sampler;

// MPS: `TonemappingFinish`'s sharpening. The luminance of the exposed scene color, bounded to L / (1 + L), so that one
// bright pixel does not rule its neighbors.
// The neighbors are clamped to the camera's viewport, so another camera's pixels or the clear color never take part.
@if(SHARPENING)
fn bounded_luminance(p: vec2<i32>, exposure: f32) -> f32 {
    let first = vec2<i32>(view.viewport.xy);
    let last = first + max(vec2<i32>(view.viewport.zw), vec2<i32>(1)) - 1;
    let c = textureLoad(hdr_texture, clamp(p, first, last), 0).rgb * exposure;
    let l = max(dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0);
    return l / (1.0 + l);
}

// MPS: contrast-adaptive sharpening on the cross (AMD's CAS, sharpen only) by that luminance; the returned gain scales
// the color, so it keeps its ratios. Low contrast takes none, where flat areas hold noise the clamp alone would amplify.
@if(SHARPENING)
fn sharpening_gain(p: vec2<i32>) -> f32 {
    // The pass covers the whole target: pixels outside the camera's viewport keep their color.
    let first = vec2<i32>(view.viewport.xy);
    if any(p < first) || any(p >= first + vec2<i32>(view.viewport.zw)) {
        return 1.0;
    }
    let exposure = exp2(view.color_grading.exposure);
    let c = bounded_luminance(p, exposure);
    let n = bounded_luminance(p + vec2<i32>(0, -1), exposure);
    let s = bounded_luminance(p + vec2<i32>(0, 1), exposure);
    let e = bounded_luminance(p + vec2<i32>(1, 0), exposure);
    let w = bounded_luminance(p + vec2<i32>(-1, 0), exposure);
    let lo = min(c, min(min(n, s), min(e, w)));
    let hi = max(c, max(max(n, s), max(e, w)));
    let headroom = sqrt(saturate(min(lo, 1.0 - hi) / max(hi, 1e-6)));
    let contrast = smoothstep(0.02, 0.08, (hi - lo) / max(hi, 0.02));
    let k = -0.20 * (f32(constants::SHARPENING_MILLI) / 1000.0) * headroom * contrast;
    let out = min(clamp((k * (n + s + e + w) + c) / (4.0 * k + 1.0), lo, hi), 1.0 - 1e-6);
    // Back through the bound: the ratio of the two luminances.
    return (out / (1.0 - out)) * (1.0 - c) / max(c, 1e-6);
}

// MPS: `TonemappingFinish`'s monochrome triangular noise fixed to the pixel, one 1/255 step at its peak: the sum of two
// uniform values (pcg2d).
@if(TRIANGULAR_DITHER)
fn triangular_dither(p: vec2<u32>) -> f32 {
    var h = p * 1664525u + 1013904223u;
    h.x += h.y * 1664525u;
    h.y += h.x * 1664525u;
    h ^= h >> vec2<u32>(16u);
    h.x += h.y * 1664525u;
    h.y += h.x * 1664525u;
    h ^= h >> vec2<u32>(16u);
    return ((f32(h.x) + f32(h.y)) / 4294967296.0 - 1.0) / 255.0;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    var hdr_color = textureSample(hdr_texture, hdr_sampler, in.uv);

@if(SHARPENING) {
    hdr_color = vec4<f32>(hdr_color.rgb * sharpening_gain(vec2<i32>(in.position.xy)), hdr_color.a);
}

    var output_rgb = tone_mapping(hdr_color, view.color_grading).rgb;

@if(TRIANGULAR_DITHER) {
    // MPS: the 8-bit target rounds the sRGB-encoded value, so the noise goes there; the target encodes again.
    let encoded = select(1.055 * pow(max(output_rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055, output_rgb * 12.92,
                         output_rgb <= vec3<f32>(0.0031308));
    let dithered = saturate(saturate(encoded) + triangular_dither(vec2<u32>(in.position.xy)));
    output_rgb = select(pow((dithered + 0.055) / 1.055, vec3<f32>(2.4)), dithered / 12.92,
                        dithered <= vec3<f32>(0.04045));
} @elif(DEBAND_DITHER) {
    output_rgb = powsafe(output_rgb.rgb, 1.0 / 2.2);
    output_rgb = output_rgb + screen_space_dither(in.position.xy);
    // This conversion back to linear space is required because our output texture format is
    // SRGB; the GPU will assume our output is linear and will apply an SRGB conversion.
    output_rgb = powsafe(output_rgb.rgb, 2.2);
}

    return vec4<f32>(output_rgb, hdr_color.a);
}
