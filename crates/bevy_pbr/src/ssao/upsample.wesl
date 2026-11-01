//! Four-tap joint bilateral upsample of half-resolution visibility.

import bevy_render::view::View;

@group(0) @binding(0) var ambient_occlusion_half: texture_2d<f32>;
@group(0) @binding(1) var depth: texture_depth_2d;
@group(0) @binding(2) var normals: texture_2d<f32>;
@if(USE_R16FLOAT)
@group(0) @binding(3) var ambient_occlusion: texture_storage_2d<r16float, write>;
@else
@group(0) @binding(3) var ambient_occlusion: texture_storage_2d<r32float, write>;
@group(1) @binding(0) var point_clamp_sampler: sampler;
@group(1) @binding(1) var linear_clamp_sampler: sampler;
@group(1) @binding(2) var<uniform> view: View;

fn view_depth(value: f32, pixel_coordinates: vec2<i32>) -> f32 {
    let uv = (vec2<f32>(pixel_coordinates) + 0.5) / view.main_pass_viewport.zw;
    let position = view.view_from_clip * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, value, 1.0);
    return abs(position.z / position.w);
}

@compute
@workgroup_size(8, 8, 1)
fn upsample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= textureDimensions(ambient_occlusion)) {
        return;
    }
    let pixel_coordinates = vec2<i32>(global_id.xy);
    let viewport_origin = vec2<i32>(view.main_pass_viewport.xy);
    let center_depth = textureLoad(depth, pixel_coordinates + viewport_origin, 0);
    if center_depth == 0.0 {
        textureStore(ambient_occlusion, pixel_coordinates, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }
    let center_z = view_depth(center_depth, pixel_coordinates);
    let center_normal = normalize(textureLoad(normals, pixel_coordinates + viewport_origin, 0).xyz * 2.0 - 1.0);
    let half_position = vec2<f32>(pixel_coordinates) * 0.5;
    let base = vec2<i32>(floor(half_position));
    let fraction = fract(half_position);
    let last_pixel = vec2<i32>(textureDimensions(ambient_occlusion_half)) - 1i;

    var visibility = 0.0;
    var weight_sum = 0.0;
    var closest_visibility = 1.0;
    var closest_difference = 3.402823466e+38;
    for (var y = 0i; y < 2i; y += 1i) {
        for (var x = 0i; x < 2i; x += 1i) {
            let half_coordinates = min(base + vec2<i32>(x, y), last_pixel);
            // MPS: Match the depth and normal used to trace this half-resolution texel.
            let sample_coordinates = half_coordinates * 2i;
            let sample_depth = textureLoad(depth, sample_coordinates + viewport_origin, 0);
            if sample_depth == 0.0 {
                continue;
            }
            let sample_normal = normalize(textureLoad(normals, sample_coordinates + viewport_origin, 0).xyz * 2.0 - 1.0);
            let depth_difference = abs(view_depth(sample_depth, sample_coordinates) - center_z) / max(center_z, 0.0001);
            let normal_similarity = saturate(dot(center_normal, sample_normal));
            // MPS: Tiny weights still normalize to full AO from an unrelated surface.
            if depth_difference > 0.05 || normal_similarity < 0.9 {
                continue;
            }
            let spatial_weight = select(1.0 - fraction.x, fraction.x, x == 1i)
                * select(1.0 - fraction.y, fraction.y, y == 1i);
            let weight = spatial_weight * exp2(-64.0 * depth_difference) * pow(normal_similarity, 8.0);
            let sample_visibility = textureLoad(ambient_occlusion_half, half_coordinates, 0).r;
            visibility += sample_visibility * weight;
            weight_sum += weight;

            let difference = 64.0 * depth_difference + 8.0 * (1.0 - normal_similarity);
            if difference < closest_difference {
                closest_difference = difference;
                closest_visibility = sample_visibility;
            }
        }
    }

    // MPS: Fall back only to a compatible tap, or leave an unmatched silhouette unoccluded.
    var result = closest_visibility;
    if weight_sum > 0.0 {
        result = visibility / weight_sum;
    }
    textureStore(ambient_occlusion, pixel_coordinates, vec4<f32>(result, 0.0, 0.0, 0.0));
}
