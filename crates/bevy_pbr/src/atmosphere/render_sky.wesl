import bevy_render::maths::orthonormalize;

import package::atmosphere::{
    bindings::{view, settings, lights},
    bruneton_functions::sample_transmittance_lut_segment,
    functions::{
        sample_transmittance_lut,
        sample_sky_view_lut, direction_world_to_atmosphere,
        uv_to_ray_direction, uv_to_ndc, ndc_to_uv, sample_aerial_view_lut,
        sample_sun_radiance, sun_glare_factor, sun_glare_profile, ndc_to_camera_dist,
        raymarch_atmosphere,
        get_view_position, max_atmosphere_distance
    },
};

@if(DUAL_SOURCE_BLENDING)
enable dual_source_blending;

@if(MULTISAMPLED)
@group(0) @binding(13) var depth_texture: texture_depth_multisampled_2d;
@else
@group(0) @binding(13) var depth_texture: texture_depth_2d;

/// The largest exposed value of the sun's disk with the glare on and no highlight roll-off, which
/// keeps it within the range of half-precision render targets.
const SUN_DISK_MAX_EXPOSED: f32 = 4096.0;

/// The number of depth taps over each sun disk that estimate how much of it geometry covers.
const SUN_VISIBILITY_TAPS: u32 = 32u;

/// The golden angle in radians: the angular step between the taps of the spiral over the disk.
const GOLDEN_ANGLE: f32 = 2.39996323;

struct RenderSkyVertexOutput {
    @builtin(position)
    position: vec4<f32>,
    @location(0)
    uv: vec2<f32>,
    /// The glare's factor of the first directional light (`sun_glare_factor` times the share of
    /// its disk that no geometry covers), the same for the whole view.
    @if(SUN_GLARE)
    @location(1) @interpolate(flat)
    sun_glare: vec3<f32>,
}

struct RenderSkyOutput {
    @if(DUAL_SOURCE_BLENDING)
    @location(0) @blend_src(0) inscattering: vec4<f32>,
    @if(DUAL_SOURCE_BLENDING)
    @location(0) @blend_src(1) transmittance: vec4<f32>,
    @else
    @location(0) inscattering: vec4<f32>,
}

/// Draws the fullscreen triangle of `fullscreen_vertex_shader`, and computes the glare's factor of
/// the first directional light once for the view (the fragments take it flat): its light
/// reaching the viewer and how much of its disk geometry covers. Further lights draw no glare:
/// the same term for the first four lights cost about 0.8 ms more a frame on an integrated GPU
/// (Intel Raptor Lake), and no scene has a second sun.
@if(SUN_GLARE)
@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> RenderSkyVertexOutput {
    let uv = vec2<f32>(f32(vertex_index >> 1u), f32(vertex_index & 1u)) * 2.0;
    let clip_position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);

    var sun_glare = vec3(0.0);
    if lights.n_directional_lights > 0u {
        let factor = sun_glare_factor(0u);
        if any(factor > vec3(0.0)) {
            sun_glare = factor * sun_disk_visibility(0u);
        }
    }
    return RenderSkyVertexOutput(clip_position, uv, sun_glare);
}

/// The share of a light's sun disk that no geometry covers: the share of depth taps on a spiral
/// over the disk that find the far plane (0 with reverse Z). Taps outside the view are clamped
/// to its edge; taps behind the camera count as uncovered.
@if(SUN_GLARE)
fn sun_disk_visibility(light_i: u32) -> f32 {
    let light = &lights.directional_lights[light_i];
    let disk_radius = (*light).sun_disk_angular_size * 0.5;
    let basis = orthonormalize((*light).direction_to_light);
    let viewport = view.main_pass_viewport;
    let last_pixel = viewport.xy + viewport.zw - 1.0;

    var visible = 0.0;
    for (var tap_i: u32 = 0u; tap_i < SUN_VISIBILITY_TAPS; tap_i++) {
        // Equal areas of the disk per tap.
        let offset = tan(disk_radius * sqrt((f32(tap_i) + 0.5) / f32(SUN_VISIBILITY_TAPS)));
        let angle = f32(tap_i) * GOLDEN_ANGLE;
        let direction = basis * vec3(offset * cos(angle), offset * sin(angle), 1.0);
        let clip = view.clip_from_world * vec4(direction, 0.0);
        if clip.w <= 0.0 {
            visible += 1.0;
            continue;
        }
        let pixel = clamp(viewport.xy + ndc_to_uv(clip.xy / clip.w) * viewport.zw, viewport.xy, last_pixel);
        let depth = textureLoad(depth_texture, vec2<i32>(pixel), 0);
        visible += f32(depth == 0.0);
    }
    return visible / f32(SUN_VISIBILITY_TAPS);
}

@fragment
fn main(in: RenderSkyVertexOutput) -> RenderSkyOutput {
    let depth = textureLoad(depth_texture, vec2<i32>(in.position.xy), 0);

    let ray_dir_ws = uv_to_ray_direction(in.uv);
    let world_pos = get_view_position();
    let r = length(world_pos);
    let up = normalize(world_pos);
    let mu = dot(ray_dir_ws, up);
    let max_samples = settings.sky_max_samples;
    let should_raymarch = settings.rendering_method == 1u;

    var transmittance: vec3<f32>;
    var inscattering: vec3<f32>;

    let sun_radiance = sample_sun_radiance(ray_dir_ws);

    if depth == 0.0 {
        let ray_dir_as = direction_world_to_atmosphere(ray_dir_ws);
        transmittance = sample_transmittance_lut(r, mu);
        inscattering = sample_sky_view_lut(r, ray_dir_as);
        if should_raymarch {
            let t_max = max_atmosphere_distance(r, mu);
            let result = raymarch_atmosphere(world_pos, ray_dir_ws, t_max, max_samples, in.uv, true);
            inscattering = result.inscattering;
            transmittance = result.transmittance;
        }

        // The sun's own light: its disk and, with `SUN_GLARE`, the glare around it.
        // With the glare, the disk's transmittance comes from the table toward the pixel, not from
        // the march (which stops below 0.001), so that a disk behind thick dust dims away.
        var disk_transmittance = transmittance;
        @if(SUN_GLARE) {
            disk_transmittance = sample_transmittance_lut(r, mu);
        }
        var sun_light = sun_radiance * disk_transmittance;
        let knee = settings.sun_highlight_knee;
        @if(SUN_GLARE) {
            if knee <= 0.0 {
                let exposed_peak = max(sun_light.r, max(sun_light.g, sun_light.b)) * view.exposure;
                if exposed_peak > SUN_DISK_MAX_EXPOSED {
                    sun_light *= SUN_DISK_MAX_EXPOSED / exposed_peak;
                }
            }
            if lights.n_directional_lights > 0u {
                let sun = &lights.directional_lights[0];
                sun_light += in.sun_glare
                    * sun_glare_profile((*sun).direction_to_light, ray_dir_ws, (*sun).sun_disk_angular_size);
            }
        }
        if knee > 0.0 {
            // The response of the film to the total light (sky and sun), per channel,
            // `knee x / (knee + x)`; the sun keeps the part above the sky's own response, so the
            // sky stays as it is and the sun's core goes to white (the knee) while its falloff
            // keeps the sun's and the air's color.
            let sky = inscattering * view.exposure;
            let total = sky + sun_light * view.exposure;
            sun_light = (knee * total / (knee + total) - knee * sky / (knee + sky)) / view.exposure;
        }
        inscattering += sun_light;
    } else {
        let t = ndc_to_camera_dist(vec3(uv_to_ndc(in.uv), depth));
        inscattering = sample_aerial_view_lut(in.uv, t);
        transmittance = sample_transmittance_lut_segment(r, mu, t);
        if should_raymarch {
            let result = raymarch_atmosphere(world_pos, ray_dir_ws, t, max_samples, in.uv, false);
            inscattering = result.inscattering;
            transmittance = result.transmittance;
        }
    }

    // exposure compensation
    inscattering *= view.exposure;

    @if(DUAL_SOURCE_BLENDING)
    return RenderSkyOutput(vec4(inscattering, 0.0), vec4(transmittance, 1.0));
    @if(!DUAL_SOURCE_BLENDING)
    let mean_transmittance = (transmittance.r + transmittance.g + transmittance.b) / 3.0;
    @if(!DUAL_SOURCE_BLENDING)
    return RenderSkyOutput(vec4(inscattering, mean_transmittance));

}
