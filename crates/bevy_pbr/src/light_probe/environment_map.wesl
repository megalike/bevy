import package::light_probe::{light_probe_iterator_new, light_probe_iterator_next, LightProbeQueryResult};
import package::render::mesh_view_bindings as bindings;
import package::render::mesh_view_bindings::light_probes;
import package::render::mesh_view_types::{
    LIGHT_PROBE_FLAG_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE, LIGHT_PROBE_FLAG_PARALLAX_CORRECT
};
import package::render::pbr_lighting::{F_Schlick_vec, LightingInput, LayerLightingInput, LAYER_BASE, LAYER_CLEARCOAT};
import package::render::clustered_forward::ClusterableObjectIndexRanges;
import bevy_render::maths::quat_rotate;

/// The maximum representable value in a 32-bit floating point number.
const FLOAT_MAX: f32 = 3.40282347e+38;

struct EnvironmentMapLight {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
};

struct EnvironmentMapRadiances {
    irradiance: vec3<f32>,
    radiance: vec3<f32>,
}

struct MultiscatterResult {
    FssEss: vec3<f32>,
    FmsEms: vec3<f32>,
    Edss: vec3<f32>,
}

/// Computes the direction at which to sample the reflection probe.
///
/// * `light_from_world` is the matrix that transforms world space into light
///   probe space (a 1×1×1 cube centered on the origin).
///
/// * `light_from_world_rotation` is the rotation part of that transform, as a
///   quaternion. The sample direction is formed in world space and rotated with
///   it alone: the full transform would scale it by the light probe's extents,
///   which distorts directions in a light probe that isn't a cube.
///
/// * `capture_origin` is the world-space point from which the cubemap was
///   captured. It's ignored if the `parallax_correct` parameter is false.
///
/// * `parallax_correction_bounds` is the half-extents of the simulated
///   reflection boundaries used for parallax correction, in light probe space.
///   It's ignored if the `parallax_correct` parameter is false.
///
/// * `parallax_correct` is true if parallax correction is to be applied and
///   false otherwise.
fn compute_cubemap_sample_dir(
    world_ray_origin: vec3<f32>,
    world_ray_direction: vec3<f32>,
    light_from_world: mat4x4<f32>,
    light_from_world_rotation: vec4<f32>,
    capture_origin: vec3<f32>,
    parallax_correction_bounds: vec3<f32>,
    parallax_correct: bool,
    view_rotation: vec4<f32>,
) -> vec3<f32> {
    var sample_dir: vec3<f32>;

    // If we're supposed to parallax correct, then intersect with the light cube.
    if (parallax_correct) {
        // Compute the direction of the ray bouncing off the surface, in light
        // probe space.
        // Recall that light probe space is a 1×1×1 cube centered at the origin.
        let ray_origin = (light_from_world * vec4(world_ray_origin, 1.0)).xyz;
        let ray_direction = (light_from_world * vec4(world_ray_direction, 0.0)).xyz;

        // Solve for the intersection of that ray with each side of the cube.
        // Since our light probe is a 1×1×1 cube centered at the origin in light
        // probe space, the faces of the cube are at X = ±0.5, Y = ±0.5, and Z =
        // ±0.5.
        var t0 = (-parallax_correction_bounds - ray_origin) / ray_direction;
        var t1 = (parallax_correction_bounds - ray_origin) / ray_direction;

        // We're shooting the rays forward, so we need to rule out negative time
        // values. So, if t is negative, make it a large value so that we won't
        // choose it below.
        // We would use infinity here but WGSL forbids it:
        // https://github.com/gfx-rs/wgpu/issues/5515
        t0 = select(vec3(FLOAT_MAX), t0, t0 >= vec3(0.0));
        t1 = select(vec3(FLOAT_MAX), t1, t1 >= vec3(0.0));

        // Choose the minimum valid time value to find the intersection of the
        // first cube face.
        let t_min = min(t0, t1);
        let t = min(min(t_min.x, t_min.y), t_min.z);

        // The transform is affine, so the ray reaches the face at the same `t`
        // in world space. Sample the cubemap in the direction from the capture
        // point to that hit, in the light probe's frame. (It doesn't have to be
        // normalized.)
        let world_hit = world_ray_origin + world_ray_direction * t;
        sample_dir = quat_rotate(light_from_world_rotation, world_hit - capture_origin);
    } else {
        // We treat the reflection as infinitely far away in the non-parallax
        // case, so the ray origin is irrelevant.
        sample_dir = quat_rotate(light_from_world_rotation, world_ray_direction);
    }

    // Rotating the world space ray direction by the environment map light view rotation,
    // it is equivalent to rotating the environment cubemap itself.
    sample_dir = quat_rotate(view_rotation, sample_dir);

    // Cubemaps are left-handed, so we negate the Z coordinate.
    sample_dir.z = -sample_dir.z;
    return sample_dir;
}

// Define two versions of this function, one for the case in which there are
// multiple light probes and one for the case in which only the view light probe
// is present.

@if(MULTIPLE_LIGHT_PROBES_IN_ARRAY)

/// The radiance and irradiance one reflection probe (or the view's environment
/// map, with `view_rotation`) contributes to a fragment, before weighting.
fn sample_reflection_probe(
    query_result: LightProbeQueryResult,
    view_rotation: vec4<f32>,
    world_position: vec3<f32>,
    N: vec3<f32>,
    radiance_direction: vec3<f32>,
    perceptual_roughness: f32,
    enable_diffuse: bool,
) -> EnvironmentMapRadiances {
    var radiances: EnvironmentMapRadiances;
    radiances.irradiance = vec3(0.0);
    radiances.radiance = vec3(0.0);

    // Split-sum approximation for image based lighting: https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
    let radiance_level = perceptual_roughness * f32(textureNumLevels(
        bindings::specular_environment_maps[query_result.texture_index]) - 1u);

    var enable_probe_diffuse = enable_diffuse;
    @if(LIGHTMAP)
    enable_probe_diffuse = enable_probe_diffuse &&
        (query_result.flags & LIGHT_PROBE_FLAG_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE) != 0u;

    let parallax_correct = (query_result.flags & LIGHT_PROBE_FLAG_PARALLAX_CORRECT) != 0u;

    if (enable_probe_diffuse) {
        let irradiance_sample_dir = compute_cubemap_sample_dir(
            world_position,
            N,
            query_result.light_from_world,
            query_result.light_from_world_rotation,
            query_result.capture_origin,
            query_result.parallax_correction_bounds,
            parallax_correct,
            view_rotation,
        );
        radiances.irradiance = textureSampleLevel(
            bindings::diffuse_environment_maps[query_result.texture_index],
            bindings::environment_map_sampler,
            irradiance_sample_dir,
            0.0).rgb * query_result.intensity;
    }

    let radiance_sample_dir = compute_cubemap_sample_dir(
        world_position,
        radiance_direction,
        query_result.light_from_world,
        query_result.light_from_world_rotation,
        query_result.capture_origin,
        query_result.parallax_correction_bounds,
        parallax_correct,
        view_rotation,
    );
    radiances.radiance = textureSampleLevel(
        bindings::specular_environment_maps[query_result.texture_index],
        bindings::environment_map_sampler,
        radiance_sample_dir,
        radiance_level).rgb * query_result.intensity;

    return radiances;
}

@if(MULTIPLE_LIGHT_PROBES_IN_ARRAY)
fn compute_radiances(
    input: LayerLightingInput,
    clusterable_object_index_ranges: ptr<function, ClusterableObjectIndexRanges>,
    world_position: vec3<f32>,
    found_diffuse_indirect: bool,
) -> EnvironmentMapRadiances {
    // Unpack.
    let N = input.N;
    let R = input.R;
    let perceptual_roughness = input.perceptual_roughness;
    let roughness = input.roughness;

    var radiances: EnvironmentMapRadiances;
    radiances.irradiance = vec3(0.0);
    radiances.radiance = vec3(0.0);

    // If we're lightmapped, and we shouldn't accumulate diffuse light from the
    // environment map, note that.
    var enable_diffuse = !found_diffuse_indirect;

    // If the view takes its diffuse light from its own environment map alone,
    // sample that once here; the reflection probes below then supply only the
    // specular term.
    let diffuse_from_view = light_probes.diffuse_from_view != 0u;
    if (diffuse_from_view) {
        var enable_view_diffuse = enable_diffuse && light_probes.view_cubemap_index >= 0;
        @if(LIGHTMAP)
        enable_view_diffuse = enable_view_diffuse &&
            light_probes.view_environment_map_affects_lightmapped_mesh_diffuse != 0u;
        if (enable_view_diffuse) {
            var irradiance_sample_dir = quat_rotate(light_probes.view_rotation, N);
            // Cubemaps are left-handed, so we negate the Z coordinate.
            irradiance_sample_dir.z = -irradiance_sample_dir.z;
            radiances.irradiance = textureSampleLevel(
                bindings::diffuse_environment_maps[light_probes.view_cubemap_index],
                bindings::environment_map_sampler,
                irradiance_sample_dir,
                0.0).rgb * light_probes.intensity_for_view;
        }
        enable_diffuse = false;
    }

    let radiance_direction = radiance_sample_direction(N, R, roughness);
    let identity_rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    // Find all reflection probes that contain the fragment. Reflection probes
    // of one priority blend in a weighted sum; where reflection probes of a
    // higher priority have weight `w` (at most 1), those of lower priorities
    // supply only the residual share `1 - w` (see `ReflectionProbePriority`).
    // The first pass accumulates the highest priority present at this
    // fragment (a higher one met on the way discards what lower ones gave);
    // each later pass accumulates the next priority down, only while the
    // higher ones left a share, and the view environment map fills what
    // remains. This allows for e.g. nice transitions between the interior of
    // a building and the outdoor environment map, and between a room and the
    // hall around it. With equal priorities everything happens in one pass.
    var iterator = light_probe_iterator_new(
        world_position,
        /*is_irradiance_volume=*/ false,
        clusterable_object_index_ranges,
    );
    var have_level = false;
    var level_priority: i32 = 0;
    var level_irradiance = vec3(0.0);
    var level_radiance = vec3(0.0);
    var level_weight = 0.0;
    var have_next = false;
    var next_priority: i32 = 0;
    while (true) {
        let query_result = light_probe_iterator_next(&iterator);
        if (query_result.texture_index < 0) {
            break;
        }
        if (have_level && query_result.priority < level_priority) {
            if (!have_next || query_result.priority > next_priority) {
                next_priority = query_result.priority;
                have_next = true;
            }
            continue;
        }
        if (!have_level || query_result.priority > level_priority) {
            // A higher priority than everything so far: that becomes the
            // next level down.
            have_next = have_level;
            next_priority = level_priority;
            have_level = true;
            level_priority = query_result.priority;
            level_irradiance = vec3(0.0);
            level_radiance = vec3(0.0);
            level_weight = 0.0;
        }
        let probe = sample_reflection_probe(
            query_result,
            identity_rotation,
            world_position,
            N,
            radiance_direction,
            perceptual_roughness,
            enable_diffuse,
        );
        level_irradiance += probe.irradiance * query_result.weight;
        level_radiance += probe.radiance * query_result.weight;
        level_weight += query_result.weight;
    }

    // The share of the environment light not yet taken by higher priorities.
    var remaining = 1.0;
    // At most one pass per reflection probe in the view.
    for (var level_pass = 0u; level_pass < 8u && have_level; level_pass++) {
        if (level_pass > 0u) {
            // Accumulate the next priority down.
            level_priority = next_priority;
            have_next = false;
            level_irradiance = vec3(0.0);
            level_radiance = vec3(0.0);
            level_weight = 0.0;
            iterator = light_probe_iterator_new(
                world_position,
                /*is_irradiance_volume=*/ false,
                clusterable_object_index_ranges,
            );
            while (true) {
                let query_result = light_probe_iterator_next(&iterator);
                if (query_result.texture_index < 0) {
                    break;
                }
                if (query_result.priority == level_priority) {
                    let probe = sample_reflection_probe(
                        query_result,
                        identity_rotation,
                        world_position,
                        N,
                        radiance_direction,
                        perceptual_roughness,
                        enable_diffuse,
                    );
                    level_irradiance += probe.irradiance * query_result.weight;
                    level_radiance += probe.radiance * query_result.weight;
                    level_weight += query_result.weight;
                } else if (query_result.priority < level_priority &&
                        (!have_next || query_result.priority > next_priority)) {
                    next_priority = query_result.priority;
                    have_next = true;
                }
            }
        }
        // Normalized within the level when its weights exceed 1, and scaled
        // by the share the higher priorities left.
        let share = remaining / max(level_weight, 1.0);
        radiances.irradiance += level_irradiance * share;
        radiances.radiance += level_radiance * share;
        remaining *= 1.0 - min(level_weight, 1.0);
        if (!have_next || remaining <= 0.0001) {
            break;
        }
    }

    if (remaining > 0.0001) {
        if (light_probes.view_cubemap_index >= 0) {
            // The view environment map fills what the reflection probes left.
            var view_result: LightProbeQueryResult;
            view_result.texture_index = light_probes.view_cubemap_index;
            view_result.intensity = light_probes.intensity_for_view;
            view_result.light_from_world = mat4x4(
                vec4(1.0, 0.0, 0.0, 0.0),
                vec4(0.0, 1.0, 0.0, 0.0),
                vec4(0.0, 0.0, 1.0, 0.0),
                vec4(0.0, 0.0, 0.0, 1.0)
            );
            view_result.parallax_correction_bounds = vec3(0.0);
            view_result.light_from_world_rotation = identity_rotation;
            view_result.capture_origin = vec3(0.0);
            if light_probes.view_environment_map_affects_lightmapped_mesh_diffuse != 0u {
                view_result.flags = LIGHT_PROBE_FLAG_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE;
            } else {
                view_result.flags = 0u;
            }
            view_result.weight = remaining;
            view_result.priority = 0;
            let view = sample_reflection_probe(
                view_result,
                light_probes.view_rotation,
                world_position,
                N,
                radiance_direction,
                perceptual_roughness,
                enable_diffuse,
            );
            radiances.irradiance += view.irradiance * remaining;
            radiances.radiance += view.radiance * remaining;
        } else if (remaining < 1.0) {
            // No view environment map: the reflection probes alone light the
            // fragment, renormalized as if their weights summed to 1.
            let taken = 1.0 - remaining;
            radiances.radiance /= taken;
            if (!diffuse_from_view) {
                radiances.irradiance /= taken;
            }
        }
    }

    return radiances;
}

@else   // MULTIPLE_LIGHT_PROBES_IN_ARRAY

fn compute_radiances(
    input: LayerLightingInput,
    clusterable_object_index_ranges: ptr<function, ClusterableObjectIndexRanges>,
    world_position: vec3<f32>,
    found_diffuse_indirect: bool,
) -> EnvironmentMapRadiances {
    // Unpack.
    let N = input.N;
    let R = input.R;
    let perceptual_roughness = input.perceptual_roughness;
    let roughness = input.roughness;

    var radiances: EnvironmentMapRadiances;

    // If we have no light probe, bail.
    if (light_probes.view_cubemap_index < 0) {
        radiances.irradiance = vec3(0.0);
        radiances.radiance = vec3(0.0);
        return radiances;
    }

    // Split-sum approximation for image based lighting: https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
    // Technically we could use textureNumLevels(specular_environment_map) - 1 here, but we use a uniform
    // because textureNumLevels() does not work on WebGL2
    let radiance_level = perceptual_roughness * f32(light_probes.smallest_specular_mip_level_for_view);

    let intensity = light_probes.intensity_for_view;

    // If we're lightmapped, and we shouldn't accumulate diffuse light from the
    // environment map, note that.
    var enable_diffuse = !found_diffuse_indirect;
    @if(LIGHTMAP)
    enable_diffuse = enable_diffuse &&
        light_probes.view_environment_map_affects_lightmapped_mesh_diffuse;

    if (enable_diffuse) {
        var irradiance_sample_dir = N;
        // Rotating the world space ray direction by the environment map light view rotation,
        // it is equivalent to rotating the environment cubemap itself.
        irradiance_sample_dir = quat_rotate(light_probes.view_rotation, irradiance_sample_dir);
        // Cube maps are left-handed so we negate the z coordinate.
        irradiance_sample_dir.z = -irradiance_sample_dir.z;
        radiances.irradiance = textureSampleLevel(
            bindings::diffuse_environment_map,
            bindings::environment_map_sampler,
            irradiance_sample_dir,
            0.0).rgb * intensity;
    }

    var radiance_sample_dir = radiance_sample_direction(N, R, roughness);
    // Rotating the world space ray direction by the environment map light view rotation,
    // it is equivalent to rotating the environment cubemap itself.
    radiance_sample_dir = quat_rotate(light_probes.view_rotation, radiance_sample_dir);
    // Cube maps are left-handed so we negate the z coordinate.
    radiance_sample_dir.z = -radiance_sample_dir.z;
    radiances.radiance = textureSampleLevel(
        bindings::specular_environment_map,
        bindings::environment_map_sampler,
        radiance_sample_dir,
        radiance_level).rgb * intensity;

    return radiances;
}

@if(STANDARD_MATERIAL_CLEARCOAT)

/// Adds the environment map light from the clearcoat layer to that of the base
/// layer.
fn environment_map_light_clearcoat(
    out: ptr<function, EnvironmentMapLight>,
    input: ptr<function, LightingInput>,
    clusterable_object_index_ranges: ptr<function, ClusterableObjectIndexRanges>,
    found_diffuse_indirect: bool,
) {
    // Unpack.
    let world_position = (*input).P;
    let clearcoat_NdotV = (*input).layers[LAYER_CLEARCOAT].NdotV;
    let clearcoat_strength = (*input).clearcoat_strength;

    // Calculate the Fresnel term `Fc` for the clearcoat layer.
    // 0.04 is a hardcoded value for F0 from the Filament spec.
    let clearcoat_F0 = vec3<f32>(0.04);
    let Fc = F_Schlick_vec(clearcoat_F0, 1.0, clearcoat_NdotV) * clearcoat_strength;
    let inv_Fc = 1.0 - Fc;

    let clearcoat_radiances = compute_radiances(
        (*input).layers[LAYER_CLEARCOAT],
        clusterable_object_index_ranges,
        world_position,
        found_diffuse_indirect,
    );

    // Composite the clearcoat layer on top of the existing one.
    // These formulas are from Filament:
    // <https://google.github.io/filament/Filament.md.html#lighting/imagebasedlights/clearcoat>
    (*out).diffuse *= inv_Fc;
    (*out).specular = (*out).specular * inv_Fc * inv_Fc + clearcoat_radiances.radiance * Fc;
}

/// Multiscattering approximation: https://www.jcgt.org/published/0008/01/03/paper.pdf
///
/// We initially used this (https://bruop.github.io/ibl) reference with Roughness Dependent
/// Fresnel, but it made fresnel very bright so we reverted to the "typical" fresnel term.
fn compute_multiscatter(
    F0: vec3<f32>,
    F_ab: vec2<f32>,
    Ems: f32,
    specular_occlusion: f32,
) -> MultiscatterResult {
    let FssEss = (F0 * F_ab.x + F_ab.y) * specular_occlusion;
    let Favg = F0 + (1.0 - F0) / 21.0;
    let FmsEms = FssEss * Favg / (1.0 - Ems * Favg) * Ems;
    let Edss = 1.0 - (FssEss + FmsEms);

    return MultiscatterResult(FssEss, FmsEms, Edss);
}

fn environment_map_light(
    input: ptr<function, LightingInput>,
    clusterable_object_index_ranges: ptr<function, ClusterableObjectIndexRanges>,
    found_diffuse_indirect: bool,
) -> EnvironmentMapLight {
    // Unpack.
    let roughness = (*input).layers[LAYER_BASE].roughness;
    let diffuse_color = (*input).diffuse_color;
    let metallic = (*input).metallic;
    let NdotV = (*input).layers[LAYER_BASE].NdotV;
    let F_ab = (*input).F_ab;
    let F0_dielectric = (*input).F0_dielectric;
    let F0_metallic = (*input).F0_metallic;
    let world_position = (*input).P;

    var out: EnvironmentMapLight;

    let radiances = compute_radiances(
        (*input).layers[LAYER_BASE],
        clusterable_object_index_ranges,
        world_position,
        found_diffuse_indirect,
    );

    if (all(radiances.irradiance == vec3(0.0)) && all(radiances.radiance == vec3(0.0))) {
        out.diffuse = vec3(0.0);
        out.specular = vec3(0.0);
        return out;
    }

    // No real world material has specular values under 0.02, so we use this range as a
    // "pre-baked specular occlusion" that extinguishes the fresnel term, for artistic control.
    // See: https://google.github.io/filament/Filament.md.html#specularocclusion
    let F0_surface = mix(F0_dielectric, F0_metallic, metallic);
    let specular_occlusion = saturate(dot(F0_surface, vec3(50.0 * 0.33)));

    // Compute per-material (dielectric and metallic separately) then mix the results.
    // We can't use F0 directly as the multiscattering term is nonlinear.
    let Ems = 1.0 - (F_ab.x + F_ab.y);

    let ms_dielectric = compute_multiscatter(F0_dielectric, F_ab, Ems, specular_occlusion);
    let ms_metallic = compute_multiscatter(F0_metallic, F_ab, Ems, specular_occlusion);

    let FssEss = mix(ms_dielectric.FssEss, ms_metallic.FssEss, metallic);
    let FmsEms = mix(ms_dielectric.FmsEms, ms_metallic.FmsEms, metallic);
    let kD = diffuse_color * ms_dielectric.Edss;

    if (!found_diffuse_indirect) {
        out.diffuse = (FmsEms + kD) * radiances.irradiance;
    } else {
        out.diffuse = vec3(0.0);
    }

    out.specular = FssEss * radiances.radiance;

    @if(STANDARD_MATERIAL_CLEARCOAT)
    environment_map_light_clearcoat(
        &out,
        input,
        clusterable_object_index_ranges,
        found_diffuse_indirect,
    );

    return out;
}

/// "Moving Frostbite to Physically Based Rendering 3.0", listing 22
/// https://seblagarde.wordpress.com/wp-content/uploads/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf#page=70
fn radiance_sample_direction(N: vec3<f32>, R: vec3<f32>, roughness: f32) -> vec3<f32> {
    let smoothness = saturate(1.0 - roughness);
    let lerp_factor = smoothness * (sqrt(smoothness) + roughness);
    return mix(N, R, lerp_factor);
}
