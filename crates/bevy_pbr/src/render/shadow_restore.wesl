// Restores texels of a cached spot light's static shadow layer into its shadow
// map: each instance is a rectangle of texels, given in normalized device
// coordinates (min x, min y, max x, max y), and each fragment writes the
// static layer's depth at its texel, exactly (no filtering), over whatever the
// shadow map held (depth compare `Always`).

@group(0) @binding(0)
var static_layer: texture_depth_2d;

// A triangle strip over the rectangle's four corners.
@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) rect: vec4<f32>,
) -> @builtin(position) vec4<f32> {
    let x = select(rect.x, rect.z, (vertex_index & 1u) != 0u);
    let y = select(rect.y, rect.w, (vertex_index & 2u) != 0u);
    return vec4<f32>(x, y, 0.0, 1.0);
}

struct FragmentOutput {
    @builtin(frag_depth) frag_depth: f32,
}

@fragment
fn fragment(@builtin(position) position: vec4<f32>) -> FragmentOutput {
    var out: FragmentOutput;
    out.frag_depth = textureLoad(static_layer, vec2<i32>(position.xy), 0);
    return out;
}
