# MPS patches on Bevy

This fork carries the changes Motion Pipes Studio (MPS, <https://github.com/megalike/MotionPipesStudio>) needs on top
of a Bevy release. Everything else is Bevy as released. MPS's rules for this fork: `knowledge/BEVY-FORK.md` in MPS.

## Conventions

- One branch per Bevy release: `mps/0.20` starts at the tag `v0.20.0`. A Bevy update starts `mps/0.21` at `v0.21.0`
  and carries over every patch below that the new release does not already contain (merged on Bevy's `main` is not
  enough).
- One change per pull request into `mps/<version>`, from a topic branch (`fix/…`, `feat/…`, `improve/…`, `perf/…`),
  squash-merged to one commit whose subject says what it is: `fix(<crate>):` a Bevy bug, `feat(<crate>):` something
  Bevy lacks, `improve(<crate>):` better quality or API, `perf(<crate>):` speed; `docs(mps):` for this file. Labels on
  the pull request say the same (`bug`, `feature`, `improvement`, `performance`) and where it may go
  (`upstream-candidate`, `mps-only`). A patch adds its row below in the same commit.
- Commit trailers, written into the squash message: `MPS-Kind` (fix / feature / improve / perf / mps-only),
  `MPS-Decision` (the MPS decision that asked for it), `MPS-Evidence` (the MPS report with the measurements). The
  upstream status (not filed / an issue or PR link / contained in a release) is kept in the table below.
- Tags mark the revisions MPS uses: `mps-<bevy version>.<n>`, for example `mps-0.20.0.1`, never moved once pushed,
  each with a GitHub release listing its patches. MPS pins exactly one tag.
- Upstream contributions are separate branches from Bevy's current `main` and follow Bevy's contribution guide and AI
  policy; the owner writes and submits them himself.

## Patches on `mps/0.20`

| Patch | Kind | Crate | What | Upstream |
| --- | --- | --- | --- | --- |
| Receiver-plane depth for spot-light PCSS | fix | bevy_pbr | Every blocker-search and filter tap of spot-light PCSS compares with the receiver plane's depth at the uv it reads, not one depth for all taps: grazing-lit walls no longer show self-shadowing combs. Forward rendering only (Bevy's deferred lighting pass has the normal-mapped normal and keeps one depth); the plane comes from the interpolated vertex normal | not filed (candidate after a minimal repro and deferred and smooth-mesh coverage) |
| Contact shadows fade with hit distance | improve | bevy_pbr | Screen-space contact shadows fade by the blocker's world distance along the ray, so long hard lines that a soft light would not draw fade out while tight contacts stay dark; applies to spot, point and directional lights and also weakens true hard-light contacts half-way along the ray | not filed (to discuss upstream as an option, not a new default) |
| Soft contact shadows from the light's size | feature | bevy_pbr | For a point or spot light with soft shadows under temporal shadow filtering, each pixel and frame marches the contact-shadow ray towards one point of the light sphere's apparent disc (uniform in solid angle, a PCG stream per pixel, frame and light) and TAA averages the penumbra; lights without a size, the directional light and other filter methods keep one hard ray | not filed (candidate; needs a motion and TAA-history check) |
| Spot-light PCSS minimum filter | improve | bevy_pbr | The receiver-plane spot PCSS never filters narrower than `SPOT_PCSS_MIN_BLUR` (2.0 `blur_size` units, an outer tap radius of about 3.7 shadow-map texels, against 0.5) like EEVEE's per-light filter radius: casters a texel or two wide are soft bands instead of aliased streaks, real contacts soften by that footprint; Bevy's deferred lighting pass keeps 0.5 | not filed (depends on the receiver-plane patch; an option, not a default) |
| Lint fixes for the fork's CI | mps-only | bevy_dev_tools, bevy_solari, bevy_pbr | Two dependency lines in the release manifests take the one-line form `taplo` asks for (only formatting, not in `main`, which still has `-dev` versions); American spelling in the patches' comments; no shader or dependency change | not applicable |
