# MPS patches on Bevy

This fork carries the changes Motion Pipes Studio (MPS, https://github.com/megalike/MotionPipesStudio) needs on top
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
