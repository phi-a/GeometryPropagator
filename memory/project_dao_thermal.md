---
name: DAO thermal simulation goals
description: Three concrete thermal analysis goals driving the DAO repo roadmap
type: project
---

Simulating a 6U CubeSat with deployable solar arrays to answer three questions:
1. How hot does the +/-Y radiator panel get (equilibrium temperature vs orbit position)?
2. How hot do the +/-Y bus surface panels get from the environment?
3. What effective background temperature do the +/-Y radiator panels see (Earth + deep space + solar panel shadowing)?

**Why:** Thermal design and radiator sizing for a real spacecraft mission. The +/-Y surfaces are the primary radiators in body-coordinate framing.

**How to apply:** When adding features, keep the three goals in mind. Temperature outputs (K) are the end product, not just W/m² flux.

## Current pipeline state (2026-04-03)

- Geometry: solid — `build_6u_double_deployable`, `RealizedGeometry`, JSON handoff
- View-factor: solid — `surface_loading_propagate` does orbit sweep with occlusion by solar panels
- Thermal background: solid — `radiative_background` gives W/m² per patch
- **Thermal solver: just added** — `thermal/solver.py` with `steady_state_temperature` and `effective_sink_temperature`
- 3D visualization: static only — Phase 4 (Plotly interactive + orbit animation) is planned next
- Slew: `SlewModeSwitch` exists, works via `law` argument to propagators

## Next steps

Phase 3b: Close +/-Y analysis loop in `background.ipynb` — demonstrate full end-to-end temperature orbit trace.
Phase 4: Dynamic 3D visualization in `geometry/CubeSat/viz3d.py` using Plotly — colored patch meshes on solar panels and +/-Y faces, orbit time as slider.
