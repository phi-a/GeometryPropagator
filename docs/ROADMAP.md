# Roadmap

## Completed

### 1. CubeSat geometry layer

Body-fixed `CubeSat` geometry builder with:
- rectangular surfaces
- optional patch grids
- one-sided / two-sided surfaces
- hinged deployables (hierarchical hinge realizations)
- `mount(...)` axis-alignment helper
- realized geometry output in body coordinates (`RealizedGeometry`)
- JSON serialization handoff (`to_json` / `from_json`)

### 2. View-factor layer with spacecraft occlusion

Orbit-sweep propagators with local ray tests. Radiator patches can be blocked by:
- deployable solar panels
- neighboring bus surfaces

Products per surface per patch per timestep:
- earth view factor
- albedo view factor
- solar view factor
- solar-panel re-radiation view factor
- other-structure view factor
- space-hemisphere view factor

### 3. Attitude laws and finite-rate slew

- steady-state LVLH-fixed and Sun-pointing laws
- `SlewModeSwitch` — finite-rate SLERP transitions at eclipse entry and exit
- `propagation_grid` refines the orbit sample grid across slew windows

## Current — Phase 3: Thermal layer completion

### 3a. Steady-state thermal balance solver  ← active

`thermal/solver.py`:
- `steady_state_temperature(background, *, alpha_solar, epsilon)` — per-patch temperature from
  incident flux components: `T = (q_absorbed / (ε σ))^0.25`
- `effective_sink_temperature(background)` — effective blackbody environment temperature the
  surface faces from combined Earth IR + panel re-radiation.
  Answers "what temperature sink is the +/-Y radiator looking at?"

`SurfaceThermalProfile` dataclass — temperature `[n_time, ny, nx]`, absorbed flux, eclipse flag.

`SinkTemperatureProfile` dataclass — `T_sink [n_time, ny, nx]` orbit trace.

### 3b. Close the +/-Y analysis loop  ← next

Complete `background.ipynb` to demonstrate end-to-end:
- realized CubeSat geometry → `surface_loading_propagate` on `bus_+Y`, `bus_-Y` →
  `radiative_background` → `steady_state_temperature`
- orbit trace of: radiator temperature, +/-Y bus face temperature,
  effective sink temperature breakdown (Earth / panel / deep space split)

## Mid-Term — Phase 4: Dynamic 3D visualization

### 4a. Per-patch data colormap (static)

`geometry/CubeSat/plots.py`:
- add `plot_patch_data(realized, surface_name, data_2d, *, title, cmap)` —
  colormapped static view of a single surface's patch data (flux or temperature)

### 4b. Full spacecraft 3D renderer

`geometry/CubeSat/viz3d.py` — Plotly-based interactive 3D renderer:
- **Solar panel leaves** — mesh grid colored by absorbed solar flux or temperature
- **`bus_+Y` / `bus_-Y` faces** — mesh grid colored by incident flux or temperature
- **Other bus faces** — white, partially translucent
- Body-axis frame arrows (velocity, orbit-normal, zenith)
- Sun direction arrow, nadir arrow
- Color scale and legend per surface group

Rationale for Plotly: renders interactively in notebooks, clean colormap on mesh patches,
no extra install burden, orbit time as a slider frame over the precomputed profile arrays.

### 4c. Orbit animation

Extend `viz3d.py` with:
- orbit-time slider or `FuncAnimation` — spacecraft body rotates in a fixed-sun frame,
  surface patches re-color at each timestep
- Shows attitude slew visually when `SlewModeSwitch` is in use

## Longer-Term — Phase 5 and beyond

### 5. Surface-to-surface geometric exchange

Patch-to-patch view products for:
- radiator to solar panel
- radiator to bus wall
- panel to panel

Remains geometric only — no temperatures in the view-factor layer.

### 6. Deep-space visibility products

Compute:
- visible space fraction
- blocked space fraction
- view to specific surface groups

### 7. Richer local geometry

Add:
- explicit baffles
- louver-like directional acceptance
- arbitrary rectangular occluder groups

### 8. Harmonize with planner-style interfaces

Define stable input/output objects that mirror the planner style:
- geometry input state
- view-factor result
- thermal load result

## Non-Goals for Now

- full CAD ingestion
- mesh engines
- hemicube rendering
- high-order thermal FEM
- lumped-node transient thermal solver (steady-state per orbit point is sufficient for now)

Those can come later if needed. The immediate goal is a minimal, robust, explainable geometry
and thermal-environment kernel with clear 3D feedback.
