# Architecture

## Purpose

GeometryPropagator is the geometry, visibility, and thermal-environment engine for spacecraft thermal analysis.

Its job is to answer:
- where is the spacecraft?
- how is it oriented?
- what realized body-frame surfaces exist?
- what does each surface or patch see?
- what is blocked by local spacecraft geometry?
- what is the steady-state temperature of each patch?
- what is the transient two-sided temperature of each solar-panel wing?
- what effective blackbody environment is each radiator facing?

It should not become a full thermal network solver.

## Design Rule

Four layers must remain distinct.

```text
Geometry -> View-factor -> Thermal -> (external solver)
```

The thermal layer in this repo ends at equilibrium temperature per patch, effective sink temperature, and 2-D analysis plots. A downstream lumped-node or FEM solver can consume those products.

## 1. Geometry Layer

This layer owns spacecraft shape, mechanism state, orbit geometry, and attitude laws.

Inputs:
- body-fixed surface definitions
- patch grids
- deployable states
- hinge angles / mechanism states
- optional geometry-frame -> body-frame mount choice

Outputs:
- realized surface poses in body coordinates
- patch centers and normals
- occluding surfaces / ray-test targets
- orbit and attitude state used downstream

The default 6U builder now realizes each deployable wing as:
- one front cell-side surface tagged `solar_panel`
- one co-planar back surface tagged `solar_panel_back`

Both carry the shared `solar_array` tag so the reradiation bucket can group the full wing set.

This layer does not know about:
- Earth IR constants
- albedo coefficients
- emissivity
- absorptivity
- temperatures

### Geometry artifact boundary

The active geometry -> view-factor boundary is `RealizedGeometry`.

That object is intentionally flat:
- one entry per realized rectangular surface
- body-frame `center`, `normal`, and `u_axis`
- face dimensions and patch metadata
- tags and stable surface names
- provenance metadata such as mechanism state and mount transform

It can be serialized directly with:
- `RealizedGeometry.to_json(path)`
- `RealizedGeometry.from_json(path)`

This is the intended handoff:

```text
run_cubesat_geometry.ipynb
  -> build body-fixed geometry
  -> realize(mechanism_state, mount_rotation)
  -> RealizedGeometry.to_json("outputs/spacecraft.json")

background.ipynb
  -> RealizedGeometry.from_json("outputs/spacecraft.json")
  -> surface_loading_propagate(...) on panel fronts and backs
  -> radiative_background(...)
  -> transient_temperature(...) for each wing
  -> radiative_background(...) for radiator faces using the shared solar-array temperature trace
  -> steady_state_temperature(...)
  -> plot_temperature_trace(...)
  -> plot_temperature_heatmap(...)
```

Once realized, the builder hierarchy and frame-convention bookkeeping are gone. Downstream consumers should use the realized surfaces only.

## 2. View-Factor Layer

This layer owns geometric visibility only.

Inputs:
- realized spacecraft geometry
- orbit state
- attitude state
- source geometry such as Earth disk or Sun direction

Outputs:
- patch-to-Earth geometric factors
- patch-to-space visibility
- patch-to-surface geometric factors
- masked visibility fractions
- occlusion maps

Current grouped warm-structure product:
- `solar_panel_view` is computed from the shared `solar_array` tag

Because the front and back wing faces are exactly co-planar in v1, this changes the grouping semantics even though many geometric baselines remain numerically unchanged under the current nearest-hit ray model.

This layer should output geometry products only.

It should not output:
- temperatures
- thermal node states

It may output source-separated geometric products that a thermal layer later scales into watts.

## 3. Thermal Layer

This layer consumes geometry products and converts them to heat inputs, equilibrium temperatures, and 2-D analysis plots.

Inputs:
- geometric visibility products (`SurfaceBackgroundProfile`)
- source radiance models already folded into the background conversion
- material properties: `alpha_solar` and `epsilon`

Outputs (current):
- incident radiative background per patch per timestep (`SurfaceBackgroundProfile`)
- absorbed heat inputs
- single-sided steady-state equilibrium temperature per patch per timestep (`SurfaceThermalProfile`)
- two-sided steady-state solar-panel temperature (`SurfaceThermalProfile`)
- transient two-sided solar-panel temperature over a converged orbit (`SurfaceThermalProfile`)
- effective IR sink temperature the surface faces (`SinkTemperatureProfile`)
- 2-D temperature envelope plots
- 2-D patch temperature heatmaps

### Thermal balance equation

```text
q_absorbed = alpha_solar * (q_solar + q_albedo) + epsilon * (q_earth_ir + q_panel_ir)
T_ss = (q_absorbed / (epsilon * sigma)) ^ 0.25
```

### Two-sided solar-panel balance

For the deployable wings, the cell side and painted back side are solved as one thermally coupled panel with a shared temperature field:

```text
q_abs,front + q_abs,back = (epsilon_front + epsilon_back) * sigma * T^4
```

The transient panel solver integrates:

```text
C_areal * dT/dt = q_abs,total(t) - (epsilon_front + epsilon_back) * sigma * T^4
```

Current notebook default:
- panel-to-panel reradiation is omitted inside the panel transient solve itself
- bus radiators then consume one area-weighted mean solar-array temperature trace

### Effective sink temperature

The composite IR environment the surface faces, expressed as a single equivalent blackbody temperature:

```text
T_sink = ((q_earth_ir + q_panel_ir) / sigma) ^ 0.25
```

This answers "what temperature blackbody environment is the +/-Y radiator looking at?" Earth IR, solar-panel re-radiation, and implicitly deep space are folded in. It is purely a property of the environment and does not depend on surface material.

## 4. Visualization Layer (planned - 3-D only)

This layer provides 3-D spacecraft renders with data-driven surface coloring.

It consumes realized geometry and precomputed profile arrays. It does not feed back into any upstream layer.

The 2-D thermal trace and heatmap plots are already part of the thermal layer. This planned layer is only for 3-D rendered visualization.

Planned outputs:
- static per-patch colormap on realized spacecraft faces
- interactive 3-D scene with colored mesh patches
- orbit-time animation with a slider or `FuncAnimation`

Surface rendering rules:
- solar panel leaves -> mesh grid colored by solar flux absorbed or temperature
- `bus_+Y` / `bus_-Y` faces -> mesh grid colored by incident background or temperature
- other bus faces -> white, partially translucent

## Current Package Map

### Geometry

- `geometry/orbit.py`
  - orbit geometry
  - Sun direction
  - eclipse geometry

- `geometry/so3.py`
  - rotations
  - alignment utilities
  - quaternion conversion
  - SLERP support

- `geometry/laws.py`
  - steady-state attitude laws

- `geometry/transitions.py`
  - finite-rate attitude wrappers
  - `SlewModeSwitch` for eclipse-boundary SLERP transitions

- `geometry/CubeSat/`
  - body-fixed CubeSat surfaces
  - hierarchical hinge realizations
  - default 6U double-deployable builder
  - front/back solar-panel surface pairing via `flip_surface(...)`
  - realized-geometry inspection helpers
  - `mount(...)` axis-alignment helper

### View factor

- `viewfactor/earthdisk.py`
  - Earth-disk quadrature
  - face-coordinate transforms
  - directional masks

- `viewfactor/panel.py`
  - patch-resolved rectangular radiator geometry
  - simple recessed local geometry

- `viewfactor/occlusion.py`
  - spacecraft self-occlusion ray tests
  - group-view integration

- `viewfactor/propagator.py`
  - orbit-sweep propagators
  - `surface_loading_propagate(realized, surface_name, orbit, law, ...)`
  - `solar_panel_view` grouped from the shared `solar_array` tag

- `viewfactor/plots.py`
  - occlusion and patch-map plotting helpers

### Thermal

- `thermal/background.py`
  - `radiative_background(profile, ...)`
  - `SurfaceBackgroundProfile`

- `thermal/solver.py`
  - `steady_state_temperature(background, *, alpha_solar, epsilon)`
  - `steady_state_temperature_two_sided(bg_front, bg_back, ...)`
  - `transient_temperature(bg_front, bg_back, ...)`
  - `effective_sink_temperature(background)`
  - `SurfaceThermalProfile`
  - `SinkTemperatureProfile`

- `thermal/plots.py`
  - `plot_temperature_trace(ax, profiles, ...)`
  - `plot_temperature_heatmap(ax, profile, ...)`
  - standalone Matplotlib helpers on thermal profile dataclasses

### Shared support

- `geometry/constants.py`
- `geometry/sampling.py`
- `viewfactor/sampling.py`
- `thermal/constants.py`

### Legacy

- `geometry/legacy/scalar.py`
  - old infinitesimal flat-plate scalar model

## CubeSat Geometry Notes

The dedicated geometry-builder package is `geometry/CubeSat/`.

Core concepts:
- `RectSurface`
  - stable geometry identity
  - body-frame position and orientation after realization
  - width / height
  - one-sided / two-sided behavior
  - optional patch grid metadata

- `SurfaceNode`
  - hierarchical builder node
  - hinge origin
  - hinge axis
  - deployment angle
  - optional parent surface

- `CubeSatGeometry`
  - collection of body-fixed builder nodes
  - realizes to a flat `RealizedGeometry`

- `RealizedGeometry`
  - flat body-frame surface list
  - JSON-serializable handoff object
  - mountable copy via `mounted(...)`

The default example is a 6U bus with two double-leaf deployable solar-panel wings mounted along the top-edge 3U rail.

## Interface Direction

The interface between layers should look like:

```text
builder  = geometry_layer.build(...)
realized = builder.realize(mechanism_state, mount_rotation)
realized.to_json(...)

loaded   = geometry_layer.RealizedGeometry.from_json(...)
vf_prof  = viewfactor_layer.surface_loading_propagate(loaded, surface_name, orbit, law)
bg_prof  = thermal_layer.radiative_background(vf_prof, ...)
th_prof  = thermal_layer.steady_state_temperature(bg_prof, alpha_solar=..., epsilon=...)
sink     = thermal_layer.effective_sink_temperature(bg_prof)
ax       = thermal_layer.plot_temperature_trace(...)
```

That compartment boundary must be preserved.

## Mounted-Surface Interpretation Rule

Two rules matter for downstream analysis:
- names such as `bus_+X` are stable builder identities and are not rewritten after a mount transform
- analysis roles such as `body +Y bus face` must be selected from the realized geometry by the current mounted normal

That rule keeps the handoff unambiguous even when the same physical surface is mounted into a different body-axis role.
