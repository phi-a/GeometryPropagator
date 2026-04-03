# Architecture

## Purpose

GeometryPropagator is the geometry, visibility, and thermal-environment engine for spacecraft
thermal analysis.

Its job is to answer:
- where is the spacecraft?
- how is it oriented?
- what realized body-frame surfaces exist?
- what does each surface or patch see?
- what is blocked by local spacecraft geometry?
- what is the steady-state temperature of each patch?
- what effective blackbody environment is each radiator facing?

It should not become a full thermal network solver.

## Design Rule

Four layers must remain distinct.

```
Geometry  →  View-factor  →  Thermal  →  (external solver)
```

The thermal layer in this repo ends at equilibrium temperature per patch and effective
sink temperature.  A downstream lumped-node or FEM solver can consume those products.

---

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

This layer does not know about:
- Earth IR constants
- albedo coefficients
- emissivity
- absorptivity
- temperatures

### Geometry artifact boundary

The active geometry → view-factor boundary is `RealizedGeometry`.

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
  -> surface_loading_propagate(...)
  -> radiative_background(...)
  -> steady_state_temperature(...)
```

Once realized, the builder hierarchy and frame-convention bookkeeping are gone. Downstream
consumers should use the realized surfaces only.

---

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

This layer should output geometry products only.

It should not output:
- temperatures
- thermal node states

It may output source-separated geometric products that a thermal layer later scales into watts.

---

## 3. Thermal Layer

This layer consumes geometry products and converts them to heat inputs and equilibrium
temperatures.

Inputs:
- geometric visibility products (`SurfaceBackgroundProfile`)
- source radiance models (J_IR, S0, albedo constant — already embedded in background profile)
- material properties: `alpha_solar` (solar absorptivity), `epsilon` (IR emissivity)

Outputs (current):
- incident radiative background per patch per timestep (`SurfaceBackgroundProfile`)
- absorbed heat inputs
- **steady-state equilibrium temperature** per patch per timestep (`SurfaceThermalProfile`)
- **effective IR sink temperature** the surface faces (`SinkTemperatureProfile`)

### Thermal balance equation

```
q_absorbed = α_solar * (q_solar + q_albedo) + ε * (q_earth_ir + q_panel_ir)
T_ss = (q_absorbed / (ε σ)) ^ 0.25
```

### Effective sink temperature

The composite IR environment the surface faces, expressed as a single equivalent
blackbody temperature:

```
T_sink = ((q_earth_ir + q_panel_ir) / σ) ^ 0.25
```

This answers "what temperature blackbody environment is the +/-Y radiator looking at?"
Earth IR, solar-panel re-radiation, and implicitly deep space (which contributes 0 W/m²)
are all folded in.  It is purely a property of the environment — independent of surface
material.

---

## 4. Visualization Layer (planned — Phase 4)

This layer provides 3D spacecraft renders with data-driven surface coloring.

It consumes realized geometry and precomputed profile arrays.  It does not feed back
into any upstream layer.

Planned outputs:
- static per-patch colormap (`geometry/CubeSat/plots.py` extension)
- interactive 3D Plotly scene with colored mesh patches (`geometry/CubeSat/viz3d.py`)
- orbit-time animation (slider or `FuncAnimation`)

Surface rendering rules:
- solar panel leaves → mesh grid colored by solar flux absorbed or temperature
- `bus_+Y` / `bus_-Y` faces → mesh grid colored by incident background or temperature
- other bus faces → white, partially translucent

---

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
  - `SlewModeSwitch` — eclipse-boundary SLERP transitions

- `geometry/CubeSat/`
  - body-fixed CubeSat surfaces
  - hierarchical hinge realizations
  - default 6U double-deployable builder
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

### Thermal

- `thermal/background.py`
  - `radiative_background(profile, ...)` — incident flux components per patch
  - `SurfaceBackgroundProfile` — per-patch W/m² orbit trace

- `thermal/solver.py`  ← Phase 3a
  - `steady_state_temperature(background, *, alpha_solar, epsilon)`
  - `effective_sink_temperature(background)`
  - `SurfaceThermalProfile` — temperature + absorbed flux orbit trace
  - `SinkTemperatureProfile` — effective IR sink temperature orbit trace

### Shared support

- `geometry/constants.py`
- `geometry/sampling.py`
- `viewfactor/sampling.py`
- `thermal/constants.py`

### Legacy

- `geometry/legacy/scalar.py`
  - old infinitesimal flat-plate scalar model

---

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

The default example is a 6U bus with two double-leaf deployable solar-panel wings mounted
along the top-edge 3U rail.

---

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
```

That compartment boundary must be preserved.

---

## Mounted-Surface Interpretation Rule

Two rules matter for downstream analysis:
- names such as `bus_+X` are stable builder identities and are not rewritten after a mount transform
- analysis roles such as `body +Y bus face` must be selected from the realized geometry by the
  current mounted normal

That rule keeps the handoff unambiguous even when the same physical surface is mounted into a
different body-axis role.
