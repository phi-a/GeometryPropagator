# Architecture

## Purpose

GeometryPropagator is the geometry and visibility engine for spacecraft thermal analysis.

Its job is to answer:
- where is the spacecraft?
- how is it oriented?
- what realized body-frame surfaces exist?
- what does each surface or patch see?
- what is blocked by local spacecraft geometry?

It should not become the thermal solver itself.

## Design Rule

Three layers must remain distinct.

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
  -> surface_loading_propagate(...)
  -> radiative_background(...)
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

This layer should output geometry products only.

It should not output:
- temperatures
- thermal node states

It may output source-separated geometric products that a thermal layer later scales into watts.

## 3. Thermal Layer

This layer consumes geometry products and converts them to heat inputs.

Inputs:
- geometric visibility products
- source radiance models
- material properties
- selected thermal assumptions

Outputs:
- incident radiative background
- absorbed heat inputs
- thermal-prep products for a later temperature solver

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
  - current example: `SlewModeSwitch`

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
  - thermal consumers built on top of geometric products
  - `radiative_background(...)`

### Shared support

- `geometry/constants.py`
- `geometry/sampling.py`
- `viewfactor/sampling.py`

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
builder = geometry_layer.build(...)
realized = builder.realize(mechanism_state, mount_rotation)
realized.to_json(...)

loaded = geometry_layer.RealizedGeometry.from_json(...)
vf_result = viewfactor_layer.evaluate(loaded, surface_name, orbit_state, attitude_state)
thermal_result = thermal_layer.evaluate(vf_result, source_models)
```

That is the compartment boundary to preserve.

## Mounted-Surface Interpretation Rule

Two rules matter for downstream analysis:
- names such as `bus_+X` are stable builder identities and are not rewritten after a mount transform
- analysis roles such as `body +Y bus face` must be selected from the realized geometry by the current mounted normal

That rule keeps the handoff unambiguous even when the same physical surface is mounted into a different body-axis role.
