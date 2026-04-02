# Architecture

## Purpose

GeometryPropagator is the geometry and visibility engine for spacecraft thermal analysis.

Its job is to answer:
- where is the spacecraft?
- how is it oriented?
- what does each surface or patch see?
- what is blocked by local geometry?

It should not become the thermal solver itself.

## Design Rule

Three layers must remain distinct:

## 1. Geometry Layer

This layer owns only spacecraft shape and kinematics.

Inputs:
- body-fixed surface definitions
- patch grids
- deployable states
- hinge angles / mechanism states

Outputs:
- realized surface poses in body coordinates
- patch centers and normals
- occluding surfaces / ray-test targets

This layer does not know about:
- Earth IR
- albedo coefficients
- emissivity
- absorptivity
- temperatures

## 2. View-Factor Layer

This layer owns only geometric visibility.

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
- watts
- temperatures
- thermal node states

## 3. Thermal Layer

This layer consumes geometry products and converts them to heat loads or temperatures.

Inputs:
- view-factor outputs
- source radiance models
- material properties
- thermal node definitions
- temperatures, when doing reradiation or coupled exchange

Outputs:
- incident heat flux
- absorbed heat
- reradiated heat
- node temperatures

## Current Package Map

### Active modules

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

- `geometry/earthdisk.py`
  - Earth-disk quadrature
  - face-coordinate transforms
  - directional masks

- `geometry/panel.py`
  - patch-resolved rectangular radiator panels
  - simple recessed local geometry

- `geometry/propagator.py`
  - active disk-integrated and panel-resolved view-factor sweeps

### Shared support

- `geometry/constants.py`
- `geometry/sampling.py`

### Legacy

- `geometry/legacy/scalar.py`
  - old infinitesimal flat-plate scalar model

## Next Geometry Layer

The next structural addition should be a minimal spacecraft geometry builder.

Recommended package direction:

```text
geometry/
  spacecraft/
    __init__.py
    surfaces.py
    assemblies.py
    builder.py
    realize.py
    rays.py
```

Minimal concepts:

- `Surface`
  - name
  - parent frame
  - center
  - normal
  - tangent axes
  - width / height
  - one-sided / two-sided
  - patch grid metadata

- `HingedSurface` or `HingedAssembly`
  - hinge origin
  - hinge axis
  - deployment angle
  - child surfaces

- `SpacecraftGeometry`
  - collection of surfaces and assemblies
  - realized body-frame geometry at a given mechanism state

## Interface Direction

The interface between geometry and view-factor should look like:

```text
realized_geometry = geometry_layer.realize(mechanism_state)
vf_result = viewfactor_layer.evaluate(realized_geometry, orbit_state, attitude_state)
thermal_result = thermal_layer.evaluate(vf_result, source_models, materials)
```

That is the compartment boundary to preserve.
