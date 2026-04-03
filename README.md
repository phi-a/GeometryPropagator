# GeometryPropagator

GeometryPropagator is a spacecraft-facing geometry and view-factor engine for orbit-driven thermal analysis.

The current focus is geometric truth:
- orbit geometry in ECI/LVLH
- attitude laws and finite-rate slew transitions
- Earth-disk integration instead of scalar `cos(alpha)` approximations
- panel-resolved radiator loading with local masking and recessed geometry
- body-fixed spacecraft geometry with deployable mechanisms
- spacecraft self-occlusion as a geometric visibility product

The package is intentionally being organized so that:
- **geometry** defines spacecraft shape and kinematics
- **view factor** computes who can see whom, and by how much
- **thermal** consumes those geometric products later

The thermal solver is not the center of this repository. This repository should become the geometry and visibility kernel that a thermal planner can call.

## Current Status

Implemented now:
- orbit geometry and Sun/eclipse geometry
- attitude laws
- finite-rate mode-switch slews
- body-fixed `CubeSat` geometry layer
- hierarchical deployable-panel realization from hinge states
- optional rigid geometry-frame -> body-frame mount transform
- nearest-hit ray queries against realized spacecraft surfaces
- patch-by-ray spacecraft self-occlusion masks
- patch-resolved directional integration with spacecraft blockage applied
- Earth-disk quadrature
- face-level directional masking
- patch-resolved rectangular radiator panels
- simple local recessed-wall geometry
- notebook demos for radiator geometry and spacecraft geometry
- legacy scalar flat-plate models retained under `geometry.legacy`

## Recent Progress

The current repo state is beyond the first `CubeSat` builder milestone. It now has:
- the old scalar flat-plate model moved into `geometry.legacy`
- finite-rate slew logic split into `geometry/transitions.py`
- a dedicated `geometry/CubeSat/` layer with a default 6U double-deployable example
- a reusable `geometry/occlusion.py` bridge for spacecraft self-obstruction
- `run_spacecraft_geometry.ipynb` as the active spacecraft-geometry demo notebook

The default CubeSat example is:
- a 6U bus
- six body faces
- two double-leaf deployable solar-panel wings
- hinged along the top-edge 3U rail
- realized from a small set of deployment angles
- optionally mounted into the body frame with one rigid transform
- each panel leaf defaults to the full 6U side-panel `y x z` dimensions

What the repo can do today:
- build and realize spacecraft geometry for default or custom deployment states
- keep geometry, mount, and attitude as separate concepts
- query first-hit intersections on realized surfaces
- compute self-occlusion maps for a chosen face or panel patch grid
- visualize direction-space blockage and face-space blockage in the notebook

What is still remaining:
- feed spacecraft self-occlusion directly into the active Earth-disk loading path
- combine local panel recess masking with global spacecraft self-obstruction in one propagator
- add geometric exchange products beyond Earth view, including patch-to-patch visibility
- keep the thermal layer downstream instead of letting it drive the geometry design

## Repo Layout

Current repository layout:

```text
GeometryPropagator/
- README.md
- pyproject.toml
- .gitignore
- geometry/
  - __init__.py
  - constants.py
  - sampling.py
  - orbit.py
  - so3.py
  - laws.py
  - transitions.py
  - CubeSat/
  - earthdisk.py
  - panel.py
  - propagator.py
  - legacy/
- tests/
- docs/
- run_geometry.ipynb
- run_spacecraft_geometry.ipynb
```

## Layer Boundaries

### 1. Geometry Layer
Owns only spacecraft shape and mechanism state.

Examples:
- body-fixed surfaces
- patch definitions
- deployables
- local occluder geometry
- default 6U double-deployable CubeSat examples

### 2. View-Factor Layer
Owns only geometric visibility and exchange.

Examples:
- Earth view factor
- deep-space visibility
- local masking/occlusion
- patch-to-patch geometric exchange factors

Outputs from this layer should be geometric products, not thermal loads.

### 3. Thermal Layer
Consumes view factors and radiance/source models.

Examples:
- Earth IR loading
- albedo loading
- solar loading
- reradiation / node balance / temperature solve

This separation is deliberate. It keeps the geometry engine reusable and keeps the thermal layer from taking over the repo structure.

## How It Works

Briefly, the current flow is:

1. Build body-fixed geometry.
2. Realize that geometry for a mechanism state.
3. Propagate orbit and attitude.
4. Evaluate geometric visibility / Earth loading.
5. Hand those geometric products to a separate thermal layer later.

Minimal example:

```python
import math
from datetime import datetime

from geometry import (
    Orbit,
    SO3,
    TargetTracking,
    build_6u_double_deployable,
)

cubesat = build_6u_double_deployable()
realized = cubesat.realize({
    'wing_port_inner_angle': math.pi / 2,
    'wing_port_outer_angle': math.pi,
    'wing_starboard_inner_angle': -math.pi / 2,
    'wing_starboard_outer_angle': -math.pi,
})

realized_mounted = cubesat.realize(
    mount_rotation=SO3.Rz(math.radians(90.0)),
    mount_offset=[0.0, 0.1, 0.0],
)

orbit = Orbit.from_epoch(
    a=6771e3,
    i=math.radians(51.6),
    omega=math.radians(30.0),
    epoch=datetime(2025, 6, 21, 12, 0, 0),
)

law = TargetTracking(math.radians(266.4168), math.radians(-29.0078))
```

To change the deployable panel size, pass explicit leaf dimensions:

```python
cubesat = build_6u_double_deployable(
    leaf_y=0.2263,
    leaf_z=0.3405,
)
```

Right now, the `CubeSat` layer gives the view-factor engine a clean geometry object to consume next. The current Earth-disk and panel propagators already exist; the next step is to let them ray-test against the realized CubeSat surfaces for local occlusion.

The important frame split is:
- geometry state: surface layout and hinge angles
- mount state: how the geometry is attached to the body axes
- attitude state: how the body frame is oriented in LVLH or ECI

## Quick Start

Run the smoke tests:

```bash
python -m unittest tests.test_geometry_package
```

Use the notebook:

```bash
jupyter notebook run_geometry.ipynb
```

Use the spacecraft geometry / local-occlusion demo notebook:

```bash
jupyter notebook run_spacecraft_geometry.ipynb
```

The current notebook flow is:
- view the default deployed geometry
- view one custom deployment state for intuition
- run ray and self-occlusion diagnostics on the default deployment
- inspect both direction-space and face-space blockage plots

## Legacy Models

The original scalar flat-plate approximations are still available:

```python
from geometry.legacy import earth_vf, propagate, thermal_propagate
```

They are retained for comparison and backward compatibility, but they are not the active modeling path.

## Architecture Docs

See:
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/ROADMAP.md`](docs/ROADMAP.md)
