# GeometryPropagator

GeometryPropagator is a spacecraft-facing geometry and view-factor engine for orbit-driven thermal analysis.

The current focus is geometric truth:
- orbit geometry in ECI/LVLH
- attitude laws and finite-rate slew transitions
- Earth-disk integration instead of scalar `cos(alpha)` approximations
- panel-resolved radiator loading with local masking and recessed geometry

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
- Earth-disk quadrature
- face-level directional masking
- patch-resolved rectangular radiator panels
- simple local recessed-wall geometry
- legacy scalar flat-plate models retained under `geometry.legacy`

## Recent Update

The latest repo update did three structural things:
- moved the old scalar flat-plate model into `geometry.legacy`
- split finite-rate slew logic into `geometry/transitions.py`
- added a new `geometry/CubeSat/` layer with a default 6U double-deployable example

The default CubeSat example is:
- a 6U bus
- six body faces
- two double-leaf deployable solar-panel wings
- hinged along the top-edge 3U rail
- realized in the body frame from a small set of deployment angles

Planned next:
- local occlusion against deployable solar panels and bus surfaces
- spacecraft local occlusion in the view-factor layer
- surface-to-surface geometric exchange products
- cleaner handoff into a separate thermal layer

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

orbit = Orbit.from_epoch(
    a=6771e3,
    i=math.radians(51.6),
    omega=math.radians(30.0),
    epoch=datetime(2025, 6, 21, 12, 0, 0),
)

law = TargetTracking(math.radians(266.4168), math.radians(-29.0078))
```

Right now, the `CubeSat` layer gives the view-factor engine a clean geometry object to consume next. The current Earth-disk and panel propagators already exist; the next step is to let them ray-test against the realized CubeSat surfaces for local occlusion.

## Quick Start

Run the smoke tests:

```bash
python -m unittest tests.test_geometry_package
```

Use the notebook:

```bash
jupyter notebook run_geometry.ipynb
```

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
