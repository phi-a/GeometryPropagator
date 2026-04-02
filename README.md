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
- Earth-disk quadrature
- face-level directional masking
- patch-resolved rectangular radiator panels
- simple local recessed-wall geometry
- legacy scalar flat-plate models retained under `geometry.legacy`

Planned next:
- a body-fixed spacecraft geometry builder/layer
- deployable solar-panel geometry
- spacecraft local occlusion in the view-factor layer
- surface-to-surface geometric exchange products
- cleaner handoff into a separate thermal layer

## Repo Layout

Current repository layout for the new repo should be:

```text
GeometryPropagator/
├── README.md
├── pyproject.toml
├── .gitignore
├── geometry/
│   ├── __init__.py
│   ├── constants.py
│   ├── sampling.py
│   ├── orbit.py
│   ├── so3.py
│   ├── laws.py
│   ├── transitions.py
│   ├── earthdisk.py
│   ├── panel.py
│   ├── propagator.py
│   └── legacy/
│       ├── __init__.py
│       └── scalar.py
├── tests/
│   ├── __init__.py
│   └── test_geometry_package.py
├── docs/
│   ├── ARCHITECTURE.md
│   └── ROADMAP.md
└── run_geometry.ipynb
```

## Layer Boundaries

### 1. Geometry Layer
Owns only spacecraft shape and mechanism state.

Examples:
- body-fixed surfaces
- patch definitions
- deployables
- local occluder geometry

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
