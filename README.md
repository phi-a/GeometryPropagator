# GeometryPropagator

GeometryPropagator is a spacecraft-facing geometry, visibility, and thermal-prep engine for orbit-driven thermal analysis.

The current focus is geometric truth:
- orbit geometry in ECI and LVLH
- steady-state attitude laws and finite-rate slew transitions
- body-fixed CubeSat geometry with deployable mechanisms
- Earth-disk integration instead of scalar `cos(alpha)` approximations
- patch-resolved spacecraft self-occlusion
- clean handoff from realized geometry to downstream view-factor and thermal consumers

The package is intentionally split so that:
- `geometry` defines orbit, attitude, and spacecraft shape
- `viewfactor` computes what each realized surface patch can see
- `thermal` converts those geometric products into background heat inputs

The thermal solver is not the center of this repository. This repository is the geometry and visibility kernel that a planner or thermal balance layer can call.

## Current Status

Implemented now:
- orbit geometry and Sun/eclipse geometry
- steady-state attitude laws
- finite-rate mode-switch slews
- body-fixed `CubeSat` geometry builders
- hierarchical deployable-panel realization from hinge states
- explicit front and back surfaces for deployable solar-panel wings
- optional rigid geometry-frame -> body-frame mount transforms
- `mount(...)` helper for self-documenting axis-to-axis mounting
- stable surface identities preserved through realization and mount
- `RealizedGeometry` JSON export/import with provenance metadata
- nearest-hit ray queries against realized spacecraft surfaces
- patch-by-ray spacecraft self-occlusion masks
- patch-resolved directional integration with spacecraft blockage applied
- Earth-disk quadrature
- face-level directional masking
- patch-resolved rectangular radiator panels
- thermal background consumers built on top of geometric view products
- steady-state single-sided and two-sided patch temperature solvers
- transient two-sided solar-panel temperature solver
- effective sink-temperature products
- standalone 2-D thermal plots built directly on `SurfaceThermalProfile`
- notebook demos for geometry realization and body-face loading analysis
- legacy scalar flat-plate models retained under `geometry.legacy`

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
  - orbit.py
  - so3.py
  - laws.py
  - transitions.py
  - sampling.py
  - CubeSat/
  - legacy/
- viewfactor/
  - __init__.py
  - earthdisk.py
  - panel.py
  - occlusion.py
  - propagator.py
  - plots.py
- thermal/
  - __init__.py
  - background.py
  - solver.py
  - plots.py
- tests/
- docs/
- run_geometry.ipynb
- run_cubesat_geometry.ipynb
- background.ipynb
```

## Layer Boundaries

### 1. Geometry Layer
Owns spacecraft shape, mechanism state, orbit, and attitude laws.

Examples:
- body-fixed surfaces
- hinge trees and deployables
- mount transforms
- realized body-frame surfaces
- orbit and attitude state generation

### 2. View-Factor Layer
Owns geometric visibility and exchange only.

Examples:
- Earth view factor
- solar-array view (front and back wing faces grouped for reradiation)
- other-structure view
- deep-space visibility
- spacecraft self-occlusion

Outputs from this layer are geometric products, not watts.

### 3. Thermal Layer
Consumes view-factor outputs and source models.

Examples:
- Earth IR background
- albedo background
- solar background
- combined radiative background products for later thermal use
- single-sided steady-state patch temperature
- two-sided and transient solar-panel temperature
- thermal profile trace and heatmap plots

## Realized Geometry Handoff

The active interface boundary is `RealizedGeometry`.

Builder-side workflow:
1. Build body-fixed geometry.
2. Realize that geometry for one mechanism state.
3. Apply one body-frame mount transform.
4. Save the flat realized surface set to JSON.

Analysis-side workflow:
1. Load the saved `RealizedGeometry` JSON.
2. Select analysis surfaces by realized body-frame normal.
3. Propagate view factors from the loaded realized geometry.
4. Convert those geometric products into thermal background inputs.

This keeps the builder complexity upstream. Downstream code consumes a flat list of body-frame surfaces only.

Important interpretation rules:
- surface names such as `bus_+X` are stable geometry identities and are not renamed after mounting
- mounted body roles such as `body +Y bus face` should be selected from the realized geometry by the current face normal
- local patch plots use the realized surface frame: `center`, `normal`, `u_axis`, and the derived in-plane `v` direction

## Minimal Example

```python
import math
from datetime import datetime
from pathlib import Path

from geometry import (
    Orbit,
    RealizedGeometry,
    SlewModeSwitch,
    SunTracking,
    TargetTracking,
    build_6u_double_deployable,
    mount,
)
from geometry.CubeSat.inspect import surface_by_normal
from thermal import radiative_background, steady_state_temperature
from viewfactor import surface_loading_propagate

builder = build_6u_double_deployable()

realized = builder.realize(
    mechanism_state={
        'wing_port_inner_angle': math.pi / 2,
        'wing_port_outer_angle': math.pi,
        'wing_starboard_inner_angle': -math.pi / 2,
        'wing_starboard_outer_angle': -math.pi,
    },
    mount_rotation=mount('+Y', '+Z', '+Z', '+X'),
)

realized.to_json(Path('outputs') / 'spacecraft.json')

loaded = RealizedGeometry.from_json(Path('outputs') / 'spacecraft.json')
plus_y_face = surface_by_normal(loaded, [0.0, 1.0, 0.0], tag='bus')

orbit = Orbit.from_epoch(
    a=6771e3,
    i=math.radians(51.6),
    omega=math.radians(30.0),
    epoch=datetime(2025, 6, 21, 12, 0, 0),
)

law = SlewModeSwitch(
    TargetTracking(math.radians(266.4168), math.radians(-29.0078)),
    SunTracking(),
    slew_rate_deg_s=0.5,
)

profile = surface_loading_propagate(loaded, plus_y_face.name, orbit, law)
background = radiative_background(
    profile,
    solar_panel_temperature_K=300.0,
    solar_panel_emittance=0.9,
)
thermal_profile = steady_state_temperature(
    background,
    alpha_solar=0.8,
    epsilon=0.9,
)
```

## Notebook Workflow

`run_cubesat_geometry.ipynb`
- build the default 6U double-deployable geometry
- choose mechanism state and body mount with `mount(...)`
- inspect the mounted role table and 3D geometry with body axes
- confirm the saved artifact now includes both solar-panel cell sides and panel backs
- save `outputs/spacecraft.json`

`background.ipynb`
- load `outputs/spacecraft.json`
- resolve body-role faces by realized normal
- propagate front and back solar-panel surfaces separately
- solve a shared transient temperature for each wing using the two-sided panel model
- feed the bus radiators from the area-weighted mean solar-array temperature trace
- use `solar_panel_temperature_K=0` inside the panel solve itself as the current first-pass approximation
- inspect Earth-only view, solar-panel view, total background, and two-sided panel temperature
- animate the orbit progression sample-by-sample at `24 fps`

`run_geometry.ipynb`
- retain the simpler flat-plate / geometry sandbox workflow

## Quick Start

Run the smoke tests:

```bash
python -m unittest tests.test_geometry_package
```

Open the geometry notebook:

```bash
jupyter notebook run_cubesat_geometry.ipynb
```

Open the background notebook:

```bash
jupyter notebook background.ipynb
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
