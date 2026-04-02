# Roadmap

## Near-Term

### 1. Build the spacecraft geometry layer

Create a minimal but robust body-fixed geometry builder with:
- rectangular surfaces
- optional patch grids
- one-sided/two-sided surfaces
- hinged deployables
- realized geometry output in body coordinates

### 2. Extend the view-factor layer to use spacecraft occluders

Add local ray tests so radiator patches can be blocked by:
- deployable solar panels
- neighboring bus surfaces
- baffles
- louvers
- cavity walls

### 3. Keep the thermal layer separate

Do not mix thermal constants and material properties back into the geometry builder.

The view-factor layer should export geometric products that the thermal layer can consume.

## Mid-Term

### 4. Add surface-to-surface geometric exchange

Add patch-to-patch view products for:
- radiator to solar panel
- radiator to bus wall
- panel to panel

This should remain geometric only.

### 5. Add deep-space visibility products

Compute:
- visible space fraction
- blocked space fraction
- view to specific surface groups

These are useful for radiator effectiveness and reradiation.

### 6. Support richer local geometry

Add:
- explicit baffles
- louver-like directional acceptance
- arbitrary rectangular occluder groups

## Longer-Term

### 7. Harmonize with planner-style interfaces

Define stable input/output objects that mirror the planner style:
- geometry input state
- view-factor result
- thermal load result

### 8. Add a dedicated thermal consumer package

This repository should likely remain the geometry/view-factor engine.
The thermal balance solver can either live separately or in a clearly separate layer/package.

## Non-Goals for Now

- full CAD ingestion
- mesh engines
- hemicube rendering
- high-order thermal FEM

Those can come later if needed. The immediate goal is a minimal, robust, explainable geometry kernel.
