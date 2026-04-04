"""Persistence helpers for SurfaceLoadingProfile collections."""

import json
from pathlib import Path

import numpy as np

from .propagator import SurfaceLoadingProfile

_ARRAY_FIELDS = (
    'u', 'earth_view', 'albedo_view', 'solar_view',
    'solar_panel_view', 'other_structure_view', 'space_view', 'eclipse',
)


def _meta_path(npz_path):
    p = Path(npz_path)
    return p.parent / (p.stem + '_meta.json')


def save_profiles(profiles, npz_path, meta_path=None, **extra):
    """Persist SurfaceLoadingProfile list to .npz + companion JSON.

    Parameters
    ----------
    profiles  : list[SurfaceLoadingProfile]
    npz_path  : path-like — destination ``.npz`` file
    meta_path : path-like, optional — defaults to ``<stem>_meta.json``
    **extra   : additional scalar metadata written to the JSON
    """
    npz_path  = Path(npz_path)
    meta_path = Path(meta_path) if meta_path else _meta_path(npz_path)

    arrays = {}
    meta   = {'surfaces': [], **extra}
    for idx, p in enumerate(profiles):
        key = f's{idx}'
        for field in _ARRAY_FIELDS:
            arrays[f'{key}_{field}'] = getattr(p, field)
        meta['surfaces'].append({
            'index':  idx,
            'name':   p.surface_name,
            'width':  float(p.width),
            'height': float(p.height),
        })

    np.savez(npz_path, **arrays)
    meta_path.write_text(json.dumps(meta, indent=2))


def load_profiles(npz_path, meta_path=None):
    """Load SurfaceLoadingProfile list from .npz + companion JSON.

    Returns
    -------
    profiles : list[SurfaceLoadingProfile]
    meta     : dict  (includes any extra fields written by save_profiles)
    """
    npz_path  = Path(npz_path)
    meta_path = Path(meta_path) if meta_path else _meta_path(npz_path)

    data = np.load(npz_path)
    meta = json.loads(meta_path.read_text())

    profiles = []
    for s in meta['surfaces']:
        key = f"s{s['index']}"
        profiles.append(SurfaceLoadingProfile(
            surface_name=s['name'],
            u=data[f'{key}_u'],
            width=s['width'],
            height=s['height'],
            earth_view=data[f'{key}_earth_view'],
            albedo_view=data[f'{key}_albedo_view'],
            solar_view=data[f'{key}_solar_view'],
            solar_panel_view=data[f'{key}_solar_panel_view'],
            other_structure_view=data[f'{key}_other_structure_view'],
            space_view=data[f'{key}_space_view'],
            eclipse=data[f'{key}_eclipse'],
        ))
    return profiles, meta
