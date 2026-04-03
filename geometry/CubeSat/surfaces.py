"""Minimal CubeSat geometry layer built from rectangular surfaces."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..so3 import SO3


BODY_FRAME_LABEL = "+X=velocity, +Y=orbit_normal, +Z=zenith"

_AXIS_MAP = (
    ('+X', np.array([1.0, 0.0, 0.0])),
    ('-X', np.array([-1.0, 0.0, 0.0])),
    ('+Y', np.array([0.0, 1.0, 0.0])),
    ('-Y', np.array([0.0, -1.0, 0.0])),
    ('+Z', np.array([0.0, 0.0, 1.0])),
    ('-Z', np.array([0.0, 0.0, -1.0])),
)
_AXIS_LOOKUP = {label: axis for label, axis in _AXIS_MAP}


def _as_vec3(v):
    arr = np.asarray(v, dtype=float).reshape(3)
    return arr


def _unit(v):
    arr = _as_vec3(v)
    n = np.linalg.norm(arr)
    if n <= 1e-15:
        raise ValueError("vector norm must be positive")
    return arr / n


def _as_rotation_matrix(rotation):
    if rotation is None:
        return np.eye(3)
    if hasattr(rotation, 'm'):
        matrix = np.asarray(rotation.m, dtype=float)
    else:
        matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix or an SO3-like object")
    return matrix


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _axis_vector_from_label(label):
    if label not in _AXIS_LOOKUP:
        raise ValueError(f"axis label must be one of {tuple(_AXIS_LOOKUP)}")
    return _AXIS_LOOKUP[label].copy()


def _rotation_about_axis(axis, angle):
    axis = _unit(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ])


def _rotate_point_about_line(point, origin, axis, angle):
    rot = _rotation_about_axis(axis, angle)
    return _as_vec3(origin) + rot @ (_as_vec3(point) - _as_vec3(origin))


def _canonical_perpendicular_axis(axis):
    axis = _unit(axis)
    for candidate in (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ):
        perp = candidate - np.dot(candidate, axis) * axis
        if np.linalg.norm(perp) > 1e-12:
            return _unit(perp)
    raise ValueError("failed to choose a perpendicular axis")


def _minimal_axis_alignment(source_axis, target_axis):
    source_axis = _unit(source_axis)
    target_axis = _unit(target_axis)
    dot = float(np.clip(np.dot(source_axis, target_axis), -1.0, 1.0))
    if dot >= 1.0 - 1e-12:
        return np.eye(3)
    if dot <= -1.0 + 1e-12:
        return _rotation_about_axis(_canonical_perpendicular_axis(source_axis), math.pi)

    cross = np.cross(source_axis, target_axis)
    angle = math.atan2(np.linalg.norm(cross), dot)
    return _rotation_about_axis(cross, angle)


def mount(geom_axis, body_axis, geom_axis2=None, body_axis2=None):
    """Return a rigid geometry-frame -> body-frame alignment rotation.

    Parameters
    ----------
    geom_axis, body_axis : {'+X', '-X', '+Y', '-Y', '+Z', '-Z'}
        Primary axis mapping, enforced exactly.
    geom_axis2, body_axis2 : {'+X', '-X', '+Y', '-Y', '+Z', '-Z'}, optional
        Optional secondary axis mapping used to fully constrain the rotation.
    """
    source_primary = _axis_vector_from_label(geom_axis)
    target_primary = _axis_vector_from_label(body_axis)

    if (geom_axis2 is None) != (body_axis2 is None):
        raise ValueError("geom_axis2 and body_axis2 must be provided together")

    if geom_axis2 is None:
        return SO3(_minimal_axis_alignment(source_primary, target_primary))

    source_secondary = _axis_vector_from_label(geom_axis2)
    target_secondary = _axis_vector_from_label(body_axis2)
    if abs(np.dot(source_primary, source_secondary)) > 1e-12:
        raise ValueError("geom_axis2 must be orthogonal to geom_axis")
    if abs(np.dot(target_primary, target_secondary)) > 1e-12:
        raise ValueError("body_axis2 must be orthogonal to body_axis")

    source_basis = np.column_stack([
        source_primary,
        source_secondary,
        np.cross(source_primary, source_secondary),
    ])
    target_basis = np.column_stack([
        target_primary,
        target_secondary,
        np.cross(target_primary, target_secondary),
    ])
    rotation = target_basis @ source_basis.T
    if not np.allclose(rotation @ source_primary, target_primary, atol=1e-12):
        raise ValueError("failed to align the requested primary axis pair")
    if not np.allclose(rotation @ source_secondary, target_secondary, atol=1e-12):
        raise ValueError("failed to align the requested secondary axis pair")
    return SO3(rotation)


def _surface_to_dict(surface):
    return {
        'name': surface.name,
        'center': _jsonify(surface.center),
        'normal': _jsonify(surface.normal),
        'u_axis': _jsonify(surface.u_axis),
        'width': float(surface.width),
        'height': float(surface.height),
        'two_sided': bool(surface.two_sided),
        'patch_shape': None if surface.patch_shape is None else list(surface.patch_shape),
        'tags': list(surface.tags),
    }


def _surface_from_dict(data):
    return RectSurface(
        name=data['name'],
        center=data['center'],
        normal=data['normal'],
        u_axis=data['u_axis'],
        width=float(data['width']),
        height=float(data['height']),
        two_sided=bool(data.get('two_sided', False)),
        patch_shape=None if data.get('patch_shape') is None else tuple(data['patch_shape']),
        tags=tuple(data.get('tags', ())),
    )


def _compose_mount_metadata(metadata, rotation, offset):
    combined = dict(metadata)
    previous_rotation = _as_rotation_matrix(combined.get('mount_rotation'))
    previous_offset = (
        np.zeros(3, dtype=float)
        if combined.get('mount_offset') is None
        else _as_vec3(combined['mount_offset'])
    )
    combined['mount_rotation'] = _jsonify(rotation @ previous_rotation)
    combined['mount_offset'] = _jsonify(offset + rotation @ previous_offset)
    return combined


@dataclass(frozen=True)
class RectSurface:
    """Rectangular surface defined in an arbitrary parent frame.

    Parameters
    ----------
    name : str
        Surface identifier.
    center : array-like (3,)
        Surface center in the parent frame.
    normal : array-like (3,)
        Surface outward normal in the parent frame.
    u_axis : array-like (3,)
        In-plane width direction in the parent frame.
    width, height : float
        Surface dimensions [m].
    two_sided : bool, optional
        Whether both sides are radiatively active.
    patch_shape : tuple[int, int], optional
        Patch grid `(nx, ny)`. When omitted, the surface is treated as one patch.
    tags : tuple[str, ...], optional
        Labels for grouping/querying surfaces.
    """
    name: str
    center: np.ndarray
    normal: np.ndarray
    u_axis: np.ndarray
    width: float
    height: float
    two_sided: bool = False
    patch_shape: tuple[int, int] | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("surface width and height must be positive")

        center = _as_vec3(self.center)
        normal = _unit(self.normal)
        u_axis = _as_vec3(self.u_axis)
        u_axis = u_axis - np.dot(u_axis, normal) * normal
        u_norm = np.linalg.norm(u_axis)
        if u_norm <= 1e-15:
            raise ValueError("u_axis must not be parallel to normal")
        u_axis = u_axis / u_norm

        if self.patch_shape is None:
            patch_shape = None
        else:
            nx, ny = self.patch_shape
            nx = int(nx)
            ny = int(ny)
            if nx <= 0 or ny <= 0:
                raise ValueError("patch_shape entries must be positive")
            patch_shape = (nx, ny)

        object.__setattr__(self, 'center', center)
        object.__setattr__(self, 'normal', normal)
        object.__setattr__(self, 'u_axis', u_axis)
        object.__setattr__(self, 'patch_shape', patch_shape)
        object.__setattr__(self, 'tags', tuple(self.tags))

    @property
    def v_axis(self):
        return np.cross(self.normal, self.u_axis)

    @property
    def frame_matrix(self):
        """Columns are `(u_axis, v_axis, normal)`."""
        return np.column_stack([self.u_axis, self.v_axis, self.normal])

    @property
    def area(self):
        return self.width * self.height

    def corners(self):
        """Return the four corner coordinates in the current frame."""
        du = 0.5 * self.width * self.u_axis
        dv = 0.5 * self.height * self.v_axis
        return np.array([
            self.center - du - dv,
            self.center + du - dv,
            self.center + du + dv,
            self.center - du + dv,
        ])

    def patch_centers(self):
        """Return patch-center coordinates with shape `(ny, nx, 3)`."""
        nx, ny = self.patch_shape if self.patch_shape is not None else (1, 1)
        x = np.linspace(-0.5 * self.width + 0.5 * self.width / nx,
                        0.5 * self.width - 0.5 * self.width / nx,
                        nx)
        y = np.linspace(-0.5 * self.height + 0.5 * self.height / ny,
                        0.5 * self.height - 0.5 * self.height / ny,
                        ny)
        xx, yy = np.meshgrid(x, y, indexing='xy')
        centers = (
            self.center[None, None, :]
            + xx[..., None] * self.u_axis[None, None, :]
            + yy[..., None] * self.v_axis[None, None, :]
        )
        return centers

    def patch_normals(self):
        """Return patch normals with shape `(ny, nx, 3)`."""
        nx, ny = self.patch_shape if self.patch_shape is not None else (1, 1)
        return np.broadcast_to(self.normal, (ny, nx, 3))

    def patch_area(self):
        nx, ny = self.patch_shape if self.patch_shape is not None else (1, 1)
        return self.area / (nx * ny)

    def ray_intersection_parameter(self, origin, direction, eps=1e-12):
        """Return the positive ray parameter to the rectangle, or `None`."""
        origin = _as_vec3(origin)
        direction = _unit(direction)
        denom = float(np.dot(direction, self.normal))
        if abs(denom) <= eps:
            return None

        t_hit = float(np.dot(self.center - origin, self.normal) / denom)
        if t_hit <= eps:
            return None

        hit = origin + t_hit * direction
        rel = hit - self.center
        u = float(np.dot(rel, self.u_axis))
        v = float(np.dot(rel, self.v_axis))
        if abs(u) > 0.5 * self.width + eps or abs(v) > 0.5 * self.height + eps:
            return None
        return t_hit


@dataclass(frozen=True)
class SurfaceNode:
    """Surface plus optional hinge state, expressed in its parent frame.

    For root nodes (`parent is None`), the parent frame is the spacecraft
    body frame. For child nodes, the parent frame is the realized surface
    frame of `parent`, whose basis is `(u_axis, v_axis, normal)`.
    """
    surface: RectSurface
    parent: str | None = None
    hinge_origin: np.ndarray | None = None
    hinge_axis: np.ndarray | None = None
    state_key: str | None = None
    default_angle: float = 0.0

    def __post_init__(self):
        if (self.hinge_origin is None) != (self.hinge_axis is None):
            raise ValueError("hinge_origin and hinge_axis must be provided together")
        if self.hinge_origin is not None:
            object.__setattr__(self, 'hinge_origin', _as_vec3(self.hinge_origin))
            object.__setattr__(self, 'hinge_axis', _unit(self.hinge_axis))


@dataclass(frozen=True)
class RealizedGeometry:
    """Realized body-frame geometry built from a `CubeSatGeometry`."""
    surfaces: tuple[RectSurface, ...]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'surfaces', tuple(self.surfaces))
        object.__setattr__(self, 'metadata', dict(self.metadata))

    def by_name(self, name):
        for surface in self.surfaces:
            if surface.name == name:
                return surface
        raise KeyError(name)

    def names(self):
        return tuple(surface.name for surface in self.surfaces)

    def by_tag(self, tag):
        return tuple(surface for surface in self.surfaces if tag in surface.tags)

    def to_json(self, path):
        """Write the realized geometry and provenance metadata to JSON."""
        payload = dict(_jsonify(self.metadata))
        payload['surfaces'] = [_surface_to_dict(surface) for surface in self.surfaces]

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
        return path

    @classmethod
    def from_json(cls, path):
        """Load realized geometry and provenance metadata from JSON."""
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        surfaces = tuple(_surface_from_dict(item) for item in payload.pop('surfaces'))
        return cls(surfaces=surfaces, metadata=payload)

    def mounted(self, *, rotation=None, offset=None):
        """Return a rigidly transformed copy of the realized geometry.

        This is intended as an optional geometry-frame -> body-frame mount
        step. When both arguments are omitted, the geometry is returned
        unchanged.
        """
        rot = _as_rotation_matrix(rotation)
        shift = np.zeros(3, dtype=float) if offset is None else _as_vec3(offset)
        if np.allclose(rot, np.eye(3)) and np.allclose(shift, 0.0):
            return self

        transformed = []
        for surface in self.surfaces:
            transformed.append(
                RectSurface(
                    name=surface.name,
                    center=shift + rot @ surface.center,
                    normal=rot @ surface.normal,
                    u_axis=rot @ surface.u_axis,
                    width=surface.width,
                    height=surface.height,
                    two_sided=surface.two_sided,
                    patch_shape=surface.patch_shape,
                    tags=surface.tags,
                )
            )
        metadata = _compose_mount_metadata(self.metadata, rot, shift)
        return RealizedGeometry(tuple(transformed), metadata=metadata)

    def first_intersection(self, origin, direction, *, exclude=()):
        """Return `(surface, t)` for the nearest intersected surface, if any."""
        blocked = set(exclude)
        best = None
        best_t = None
        for surface in self.surfaces:
            if surface.name in blocked:
                continue
            t_hit = surface.ray_intersection_parameter(origin, direction)
            if t_hit is None:
                continue
            if best is None or t_hit < best_t:
                best = surface
                best_t = t_hit
        if best is None:
            return None
        return best, best_t


@dataclass(frozen=True)
class CubeSatGeometry:
    """Hierarchical CubeSat geometry composed of rectangular surfaces."""
    nodes: tuple[SurfaceNode, ...]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        nodes = tuple(self.nodes)
        names = [node.surface.name for node in nodes]
        if len(names) != len(set(names)):
            raise ValueError("surface names must be unique")
        name_set = set(names)
        for node in nodes:
            if node.parent is not None and node.parent not in name_set:
                raise ValueError(f"unknown parent surface {node.parent!r}")
        object.__setattr__(self, 'nodes', nodes)
        object.__setattr__(self, 'metadata', dict(self.metadata))

    def default_state(self):
        state = {}
        for node in self.nodes:
            if node.state_key is not None:
                state[node.state_key] = node.default_angle
        return state

    def realize(self, state=None, *, mount_rotation=None, mount_offset=None):
        """Realize all surfaces for the given mechanism state.

        By default the geometry is realized directly in the body frame.
        Optionally, apply one rigid geometry-frame -> body-frame mount
        transform through `mount_rotation` and `mount_offset`.
        """
        if state is None:
            state = {}
        state = {**self.default_state(), **state}
        mechanism_state = {
            node.state_key: float(state[node.state_key])
            for node in self.nodes
            if node.state_key is not None
        }
        node_map = {node.surface.name: node for node in self.nodes}
        cache = {}

        def resolve(name):
            if name in cache:
                return cache[name]

            node = node_map[name]
            if node.parent is None:
                base_rot = np.eye(3)
                base_origin = np.zeros(3, dtype=float)
            else:
                _, base_rot, base_origin = resolve(node.parent)

            center_local = node.surface.center
            normal_local = node.surface.normal
            u_local = node.surface.u_axis

            if node.hinge_axis is not None:
                angle = mechanism_state.get(node.state_key, node.default_angle)
                center_local = _rotate_point_about_line(
                    center_local, node.hinge_origin, node.hinge_axis, angle
                )
                rot_local = _rotation_about_axis(node.hinge_axis, angle)
                normal_local = rot_local @ normal_local
                u_local = rot_local @ u_local

            center_body = base_origin + base_rot @ center_local
            normal_body = base_rot @ normal_local
            u_body = base_rot @ u_local
            realized = RectSurface(
                name=node.surface.name,
                center=center_body,
                normal=normal_body,
                u_axis=u_body,
                width=node.surface.width,
                height=node.surface.height,
                two_sided=node.surface.two_sided,
                patch_shape=node.surface.patch_shape,
                tags=node.surface.tags,
            )
            cache[name] = (realized, realized.frame_matrix, realized.center)
            return cache[name]

        realized_surfaces = tuple(resolve(node.surface.name)[0] for node in self.nodes)
        realized = RealizedGeometry(
            realized_surfaces,
            metadata={
                'geometry_name': self.metadata.get('example'),
                'body_frame': BODY_FRAME_LABEL,
                'mechanism_state': _jsonify(mechanism_state),
                'mount_rotation': _jsonify(np.eye(3)),
                'mount_offset': _jsonify(np.zeros(3, dtype=float)),
                'builder_metadata': _jsonify(self.metadata),
            },
        )
        return realized.mounted(rotation=mount_rotation, offset=mount_offset)
