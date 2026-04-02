"""Minimal CubeSat geometry layer built from rectangular surfaces."""

import math
from dataclasses import dataclass, field

import numpy as np


def _as_vec3(v):
    arr = np.asarray(v, dtype=float).reshape(3)
    return arr


def _unit(v):
    arr = _as_vec3(v)
    n = np.linalg.norm(arr)
    if n <= 1e-15:
        raise ValueError("vector norm must be positive")
    return arr / n


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

    def by_name(self, name):
        for surface in self.surfaces:
            if surface.name == name:
                return surface
        raise KeyError(name)

    def names(self):
        return tuple(surface.name for surface in self.surfaces)

    def by_tag(self, tag):
        return tuple(surface for surface in self.surfaces if tag in surface.tags)

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

    def realize(self, state=None):
        """Realize all surfaces in the body frame for the given mechanism state."""
        if state is None:
            state = {}
        state = {**self.default_state(), **state}
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
                angle = float(state.get(node.state_key, node.default_angle))
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
        return RealizedGeometry(realized_surfaces)
