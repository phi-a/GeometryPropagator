"""Default CubeSat geometry builders."""

import math

import numpy as np

from .surfaces import CubeSatGeometry, RectSurface, SurfaceNode


def build_6u_double_deployable(*,
                               bus_x=0.100,
                               bus_y=0.2263,
                               bus_z=0.3405,
                               wing_span=0.100,
                               wing_length=None,
                               bus_patch_shape=None,
                               wing_patch_shape=None):
    """Build a 6U CubeSat with two double-leaf deployable solar-panel wings.

    The default example is a 6U bus with one two-leaf wing on each side,
    attached along the top-edge 3U rail and deployed into a common plane.
    """
    wing_length = bus_z if wing_length is None else wing_length

    nodes = []

    def add_root(name, center, normal, u_axis, width, height, tags=(), patch_shape=None):
        nodes.append(SurfaceNode(
            surface=RectSurface(
                name=name,
                center=np.asarray(center, dtype=float),
                normal=np.asarray(normal, dtype=float),
                u_axis=np.asarray(u_axis, dtype=float),
                width=width,
                height=height,
                patch_shape=patch_shape,
                tags=tags,
            )
        ))

    add_root(
        'bus_+X',
        [0.5 * bus_x, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        bus_z,
        bus_y,
        tags=('bus', '+X'),
        patch_shape=bus_patch_shape,
    )
    add_root(
        'bus_-X',
        [-0.5 * bus_x, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        bus_z,
        bus_y,
        tags=('bus', '-X'),
        patch_shape=bus_patch_shape,
    )
    add_root(
        'bus_+Y',
        [0.0, 0.5 * bus_y, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        bus_z,
        bus_x,
        tags=('bus', '+Y'),
        patch_shape=bus_patch_shape,
    )
    add_root(
        'bus_-Y',
        [0.0, -0.5 * bus_y, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        bus_z,
        bus_x,
        tags=('bus', '-Y'),
        patch_shape=bus_patch_shape,
    )
    add_root(
        'bus_+Z',
        [0.0, 0.0, 0.5 * bus_z],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        bus_x,
        bus_y,
        tags=('bus', '+Z'),
        patch_shape=bus_patch_shape,
    )
    add_root(
        'bus_-Z',
        [0.0, 0.0, -0.5 * bus_z],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        bus_x,
        bus_y,
        tags=('bus', '-Z'),
        patch_shape=bus_patch_shape,
    )

    # Port-side inner wing, stowed against +X and deployed +90 deg into +Y.
    nodes.append(SurfaceNode(
        surface=RectSurface(
            name='wing_port_inner',
            center=np.array([0.5 * bus_x, 0.5 * bus_y - 0.5 * wing_span, 0.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            u_axis=np.array([0.0, 0.0, 1.0]),
            width=wing_length,
            height=wing_span,
            patch_shape=wing_patch_shape,
            tags=('deployable', 'solar_panel', 'port', 'inner'),
        ),
        hinge_origin=np.array([0.5 * bus_x, 0.5 * bus_y, 0.0]),
        hinge_axis=np.array([0.0, 0.0, 1.0]),
        state_key='wing_port_inner_angle',
        default_angle=math.pi / 2.0,
    ))

    # Outer leaf is defined in the realized frame of the inner leaf.
    nodes.append(SurfaceNode(
        surface=RectSurface(
            name='wing_port_outer',
            center=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            width=wing_length,
            height=wing_span,
            patch_shape=wing_patch_shape,
            tags=('deployable', 'solar_panel', 'port', 'outer'),
        ),
        parent='wing_port_inner',
        hinge_origin=np.array([0.0, 0.5 * wing_span, 0.0]),
        hinge_axis=np.array([1.0, 0.0, 0.0]),
        state_key='wing_port_outer_angle',
        default_angle=math.pi,
    ))

    # Starboard-side inner wing, deployed -90 deg so both wings face +Y.
    nodes.append(SurfaceNode(
        surface=RectSurface(
            name='wing_starboard_inner',
            center=np.array([-0.5 * bus_x, 0.5 * bus_y - 0.5 * wing_span, 0.0]),
            normal=np.array([-1.0, 0.0, 0.0]),
            u_axis=np.array([0.0, 0.0, 1.0]),
            width=wing_length,
            height=wing_span,
            patch_shape=wing_patch_shape,
            tags=('deployable', 'solar_panel', 'starboard', 'inner'),
        ),
        hinge_origin=np.array([-0.5 * bus_x, 0.5 * bus_y, 0.0]),
        hinge_axis=np.array([0.0, 0.0, 1.0]),
        state_key='wing_starboard_inner_angle',
        default_angle=-math.pi / 2.0,
    ))

    nodes.append(SurfaceNode(
        surface=RectSurface(
            name='wing_starboard_outer',
            center=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            width=wing_length,
            height=wing_span,
            patch_shape=wing_patch_shape,
            tags=('deployable', 'solar_panel', 'starboard', 'outer'),
        ),
        parent='wing_starboard_inner',
        hinge_origin=np.array([0.0, -0.5 * wing_span, 0.0]),
        hinge_axis=np.array([1.0, 0.0, 0.0]),
        state_key='wing_starboard_outer_angle',
        default_angle=-math.pi,
    ))

    return CubeSatGeometry(
        nodes=tuple(nodes),
        metadata={
            'example': '6U_double_deployable',
            'bus_dimensions_m': (bus_x, bus_y, bus_z),
            'wing_span_m': wing_span,
            'wing_length_m': wing_length,
        },
    )
