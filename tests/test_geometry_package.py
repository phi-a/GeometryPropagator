import math
import unittest
from datetime import datetime

import numpy as np

from geometry import (LVLHFixed, Orbit, RectangularPanel, SlewModeSwitch,
                      SO3,
                      SunTracking, TargetTracking, build_6u_double_deployable,
                      integrate_surface_response, panel_loading_propagate,
                      propagate, spacecraft_occlusion_mask, thermal_propagate)
from geometry.CubeSat import RealizedGeometry, RectSurface
from geometry.legacy import propagate as legacy_propagate
from geometry.legacy import thermal_propagate as legacy_thermal_propagate


SGR_A = (math.radians(266.4168), math.radians(-29.0078))


def _sample_orbit():
    return Orbit.from_epoch(
        a=6771e3,
        i=math.radians(51.6),
        omega=math.radians(30.0),
        epoch=datetime(2025, 6, 21, 12, 0, 0),
    )


class GeometryPackageTests(unittest.TestCase):
    def test_top_level_legacy_exports_still_exist(self):
        self.assertIs(propagate, legacy_propagate)
        self.assertIs(thermal_propagate, legacy_thermal_propagate)

    def test_open_panel_is_uniform(self):
        orbit = _sample_orbit()
        panel = RectangularPanel(width=20e-3, height=30e-3, nx=4, ny=6, wall_height=0.0)
        profile = panel_loading_propagate(
            orbit,
            LVLHFixed(),
            panel,
            face='+Y',
            n=12,
            n_mu=6,
            n_az=16,
        )
        snapshot = profile.view[0]
        self.assertTrue(np.allclose(snapshot, snapshot[0, 0]))

    def test_slew_begins_at_boundary(self):
        orbit = _sample_orbit()
        eclipse_law = TargetTracking(*SGR_A)
        sunlit_law = SunTracking()
        law = SlewModeSwitch(eclipse_law, sunlit_law, slew_rate_deg_s=0.5)

        transitions = law._transitions(orbit)
        self.assertEqual(len(transitions), 2)

        for index, transition in enumerate(transitions):
            from_law = sunlit_law if index == 0 else eclipse_law
            start_u = transition.start
            epsilon = 1e-6
            before = law(start_u - epsilon, orbit)
            at_boundary = law(start_u, orbit)
            after = law(start_u + min(epsilon, transition.span * 0.5), orbit)
            reference = from_law(start_u, orbit)

            self.assertTrue(np.allclose(before.m, reference.m, atol=1e-9))
            self.assertTrue(np.allclose(at_boundary.m, reference.m, atol=1e-9))
            self.assertFalse(np.allclose(after.m, at_boundary.m, atol=1e-9))

    def test_default_6u_builder_realizes_deployables(self):
        cubesat = build_6u_double_deployable()
        realized = cubesat.realize()

        self.assertEqual(len(realized.surfaces), 10)
        self.assertIn('wing_port_inner', realized.names())
        self.assertIn('wing_starboard_outer', realized.names())

        port_inner = realized.by_name('wing_port_inner')
        port_outer = realized.by_name('wing_port_outer')
        star_inner = realized.by_name('wing_starboard_inner')
        star_outer = realized.by_name('wing_starboard_outer')

        self.assertTrue(np.allclose(port_inner.normal, [0.0, 1.0, 0.0], atol=1e-9))
        self.assertTrue(np.allclose(port_outer.normal, [0.0, 1.0, 0.0], atol=1e-9))
        self.assertTrue(np.allclose(star_inner.normal, [0.0, 1.0, 0.0], atol=1e-9))
        self.assertTrue(np.allclose(star_outer.normal, [0.0, 1.0, 0.0], atol=1e-9))

        self.assertAlmostEqual(cubesat.metadata['leaf_y_m'], 0.2263)
        self.assertAlmostEqual(cubesat.metadata['leaf_z_m'], 0.3405)
        self.assertAlmostEqual(port_inner.height, 0.2263)
        self.assertAlmostEqual(port_inner.width, 0.3405)

        self.assertEqual(len(realized.by_tag('solar_panel')), 4)

        self.assertGreater(port_outer.center[0], port_inner.center[0])
        self.assertLess(star_outer.center[0], star_inner.center[0])

    def test_builder_accepts_explicit_leaf_dimensions(self):
        cubesat = build_6u_double_deployable(leaf_y=0.180, leaf_z=0.300)
        realized = cubesat.realize()
        port_inner = realized.by_name('wing_port_inner')

        self.assertAlmostEqual(cubesat.metadata['leaf_y_m'], 0.180)
        self.assertAlmostEqual(cubesat.metadata['leaf_z_m'], 0.300)
        self.assertAlmostEqual(port_inner.height, 0.180)
        self.assertAlmostEqual(port_inner.width, 0.300)

    def test_realized_geometry_mount_applies_rigid_transform(self):
        source = RectSurface(
            name='source',
            center=np.array([1.0, 0.0, 0.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            u_axis=np.array([0.0, 0.0, 1.0]),
            width=1.0,
            height=2.0,
        )
        realized = RealizedGeometry((source,))

        mounted = realized.mounted(
            rotation=SO3.Rz(math.pi / 2.0),
            offset=np.array([0.0, 1.0, 0.0]),
        )
        transformed = mounted.by_name('source')

        self.assertTrue(np.allclose(transformed.center, [0.0, 2.0, 0.0], atol=1e-9))
        self.assertTrue(np.allclose(transformed.normal, [0.0, 1.0, 0.0], atol=1e-9))
        self.assertTrue(np.allclose(transformed.u_axis, [0.0, 0.0, 1.0], atol=1e-9))

    def test_builder_realize_accepts_optional_mount_transform(self):
        cubesat = build_6u_double_deployable()
        plain = cubesat.realize()
        mounted = cubesat.realize(
            mount_rotation=SO3.Rz(math.pi / 2.0),
            mount_offset=np.array([0.0, 0.1, 0.0]),
        )

        plain_face = plain.by_name('bus_+X')
        mounted_face = mounted.by_name('bus_+X')

        expected_center = np.array([0.0, 0.15, 0.0])
        expected_normal = np.array([0.0, 1.0, 0.0])
        expected_u = np.array([0.0, 0.0, 1.0])

        self.assertTrue(np.allclose(mounted_face.center, expected_center, atol=1e-9))
        self.assertTrue(np.allclose(mounted_face.normal, expected_normal, atol=1e-9))
        self.assertTrue(np.allclose(mounted_face.u_axis, expected_u, atol=1e-9))
        self.assertFalse(np.allclose(mounted_face.center, plain_face.center, atol=1e-9))

    def test_spacecraft_occlusion_mask_blocks_parallel_rectangle(self):
        source = RectSurface(
            name='source',
            center=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            width=2.0,
            height=2.0,
            patch_shape=(2, 2),
        )
        blocker = RectSurface(
            name='blocker',
            center=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            width=2.0,
            height=2.0,
        )
        realized = RealizedGeometry((source, blocker))
        dirs_body = np.array([[0.0, 0.0, 1.0]])

        visible = spacecraft_occlusion_mask(realized, 'source', dirs_body)

        self.assertEqual(visible.shape, (4, 1))
        self.assertTrue(np.all(~visible))

    def test_integrate_surface_response_respects_occlusion(self):
        source = RectSurface(
            name='source',
            center=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            width=1.0,
            height=1.0,
            patch_shape=(1, 1),
        )
        blocker = RectSurface(
            name='blocker',
            center=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            width=1.0,
            height=1.0,
        )
        clear = RealizedGeometry((source,))
        blocked = RealizedGeometry((source, blocker))

        dirs_body = np.array([[0.0, 0.0, 1.0]])
        solid_angle_weights = np.array([math.pi])

        clear_value = integrate_surface_response(
            clear,
            'source',
            dirs_body,
            solid_angle_weights,
        )
        blocked_value = integrate_surface_response(
            blocked,
            'source',
            dirs_body,
            solid_angle_weights,
        )

        self.assertTrue(np.allclose(clear_value, [[1.0]], atol=1e-12))
        self.assertTrue(np.allclose(blocked_value, [[0.0]], atol=1e-12))


if __name__ == '__main__':
    unittest.main()
