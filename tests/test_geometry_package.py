import math
import unittest
from datetime import datetime

import numpy as np

from geometry import (LVLHFixed, Orbit, RectangularPanel, SlewModeSwitch,
                      SunTracking, TargetTracking, build_6u_double_deployable,
                      panel_loading_propagate, propagate, thermal_propagate)
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

        self.assertEqual(len(realized.by_tag('solar_panel')), 4)

        self.assertGreater(port_outer.center[0], port_inner.center[0])
        self.assertLess(star_outer.center[0], star_inner.center[0])


if __name__ == '__main__':
    unittest.main()
