import math
import unittest
from datetime import datetime

import numpy as np

from geometry import (LVLHFixed, Orbit, RectangularPanel, SlewModeSwitch,
                      SunTracking, TargetTracking, panel_loading_propagate,
                      propagate, thermal_propagate)
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


if __name__ == '__main__':
    unittest.main()
