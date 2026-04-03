import math
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from geometry import (LVLHFixed, Orbit, SlewModeSwitch, SO3,
                      SunTracking, TargetTracking, build_6u_double_deployable,
                      mount,
                      propagate, thermal_propagate)
from geometry.CubeSat import (CubeSatGeometry, RealizedGeometry, RectSurface,
                              SurfaceNode, surface_body_role, surface_by_normal)
from geometry.sampling import propagation_grid
from geometry.legacy import propagate as legacy_propagate
from geometry.legacy import thermal_propagate as legacy_thermal_propagate
from viewfactor import (EarthDiskQuadrature, RectangularPanel, panel_loading_propagate,
                        spacecraft_occlusion_mask, integrate_surface_response,
                        hemisphere_group_view, SurfaceLoadingProfile,
                        surface_loading_propagate)
import viewfactor.occlusion as occlusion_impl
from thermal import SIGMA_SB, radiative_background


SGR_A = (math.radians(266.4168), math.radians(-29.0078))


def _sample_orbit():
    return Orbit.from_epoch(
        a=6771e3,
        i=math.radians(51.6),
        omega=math.radians(30.0),
        epoch=datetime(2025, 6, 21, 12, 0, 0),
    )


def _simple_geometry(*, blocker_tags=(), include_blocker=True):
    nodes = [
        SurfaceNode(
            RectSurface(
                name='source',
                center=np.array([0.0, 0.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                u_axis=np.array([1.0, 0.0, 0.0]),
                width=1.0,
                height=1.0,
                patch_shape=(1, 1),
            )
        )
    ]
    if include_blocker:
        nodes.append(
            SurfaceNode(
                RectSurface(
                    name='blocker',
                    center=np.array([0.0, 0.0, 1.0]),
                    normal=np.array([0.0, 0.0, -1.0]),
                    u_axis=np.array([1.0, 0.0, 0.0]),
                    width=1.0,
                    height=1.0,
                    tags=blocker_tags,
                )
            )
        )
    return CubeSatGeometry(tuple(nodes))


def _resolve_source(realized, source):
    if isinstance(source, str):
        return realized.by_name(source)
    return source


def _scalar_occlusion_mask(realized, source, dirs_body, *, exclude=(), eps=1e-9):
    dirs_body = np.asarray(dirs_body, dtype=float)
    surface = _resolve_source(realized, source)
    centers = surface.patch_centers().reshape(-1, 3)
    normals = np.broadcast_to(surface.normal, centers.shape)
    visible = np.zeros((centers.shape[0], dirs_body.shape[0]), dtype=bool)

    ignored = set(exclude)
    ignored.add(surface.name)
    ignored_tuple = tuple(ignored)

    for patch_index, (center, normal) in enumerate(zip(centers, normals)):
        dots = dirs_body @ normal
        for ray_index, dot in enumerate(dots):
            origin = occlusion_impl._ray_origin(
                center,
                normal,
                dot,
                two_sided=surface.two_sided,
                eps=eps,
            )
            if origin is None:
                continue
            hit = realized.first_intersection(
                origin,
                dirs_body[ray_index],
                exclude=ignored_tuple,
            )
            visible[patch_index, ray_index] = hit is None

    return visible


def _scalar_integrate_surface_response(realized, source, dirs_body, solid_angle_weights, *,
                                       sample_weight=None, exclude=(), eps=1e-9):
    dirs_body = np.asarray(dirs_body, dtype=float)
    solid_angle_weights = np.asarray(solid_angle_weights, dtype=float)
    surface = _resolve_source(realized, source)
    visibility = _scalar_occlusion_mask(
        realized,
        surface,
        dirs_body,
        exclude=exclude,
        eps=eps,
    ).astype(float)

    cosine = dirs_body @ surface.normal
    if surface.two_sided:
        cosine = np.abs(cosine)
    else:
        cosine = np.clip(cosine, 0.0, None)

    kernel = visibility * cosine[None, :]
    if sample_weight is not None:
        kernel = kernel * np.asarray(sample_weight, dtype=float)[None, :]

    values = np.sum(kernel * solid_angle_weights[None, :], axis=1) / math.pi
    ny, nx = surface.patch_centers().shape[:2]
    return values.reshape(ny, nx)


def _scalar_hemisphere_group_view(realized, source, group_tags, *,
                                  n_az=73, n_el=33,
                                  elevation_min_deg=5.0,
                                  elevation_max_deg=85.0,
                                  exclude=(), eps=1e-9):
    surface = _resolve_source(realized, source)
    dirs_body, _, cosine_weights = occlusion_impl._hemisphere_quadrature(
        surface,
        n_az=n_az,
        n_el=n_el,
        elevation_min_deg=elevation_min_deg,
        elevation_max_deg=elevation_max_deg,
    )
    weighted_kernel = cosine_weights / math.pi

    group_names = []
    for _, group_name in group_tags:
        if group_name not in group_names:
            group_names.append(group_name)

    centers = surface.patch_centers().reshape(-1, 3)
    normals = np.broadcast_to(surface.normal, centers.shape)
    grouped = {name: np.zeros(centers.shape[0], dtype=float) for name in group_names}
    other_structure = np.zeros(centers.shape[0], dtype=float)
    space_view = np.zeros(centers.shape[0], dtype=float)

    ignored = set(exclude)
    ignored.add(surface.name)
    ignored_tuple = tuple(ignored)

    for patch_index, (center, normal) in enumerate(zip(centers, normals)):
        dots = dirs_body @ normal
        for ray_index, dot in enumerate(dots):
            origin = occlusion_impl._ray_origin(
                center,
                normal,
                dot,
                two_sided=surface.two_sided,
                eps=eps,
            )
            if origin is None:
                continue
            hit = realized.first_intersection(
                origin,
                dirs_body[ray_index],
                exclude=ignored_tuple,
            )
            if hit is None:
                space_view[patch_index] += weighted_kernel[ray_index]
                continue

            hit_surface, _ = hit
            group_name = None
            for tag, candidate_name in group_tags:
                if tag in hit_surface.tags:
                    group_name = candidate_name
                    break
            if group_name is None:
                other_structure[patch_index] += weighted_kernel[ray_index]
            else:
                grouped[group_name][patch_index] += weighted_kernel[ray_index]

    ny, nx = surface.patch_centers().shape[:2]
    result = {name: values.reshape(ny, nx) for name, values in grouped.items()}
    result['other_structure_view'] = other_structure.reshape(ny, nx)
    result['space_view'] = space_view.reshape(ny, nx)
    return result


def _scalar_surface_loading_reference(realized, surface_name, orbit, law, *,
                                      n=180, n_mu=24, n_az=72,
                                      hemi_n_az=73, hemi_n_el=33,
                                      hemi_elevation_min_deg=5.0,
                                      hemi_elevation_max_deg=85.0):
    surface = realized.by_name(surface_name)
    u_arr = propagation_grid(orbit, law, n)
    quad = EarthDiskQuadrature.build(orbit.rho, n_mu=n_mu, n_az=n_az)
    sun_eci = orbit.sun_eci()
    static_views = _scalar_hemisphere_group_view(
        realized,
        surface,
        [('solar_panel', 'solar_panel_view')],
        n_az=hemi_n_az,
        n_el=hemi_n_el,
        elevation_min_deg=hemi_elevation_min_deg,
        elevation_max_deg=hemi_elevation_max_deg,
    )

    ny, nx = surface.patch_centers().shape[:2]
    n_samp = u_arr.size
    earth_view = np.zeros((n_samp, ny, nx), dtype=float)
    albedo_view = np.zeros_like(earth_view)
    solar_view = np.zeros_like(earth_view)
    eclipse = np.empty(n_samp, dtype=bool)
    sun_weight = np.array([math.pi], dtype=float)

    for k, uk in enumerate(u_arr):
        rotation = law(uk, orbit)
        eclipse[k] = orbit.in_eclipse(uk)
        samples = quad.sample(orbit.nadir_eci(uk), orbit.a)
        dirs_body = samples.dirs_eci @ rotation.m

        earth_view[k] = _scalar_integrate_surface_response(
            realized,
            surface,
            dirs_body,
            samples.weights,
        )
        albedo_view[k] = _scalar_integrate_surface_response(
            realized,
            surface,
            dirs_body,
            samples.weights,
            sample_weight=samples.surface_solar_cosine(sun_eci),
        )
        if not eclipse[k]:
            sun_body = rotation.T.apply(sun_eci)
            solar_view[k] = _scalar_integrate_surface_response(
                realized,
                surface,
                sun_body[None, :],
                sun_weight,
            )

    return SurfaceLoadingProfile(
        surface_name=surface_name,
        u=u_arr,
        earth_view=earth_view,
        albedo_view=albedo_view,
        solar_view=solar_view,
        solar_panel_view=np.repeat(static_views['solar_panel_view'][None, :, :], n_samp, axis=0),
        other_structure_view=np.repeat(static_views['other_structure_view'][None, :, :], n_samp, axis=0),
        space_view=np.repeat(static_views['space_view'][None, :, :], n_samp, axis=0),
        eclipse=eclipse,
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

    def test_mount_identity_for_matching_axis_pair(self):
        rotation = mount('+Y', '+Y')

        self.assertTrue(np.allclose(rotation.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12))
        self.assertTrue(np.allclose(rotation.apply([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12))
        self.assertTrue(np.allclose(rotation.apply([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-12))

    def test_mount_single_pair_uses_canonical_minimal_rotation(self):
        rotation = mount('+Y', '+Z')

        self.assertTrue(np.allclose(rotation.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12))
        self.assertTrue(np.allclose(rotation.apply([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12))
        self.assertTrue(np.allclose(rotation.apply([0.0, 0.0, 1.0]), [0.0, -1.0, 0.0], atol=1e-12))

    def test_mount_opposite_axis_is_deterministic(self):
        rotation = mount('+X', '-X')

        self.assertTrue(np.allclose(rotation.apply([1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0], atol=1e-12))
        self.assertTrue(np.allclose(rotation.apply([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12))
        self.assertTrue(np.allclose(rotation.apply([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0], atol=1e-12))

    def test_mount_two_pairs_aligns_full_frame(self):
        rotation = mount('+Y', '+Z', '+Z', '+X')

        self.assertTrue(np.allclose(rotation.apply([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12))
        self.assertTrue(np.allclose(rotation.apply([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-12))
        self.assertTrue(np.allclose(rotation.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12))

    def test_mount_rejects_invalid_secondary_axis_pair(self):
        with self.assertRaises(ValueError):
            mount('+Y', '+Z', '+Y', '+X')
        with self.assertRaises(ValueError):
            mount('+Y', '+Z', '+X')

    def test_realized_geometry_json_roundtrip_preserves_surfaces_and_metadata(self):
        cubesat = build_6u_double_deployable(bus_patch_shape=(2, 2))
        realized = cubesat.realize(
            {
                'wing_port_inner_angle': math.pi / 3.0,
                'wing_port_outer_angle': math.pi,
                'wing_starboard_inner_angle': -math.pi / 2.0,
                'wing_starboard_outer_angle': -math.pi,
            },
            mount_rotation=mount('+Y', '+Z', '+Z', '+X'),
            mount_offset=np.array([0.01, -0.02, 0.03]),
        )

        path = Path('output/test_realized_geometry_roundtrip.json')
        path.unlink(missing_ok=True)
        try:
            realized.to_json(path)
            loaded = RealizedGeometry.from_json(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(loaded.metadata, realized.metadata)
        self.assertEqual(loaded.names(), realized.names())
        for loaded_surface, realized_surface in zip(loaded.surfaces, realized.surfaces):
            self.assertEqual(loaded_surface.name, realized_surface.name)
            self.assertTrue(np.allclose(loaded_surface.center, realized_surface.center, atol=1e-12))
            self.assertTrue(np.allclose(loaded_surface.normal, realized_surface.normal, atol=1e-12))
            self.assertTrue(np.allclose(loaded_surface.u_axis, realized_surface.u_axis, atol=1e-12))
            self.assertEqual(loaded_surface.width, realized_surface.width)
            self.assertEqual(loaded_surface.height, realized_surface.height)
            self.assertEqual(loaded_surface.two_sided, realized_surface.two_sided)
            self.assertEqual(loaded_surface.patch_shape, realized_surface.patch_shape)
            self.assertEqual(loaded_surface.tags, realized_surface.tags)

    def test_surface_body_role_uses_realized_normals(self):
        realized = build_6u_double_deployable().realize(
            mount_rotation=mount('+Y', '+Z', '+Z', '+X'),
        )

        self.assertEqual(
            surface_body_role(surface_by_normal(realized, [0.0, 1.0, 0.0], tag='bus')),
            'body +Y bus face',
        )
        self.assertEqual(
            surface_body_role(surface_by_normal(realized, [1.0, 0.0, 0.0], tag='bus')),
            'body +X bus face',
        )
        self.assertEqual(
            surface_body_role(realized.by_name('wing_port_inner')),
            'body +Z solar panel',
        )

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

    def test_spacecraft_occlusion_mask_matches_scalar_reference(self):
        cubesat = build_6u_double_deployable(bus_patch_shape=(2, 2))
        realized = cubesat.realize()
        surface = realized.by_name('bus_+Y')
        dirs_body, _, _ = occlusion_impl._hemisphere_quadrature(
            surface,
            n_az=16,
            n_el=8,
            elevation_min_deg=5.0,
            elevation_max_deg=85.0,
        )

        visible = spacecraft_occlusion_mask(realized, surface, dirs_body)
        reference = _scalar_occlusion_mask(realized, surface, dirs_body)

        self.assertTrue(np.array_equal(visible, reference))

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

    def test_integrate_surface_response_matches_scalar_reference(self):
        cubesat = build_6u_double_deployable(bus_patch_shape=(2, 2))
        realized = cubesat.realize()
        surface = realized.by_name('bus_+Y')
        dirs_body, _, _ = occlusion_impl._hemisphere_quadrature(
            surface,
            n_az=14,
            n_el=7,
            elevation_min_deg=5.0,
            elevation_max_deg=85.0,
        )
        solid_angle_weights = np.linspace(0.1, 0.9, dirs_body.shape[0])
        sample_weight = np.linspace(0.2, 1.0, dirs_body.shape[0])

        value = integrate_surface_response(
            realized,
            surface,
            dirs_body,
            solid_angle_weights,
            sample_weight=sample_weight,
        )
        reference = _scalar_integrate_surface_response(
            realized,
            surface,
            dirs_body,
            solid_angle_weights,
            sample_weight=sample_weight,
        )

        self.assertTrue(np.allclose(value, reference, atol=1e-12))

    def test_hemisphere_group_view_uses_first_matching_tag_order(self):
        geometry = _simple_geometry(blocker_tags=('solar_panel', 'deployable'))
        realized = geometry.realize()

        grouped = hemisphere_group_view(
            realized,
            'source',
            [('deployable', 'deployable_view'), ('solar_panel', 'solar_panel_view')],
            n_az=120,
            n_el=60,
        )

        self.assertGreater(grouped['deployable_view'][0, 0], 0.0)
        self.assertAlmostEqual(grouped['solar_panel_view'][0, 0], 0.0, places=12)

    def test_hemisphere_group_view_closure_for_open_hemisphere(self):
        geometry = _simple_geometry(include_blocker=False)
        realized = geometry.realize()

        grouped = hemisphere_group_view(
            realized,
            'source',
            [('solar_panel', 'solar_panel_view')],
            n_az=160,
            n_el=80,
        )
        total = (
            grouped['solar_panel_view']
            + grouped['other_structure_view']
            + grouped['space_view']
        )

        self.assertTrue(np.allclose(total, [[1.0]], atol=2e-3))

    def test_hemisphere_group_view_matches_scalar_reference(self):
        cubesat = build_6u_double_deployable(bus_patch_shape=(2, 2))
        realized = cubesat.realize()

        grouped = hemisphere_group_view(
            realized,
            'bus_+Y',
            [('solar_panel', 'solar_panel_view')],
            n_az=20,
            n_el=10,
        )
        reference = _scalar_hemisphere_group_view(
            realized,
            'bus_+Y',
            [('solar_panel', 'solar_panel_view')],
            n_az=20,
            n_el=10,
        )

        for key in ('solar_panel_view', 'other_structure_view', 'space_view'):
            self.assertTrue(np.allclose(grouped[key], reference[key], atol=1e-12))

    def test_surface_loading_propagate_keeps_warm_views_static(self):
        orbit = _sample_orbit()
        realized = build_6u_double_deployable(bus_patch_shape=(2, 2)).realize()
        law = SlewModeSwitch(TargetTracking(*SGR_A), SunTracking(), slew_rate_deg_s=0.5)

        profile = surface_loading_propagate(
            realized,
            'bus_+Y',
            orbit,
            law,
            n=24,
            n_mu=6,
            n_az=16,
            hemi_n_az=72,
            hemi_n_el=36,
        )

        self.assertTrue(np.allclose(
            profile.solar_panel_view,
            np.repeat(profile.solar_panel_view[:1], profile.u.size, axis=0),
        ))
        self.assertGreater(np.ptp(profile.average_earth_view()), 1e-4)
        self.assertGreater(np.ptp(profile.average_solar_view()), 1e-4)

    def test_surface_loading_propagate_direct_solar_occlusion(self):
        orbit = _sample_orbit()
        blocked_geometry = _simple_geometry(blocker_tags=('solar_panel',)).realize()
        clear_geometry = _simple_geometry(include_blocker=False).realize()

        blocked = surface_loading_propagate(
            blocked_geometry,
            'source',
            orbit,
            SunTracking(),
            n=8,
            n_mu=4,
            n_az=12,
            hemi_n_az=48,
            hemi_n_el=24,
        )
        clear = surface_loading_propagate(
            clear_geometry,
            'source',
            orbit,
            SunTracking(),
            n=8,
            n_mu=4,
            n_az=12,
            hemi_n_az=48,
            hemi_n_el=24,
        )

        self.assertLess(blocked.solar_view.mean(), clear.solar_view.mean())
        self.assertGreater(blocked.solar_panel_view.mean(), 0.0)
        self.assertTrue(np.allclose(clear.solar_panel_view, 0.0, atol=1e-12))

    def test_surface_loading_propagate_matches_scalar_reference(self):
        orbit = _sample_orbit()
        realized = build_6u_double_deployable(bus_patch_shape=(2, 2)).realize()
        law = SlewModeSwitch(TargetTracking(*SGR_A), SunTracking(), slew_rate_deg_s=0.5)

        profile = surface_loading_propagate(
            realized,
            'bus_+Y',
            orbit,
            law,
            n=8,
            n_mu=4,
            n_az=12,
            hemi_n_az=16,
            hemi_n_el=8,
        )
        reference = _scalar_surface_loading_reference(
            realized,
            'bus_+Y',
            orbit,
            law,
            n=8,
            n_mu=4,
            n_az=12,
            hemi_n_az=16,
            hemi_n_el=8,
        )

        self.assertTrue(np.allclose(profile.earth_view, reference.earth_view, atol=1e-12))
        self.assertTrue(np.allclose(profile.albedo_view, reference.albedo_view, atol=1e-12))
        self.assertTrue(np.allclose(profile.solar_view, reference.solar_view, atol=1e-12))
        self.assertTrue(np.allclose(profile.solar_panel_view, reference.solar_panel_view, atol=1e-12))
        self.assertTrue(np.allclose(profile.other_structure_view, reference.other_structure_view, atol=1e-12))
        self.assertTrue(np.allclose(profile.space_view, reference.space_view, atol=1e-12))

    def test_surface_loading_propagate_loaded_json_matches_in_memory_realized(self):
        orbit = _sample_orbit()
        realized = build_6u_double_deployable(bus_patch_shape=(2, 2)).realize(
            mount_rotation=mount('+Y', '+Z', '+Z', '+X'),
        )
        law = SlewModeSwitch(TargetTracking(*SGR_A), SunTracking(), slew_rate_deg_s=0.5)

        path = Path('output/test_surface_loading_roundtrip.json')
        path.unlink(missing_ok=True)
        try:
            realized.to_json(path)
            loaded = RealizedGeometry.from_json(path)
        finally:
            path.unlink(missing_ok=True)

        profile = surface_loading_propagate(
            realized,
            'bus_+Y',
            orbit,
            law,
            n=8,
            n_mu=4,
            n_az=12,
            hemi_n_az=16,
            hemi_n_el=8,
        )
        loaded_profile = surface_loading_propagate(
            loaded,
            'bus_+Y',
            orbit,
            law,
            n=8,
            n_mu=4,
            n_az=12,
            hemi_n_az=16,
            hemi_n_el=8,
        )

        self.assertTrue(np.allclose(profile.earth_view, loaded_profile.earth_view, atol=1e-12))
        self.assertTrue(np.allclose(profile.albedo_view, loaded_profile.albedo_view, atol=1e-12))
        self.assertTrue(np.allclose(profile.solar_view, loaded_profile.solar_view, atol=1e-12))
        self.assertTrue(np.allclose(profile.solar_panel_view, loaded_profile.solar_panel_view, atol=1e-12))
        self.assertTrue(np.allclose(profile.other_structure_view, loaded_profile.other_structure_view, atol=1e-12))
        self.assertTrue(np.allclose(profile.space_view, loaded_profile.space_view, atol=1e-12))

    def test_radiative_background_matches_component_formulas(self):
        profile = SurfaceLoadingProfile(
            surface_name='source',
            u=np.array([0.0, 1.0]),
            earth_view=np.array([[[0.5]], [[0.25]]]),
            albedo_view=np.array([[[0.2]], [[0.1]]]),
            solar_view=np.array([[[0.3]], [[0.0]]]),
            solar_panel_view=np.array([[[0.1]], [[0.1]]]),
            other_structure_view=np.zeros((2, 1, 1)),
            space_view=np.ones((2, 1, 1)),
            eclipse=np.array([False, True]),
        )

        background = radiative_background(
            profile,
            solar_panel_temperature_K=300.0,
            solar_panel_emittance=0.8,
        )

        expected_panel = 0.8 * SIGMA_SB * 300.0 ** 4 * profile.solar_panel_view
        self.assertTrue(np.allclose(background.earth_ir, 240.0 * profile.earth_view))
        self.assertTrue(np.allclose(background.albedo, 0.30 * 1361.0 * profile.albedo_view))
        self.assertTrue(np.allclose(background.solar, 1361.0 * profile.solar_view))
        self.assertTrue(np.allclose(background.solar_panel_ir, expected_panel))
        self.assertTrue(np.allclose(
            background.total,
            background.earth_ir + background.albedo + background.solar + background.solar_panel_ir,
        ))

    def test_radiative_background_temperature_is_optional_only_for_zero_warm_view(self):
        cold_profile = SurfaceLoadingProfile(
            surface_name='source',
            u=np.array([0.0]),
            earth_view=np.zeros((1, 1, 1)),
            albedo_view=np.zeros((1, 1, 1)),
            solar_view=np.zeros((1, 1, 1)),
            solar_panel_view=np.zeros((1, 1, 1)),
            other_structure_view=np.zeros((1, 1, 1)),
            space_view=np.ones((1, 1, 1)),
            eclipse=np.array([False]),
        )
        warm_profile = SurfaceLoadingProfile(
            surface_name='source',
            u=np.array([0.0]),
            earth_view=np.zeros((1, 1, 1)),
            albedo_view=np.zeros((1, 1, 1)),
            solar_view=np.zeros((1, 1, 1)),
            solar_panel_view=np.ones((1, 1, 1)) * 0.1,
            other_structure_view=np.zeros((1, 1, 1)),
            space_view=np.ones((1, 1, 1)) * 0.9,
            eclipse=np.array([False]),
        )

        cold = radiative_background(cold_profile)
        self.assertTrue(np.allclose(cold.solar_panel_ir, 0.0, atol=1e-12))

        with self.assertRaises(ValueError):
            radiative_background(warm_profile)


if __name__ == '__main__':
    unittest.main()
