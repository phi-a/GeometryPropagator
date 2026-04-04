"""Plotly 3D spacecraft scene with animated orbit-environment vectors."""

import numpy as np
import plotly.graph_objects as go

from .surfaces import RealizedGeometry, rect_patch_grid


# ── Mesh construction ─────────────────────────────────────────────────────────

def _quad_arrays(surface):
    """Vertex coords and triangle indices for a patch-resolved surface."""
    nx, ny = surface.patch_shape or (1, 1)
    dx = surface.width / nx
    dy = surface.height / ny
    u_ax = surface.u_axis
    v_ax = surface.v_axis
    xx, yy = rect_patch_grid(surface.width, surface.height, nx, ny)

    n = ny * nx
    verts = np.empty((n * 4, 3))
    ti = np.empty(n * 2, dtype=int)
    tj = np.empty(n * 2, dtype=int)
    tk = np.empty(n * 2, dtype=int)

    vi, ci = 0, 0
    for j in range(ny):
        for i in range(nx):
            cx, cy = float(xx[j, i]), float(yy[j, i])
            for sv, su in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                verts[vi] = (
                    surface.center
                    + (cx + su * dx / 2) * u_ax
                    + (cy + sv * dy / 2) * v_ax
                )
                vi += 1
            b = vi - 4
            ti[ci], tj[ci], tk[ci] = b, b + 1, b + 2
            ti[ci + 1], tj[ci + 1], tk[ci + 1] = b, b + 2, b + 3
            ci += 2

    return verts, ti, tj, tk


def mesh(surface, *, data=None, color='#C0C0C0', opacity=0.85,
         cmin=None, cmax=None, colorscale='Plasma',
         showscale=False, cbar_title='', name=None):
    """Build a ``go.Mesh3d`` for one rectangular surface."""
    verts, ti, tj, tk = _quad_arrays(surface)
    kw = dict(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=ti, j=tj, k=tk,
        opacity=opacity,
        flatshading=True,
        name=name or surface.name,
        showlegend=False,
    )
    if data is not None:
        kw.update(
            intensity=np.repeat(np.asarray(data, dtype=float).ravel(), 2),
            intensitymode='cell',
            colorscale=colorscale,
            cmin=cmin, cmax=cmax,
            showscale=showscale,
            colorbar=dict(title=cbar_title, len=0.55) if showscale else None,
        )
    else:
        kw['color'] = color
    return go.Mesh3d(**kw)


def edges(surface, *, color='rgba(255,255,255,0.3)', width=1):
    """Build a ``go.Scatter3d`` for the patch grid lines of one surface."""
    nx, ny = surface.patch_shape or (1, 1)
    u_ax, v_ax = surface.u_axis, surface.v_axis
    hw, hh = 0.5 * surface.width, 0.5 * surface.height
    xs = np.linspace(-hw, hw, nx + 1)
    ys = np.linspace(-hh, hh, ny + 1)
    c = surface.center

    nan3 = np.array([np.nan, np.nan, np.nan])
    pts = []
    for xi in xs:
        pts += [c + xi * u_ax + ys[0] * v_ax,
                c + xi * u_ax + ys[-1] * v_ax, nan3]
    for yi in ys:
        pts += [c + xs[0] * u_ax + yi * v_ax,
                c + xs[-1] * u_ax + yi * v_ax, nan3]

    p = np.array(pts)
    return go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2],
        mode='lines', line=dict(color=color, width=width),
        showlegend=False, hoverinfo='skip',
    )


# ── Environment vectors ──────────────────────────────────────────────────────

def _arrow(direction, length, color, label, origin=None):
    """Shaft + cone head + text label for a directional arrow."""
    origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    n = np.linalg.norm(d)
    d = d / n if n > 1e-15 else np.array([0.0, 0.0, 1.0])
    tip = origin + d * length

    shaft = go.Scatter3d(
        x=[origin[0], tip[0]], y=[origin[1], tip[1]], z=[origin[2], tip[2]],
        mode='lines', line=dict(color=color, width=5),
        showlegend=False, hoverinfo='skip',
    )
    head = go.Cone(
        x=[tip[0]], y=[tip[1]], z=[tip[2]],
        u=[d[0]], v=[d[1]], w=[d[2]],
        sizemode='absolute', sizeref=length * 0.07,
        colorscale=[[0, color], [1, color]],
        showscale=False, showlegend=False, hoverinfo='skip',
    )
    off = tip + d * length * 0.12
    txt = go.Scatter3d(
        x=[off[0]], y=[off[1]], z=[off[2]],
        mode='text', text=[label],
        textfont=dict(color=color, size=13),
        showlegend=False, hoverinfo='skip',
    )
    return [shaft, head, txt]


def _body_axes(scale):
    """Body-frame axis indicators (+X red, +Y green, +Z blue)."""
    traces = []
    for lbl, vec, col in [
        ('+X', [1, 0, 0], '#e74c3c'),
        ('+Y', [0, 1, 0], '#2ecc71'),
        ('+Z', [0, 0, 1], '#3498db'),
    ]:
        traces.extend(_arrow(vec, scale, col, lbl))
    return traces


# ── Orbit helpers ─────────────────────────────────────────────────────────────

def orbit_vectors(orbit, law, u):
    """Sun and Earth directions in body frame at each orbit sample.

    Returns (sun_body, earth_body), each shaped (n, 3).
    """
    u = np.asarray(u, dtype=float)
    sun_eci = orbit.sun_eci()
    n = u.size
    sun_body = np.zeros((n, 3))
    earth_body = np.zeros((n, 3))
    for k in range(n):
        R = law(u[k], orbit)
        sun_body[k] = R.m.T @ sun_eci
        earth_body[k] = R.m.T @ orbit.nadir_eci(u[k])
    return sun_body, earth_body


# ── Layout ────────────────────────────────────────────────────────────────────

_BG = '#080820'


def _layout(title=''):
    return go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            bgcolor=_BG,
            camera=dict(
                eye=dict(x=1.4, y=-1.6, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor=_BG,
        font=dict(color='white', family='monospace'),
        margin=dict(l=0, r=0, t=50, b=10),
        title=dict(text=title, font=dict(size=14)),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def _tag_color(surface):
    """Default flat color for a surface by tag."""
    if 'solar_panel' in surface.tags:
        return '#5B8FF9'
    if 'bus' in surface.tags:
        return '#888888'
    return '#666666'


def _extent(realized):
    """Max span of the spacecraft for arrow scaling."""
    corners = np.concatenate([s.corners() for s in realized.surfaces], axis=0)
    return float(np.ptp(corners, axis=0).max())


def scene(realized, *, data=None, label='', colorscale='Plasma',
          sun=None, earth=None, title='Spacecraft geometry'):
    """Static 3D overview of the realized spacecraft geometry.

    Parameters
    ----------
    realized : RealizedGeometry
    data : dict[str, ndarray [ny, nx]], optional
        Per-patch scalar overlay for named surfaces (e.g. view factors,
        temperatures).  Surfaces not in ``data`` render with their default
        flat colour.
    label : str
        Colorbar title when ``data`` is supplied.
    colorscale : str
        Plotly colorscale name (default ``'Plasma'``).
    sun, earth : array-like (3,), optional
        Body-frame direction vectors drawn as labelled arrows.
    title : str
    """
    data = data or {}
    all_vals = np.concatenate([np.asarray(v).ravel() for v in data.values()]) if data else None
    cmin = float(all_vals.min()) if all_vals is not None else None
    cmax = float(all_vals.max()) if all_vals is not None else None

    traces = []
    first_data_surf = True
    for s in realized.surfaces:
        if 'solar_panel_back' in s.tags:
            continue
        if s.name in data:
            traces.append(mesh(
                s, data=data[s.name],
                cmin=cmin, cmax=cmax, colorscale=colorscale,
                showscale=first_data_surf, cbar_title=label, opacity=0.92,
            ))
            first_data_surf = False
        else:
            traces.append(mesh(s, color=_tag_color(s), opacity=0.8))
        if s.patch_shape is not None:
            traces.append(edges(s))

    ext = _extent(realized)
    traces.extend(_body_axes(ext * 0.45))
    if sun is not None:
        traces.extend(_arrow(sun, ext * 0.7, '#f1c40f', 'Sun'))
    if earth is not None:
        traces.extend(_arrow(earth, ext * 0.7, '#00bcd4', 'Earth'))

    return go.Figure(data=traces, layout=_layout(title))


def animate(realized, data, *, sun, earth, u, eclipse,
            label='', colorscale='Plasma', title=''):
    """Animated 3D scene with per-patch data evolution and environment vectors.

    Parameters
    ----------
    realized : RealizedGeometry
    data : dict[str, ndarray]
        {surface_name: array [n_time, ny, nx]} for each animated surface.
    sun, earth : ndarray [n_time, 3]
        Body-frame directions at each orbit sample.
    u : ndarray [n_time]
        Argument of latitude.
    eclipse : ndarray [n_time] bool
    label : str
        Colorbar title.
    colorscale : str
        Plotly colorscale name.
    """
    u = np.asarray(u, dtype=float)
    eclipse = np.asarray(eclipse, dtype=bool)
    n_time = u.size

    # global colour range across all surfaces and time
    all_vals = np.concatenate([v.ravel() for v in data.values()])
    cmin, cmax = float(all_vals.min()), float(all_vals.max())

    ext = _extent(realized)
    arrow_len = ext * 0.7

    # ── fixed traces (static geometry + grid edges + body axes) ───────────
    fixed = []
    for s in realized.surfaces:
        if 'solar_panel_back' in s.tags:
            continue
        if s.name in data:
            continue
        fixed.append(mesh(s, color=_tag_color(s), opacity=0.55))

    for s in realized.surfaces:
        if 'solar_panel_back' in s.tags:
            continue
        if s.patch_shape is not None:
            fixed.append(edges(s, color='rgba(255,255,255,0.2)', width=1))

    fixed.extend(_body_axes(ext * 0.45))
    n_fixed = len(fixed)

    # ── animated traces (data-mapped surfaces + arrows, frame 0) ──────────
    ordered = [realized.by_name(nm) for nm in data]
    anim0 = []
    for idx, s in enumerate(ordered):
        anim0.append(mesh(
            s, data=data[s.name][0],
            cmin=cmin, cmax=cmax, colorscale=colorscale,
            showscale=(idx == 0), cbar_title=label, opacity=0.92,
        ))
    anim0.extend(_arrow(sun[0], arrow_len, '#f1c40f', 'Sun'))
    anim0.extend(_arrow(earth[0], arrow_len, '#00bcd4', 'Earth'))
    n_anim = len(anim0)
    anim_idx = list(range(n_fixed, n_fixed + n_anim))

    # ── animation frames ──────────────────────────────────────────────────
    frames = []
    for k in range(n_time):
        fd = []
        for idx, s in enumerate(ordered):
            fd.append(mesh(
                s, data=data[s.name][k],
                cmin=cmin, cmax=cmax, colorscale=colorscale,
                showscale=(idx == 0), cbar_title=label, opacity=0.92,
            ))
        fd.extend(_arrow(sun[k], arrow_len, '#f1c40f', 'Sun'))
        fd.extend(_arrow(earth[k], arrow_len, '#00bcd4', 'Earth'))
        frames.append(go.Frame(data=fd, traces=anim_idx, name=str(k)))

    # ── assemble ──────────────────────────────────────────────────────────
    fig = go.Figure(data=fixed + anim0, frames=frames, layout=_layout(title))

    steps = [
        dict(args=[[str(k)], dict(mode='immediate',
                                  frame=dict(duration=0, redraw=True))],
             label=f'{np.degrees(u[k]):.0f}', method='animate')
        for k in range(n_time)
    ]
    fig.update_layout(
        sliders=[dict(
            active=0, steps=steps,
            currentvalue=dict(prefix='u = ', suffix='\u00b0'),
            x=0.08, len=0.84, font=dict(size=10),
        )],
        updatemenus=[dict(
            type='buttons', showactive=False, x=0.02, y=0.02,
            buttons=[
                dict(label='\u25b6', method='animate',
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                      fromcurrent=True, mode='immediate')]),
                dict(label='\u23f8', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=True),
                                        mode='immediate')]),
            ],
        )],
    )

    return fig
