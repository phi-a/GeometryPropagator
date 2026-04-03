"""SV-7 Systems Measures Matrix for the LEO X-ray payload."""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from constants import DEFAULT_FOV_HALF_DEG, DEFAULT_N_MAX, J2, MU, R_E
from optimize import Instrument
from orbit import raan_for_ltan
from schedule import schedule_range
from target import GALACTIC_CENTER

__all__: list[str] = []


# Orbit definitions
ALT_KM = 500
A = R_E + ALT_KM * 1e3
E = 0.0
EPOCH = datetime(2027, 3, 21, tzinfo=timezone.utc)
FOV_HALF = math.radians(DEFAULT_FOV_HALF_DEG)
SEASON_DAYS = 185
RAAN_STEP_DEG = 15

# ISS-like
I_ISS = math.radians(51.6)

# SSO: solve for inclination that gives dOmega/dt = +0.9856 deg/day
_n = math.sqrt(MU / A**3)
_target_rate = math.radians(0.9856474) / 86400.0
_cos_i_sso = _target_rate / (-1.5 * J2 * _n * (R_E / A) ** 2)
I_SSO = math.acos(_cos_i_sso)

LTAN_SSO_HOURS = 12.0

OMEGA_SSO_NOON = raan_for_ltan(EPOCH, LTAN_SSO_HOURS)


def _clip_equinoxes(results: list[dict]) -> list[dict]:
    year = results[0]["date"].year
    v_eq = datetime(year, 3, 20, tzinfo=timezone.utc)
    a_eq = datetime(year, 9, 22, tzinfo=timezone.utc)
    return [r for r in results if v_eq <= r["date"] <= a_eq]


@dataclass
class RunStats:
    """Aggregated performance measures for one schedule run."""

    label: str
    raan0_deg: float
    n_viable: int
    n_total: int
    eclipse_min: float
    eclipse_max: float
    eclipse_mean: float
    open_sky_min: float
    open_sky_max: float
    open_sky_mean: float
    n_lo: int
    n_hi: int
    n_mode: int
    b_min_s: float
    b_max_s: float
    b_mean_s: float
    obj_min_s: float
    obj_max_s: float
    obj_mean_s: float
    obj_cum_s: float


def _compute_stats(
    season: list[dict], label: str, raan0_deg: float
) -> RunStats | None:
    viable = [r for r in season if r["solution"]]
    if not viable:
        return None

    ecl = [r["eclipse_s"] for r in viable]
    osk = [r["open_sky_s"] for r in viable]
    ns = [r["solution"].n_samples for r in viable]
    bs = [r["solution"].exposure_s for r in viable]
    objs = [r["solution"].objective for r in viable]

    from statistics import mode

    return RunStats(
        label=label,
        raan0_deg=raan0_deg,
        n_viable=len(viable),
        n_total=len(season),
        eclipse_min=min(ecl),
        eclipse_max=max(ecl),
        eclipse_mean=sum(ecl) / len(ecl),
        open_sky_min=min(osk),
        open_sky_max=max(osk),
        open_sky_mean=sum(osk) / len(osk),
        n_lo=min(ns),
        n_hi=max(ns),
        n_mode=mode(ns),
        b_min_s=min(bs),
        b_max_s=max(bs),
        b_mean_s=sum(bs) / len(bs),
        obj_min_s=min(objs),
        obj_max_s=max(objs),
        obj_mean_s=sum(objs) / len(objs),
        obj_cum_s=sum(objs),
    )


def sweep_iss(inst: Instrument, raan_step_deg: int = RAAN_STEP_DEG) -> list[RunStats]:
    """Run the ISS-like orbit across RAAN0 and collect envelope stats."""
    stats: list[RunStats] = []
    for omega_deg in range(0, 360, raan_step_deg):
        omega_rad = math.radians(omega_deg)
        results = schedule_range(
            EPOCH,
            SEASON_DAYS,
            A,
            E,
            I_ISS,
            omega_rad,
            EPOCH,
            inst,
            n_max=DEFAULT_N_MAX,
            step_hours=24,
            target=GALACTIC_CENTER,
            fov_half=FOV_HALF,
        )
        season = _clip_equinoxes(results)
        stats_run = _compute_stats(season, f"ISS RAAN0={omega_deg} deg", omega_deg)
        if stats_run:
            stats.append(stats_run)
    return stats


def run_sso_noon(inst: Instrument) -> RunStats | None:
    """Run the SSO LTAN-noon orbit."""
    results = schedule_range(
        EPOCH,
        SEASON_DAYS,
        A,
        E,
        I_SSO,
        OMEGA_SSO_NOON,
        EPOCH,
        inst,
        n_max=DEFAULT_N_MAX,
        step_hours=24,
        target=GALACTIC_CENTER,
        fov_half=FOV_HALF,
    )
    season = _clip_equinoxes(results)
    return _compute_stats(season, "SSO noon", math.degrees(OMEGA_SSO_NOON))


def _envelope(stats: list[RunStats]) -> dict:
    """Worst/best values across the full ISS-like RAAN sweep."""
    return {
        "viable_min": min(s.n_viable for s in stats),
        "viable_max": max(s.n_viable for s in stats),
        "viable_pct_min": min(100 * s.n_viable / s.n_total for s in stats),
        "viable_pct_max": max(100 * s.n_viable / s.n_total for s in stats),
        "osk_min": min(s.open_sky_min for s in stats),
        "osk_max": max(s.open_sky_max for s in stats),
        "osk_mean_min": min(s.open_sky_mean for s in stats),
        "osk_mean_max": max(s.open_sky_mean for s in stats),
        "n_lo": min(s.n_lo for s in stats),
        "n_hi": max(s.n_hi for s in stats),
        "n_mode_range": (
            min(s.n_mode for s in stats),
            max(s.n_mode for s in stats),
        ),
        "b_min": min(s.b_min_s for s in stats),
        "b_max": max(s.b_max_s for s in stats),
        "b_mean_min": min(s.b_mean_s for s in stats),
        "b_mean_max": max(s.b_mean_s for s in stats),
        "obj_min": min(s.obj_min_s for s in stats),
        "obj_max": max(s.obj_max_s for s in stats),
        "obj_cum_min": min(s.obj_cum_s for s in stats),
        "obj_cum_max": max(s.obj_cum_s for s in stats),
    }


# Threshold = minimum acceptable. Objective = desired target.
THRESH = {
    "viable_pct": 30,
    "open_sky_s": 200,
    "n_samples": 1,
    "exposure_s": 300,
    "obj_pass_s": 300,
    "obj_cum_s": None,
}

OBJ = {
    "viable_pct": 50,
    "open_sky_s": 1000,
    "n_samples": 8,
    "exposure_s": 600,
    "obj_pass_s": 5000,
    "obj_cum_s": None,
}


def _thresh(key: str) -> str:
    value = THRESH[key]
    return "TBD [OV-3]" if value is None else f">= {value}"


def _obj(key: str) -> str:
    value = OBJ[key]
    return "TBD [OV-3]" if value is None else f">= {value}"


SV7_HEADER = [
    "Function",
    "Measure",
    "Units",
    "Threshold",
    "Objective",
    "ISS-like",
    "SSO-noon",
]


def _fmt_range(lo: float, hi: float, fmt: str = ".0f") -> str:
    if abs(lo - hi) < 0.5:
        return f"{lo:{fmt}}"
    return f"{lo:{fmt}}-{hi:{fmt}}"


def build_sv7_rows(env: dict, sso: RunStats, inst: Instrument) -> list[list[str]]:
    """Build the SV-7 cell data."""
    readout_s = inst.readout_per_sample
    return [
        [
            "Seasonal coverage",
            "Viable passes",
            "% of season",
            _thresh("viable_pct"),
            _obj("viable_pct"),
            _fmt_range(env["viable_pct_min"], env["viable_pct_max"], ".0f"),
            f"{100 * sso.n_viable / sso.n_total:.0f}",
        ],
        [
            "Eclipse budget",
            "Open-sky budget T_open",
            "s",
            _thresh("open_sky_s"),
            _obj("open_sky_s"),
            _fmt_range(env["osk_min"], env["osk_max"], ".0f"),
            _fmt_range(sso.open_sky_min, sso.open_sky_max, ".0f"),
        ],
        [
            "Payload startup",
            "Boot time t_su",
            "s",
            f"<= {inst.startup_s:.0f}",
            f"<= {inst.startup_s:.0f}",
            f"{inst.startup_s:.0f}",
            f"{inst.startup_s:.0f}",
        ],
        [
            "CCD readout",
            "Readout/sample C",
            "s",
            "fixed",
            "fixed",
            f"{readout_s:.1f}",
            f"{readout_s:.1f}",
        ],
        [
            "CCD readout",
            "Sample count N",
            "samples",
            _thresh("n_samples"),
            _obj("n_samples"),
            _fmt_range(env["n_lo"], env["n_hi"], ".0f"),
            _fmt_range(sso.n_lo, sso.n_hi, ".0f"),
        ],
        [
            "Science exposure",
            "Exposure time B",
            "s",
            _thresh("exposure_s"),
            _obj("exposure_s"),
            _fmt_range(env["b_min"], env["b_max"], ".0f"),
            _fmt_range(sso.b_min_s, sso.b_max_s, ".0f"),
        ],
        [
            "SNR objective",
            "B x N single-pass",
            "s",
            _thresh("obj_pass_s"),
            _obj("obj_pass_s"),
            _fmt_range(env["obj_min"], env["obj_max"], ".0f"),
            _fmt_range(sso.obj_min_s, sso.obj_max_s, ".0f"),
        ],
        [
            "SNR objective",
            "B x N cumulative",
            "s",
            _thresh("obj_cum_s"),
            _obj("obj_cum_s"),
            _fmt_range(env["obj_cum_min"], env["obj_cum_max"], ".0f"),
            f"{sso.obj_cum_s:.0f}",
        ],
    ]


def render_sv7_table(rows: list[list[str]], inst: Instrument) -> plt.Figure:
    """Render the SV-7 matrix as a tight minimalist table figure."""
    del inst  # Table rendering only depends on the already-formatted rows.

    fig, ax = plt.subplots(figsize=(14.5, 4.6))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    col_widths = [0.165, 0.21, 0.085, 0.1125, 0.1125, 0.1575, 0.1575]
    tbl = ax.table(
        cellText=rows,
        colLabels=SV7_HEADER,
        colWidths=col_widths,
        bbox=[0.005, 0.12, 0.99, 0.79],
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)

    header_bg = "#253140"
    header_iss = "#33465f"
    header_sso = "#3d5d5a"
    grid = "#d8dee6"
    row_white = "#ffffff"
    row_gray = "#f7f8fa"
    iss_white = "#eef3f9"
    iss_gray = "#e8eef6"
    sso_white = "#edf5f2"
    sso_gray = "#e7f0ec"
    text_dark = "#15202b"
    text_muted = "#54606c"

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(grid)
        cell.set_linewidth(0.8)
        cell.PAD = 0.012
        cell.get_text().set_wrap(False)

        if row == 0:
            if col == 5:
                cell.set_facecolor(header_iss)
            elif col == 6:
                cell.set_facecolor(header_sso)
            else:
                cell.set_facecolor(header_bg)
            cell.set_text_props(color="white", fontweight="bold", fontsize=14)
            cell.set_height(0.092)
            continue

        is_even = row % 2 == 0
        if col == 5:
            cell.set_facecolor(iss_gray if is_even else iss_white)
        elif col == 6:
            cell.set_facecolor(sso_gray if is_even else sso_white)
        else:
            cell.set_facecolor(row_gray if is_even else row_white)

        cell.set_height(0.068)
        cell.set_text_props(color=text_dark, fontsize=14)

        if col in (3, 4):
            cell.set_text_props(fontsize=14, color=text_muted)
        if col in (5, 6):
            cell.set_text_props(fontweight="bold", fontsize=14, color=text_dark)
        if col == 0:
            cell.set_text_props(fontweight="semibold", fontsize=14, color=text_dark)

    fig.suptitle(
        f"SV-7 Systems Measures Matrix | LEO X-ray Payload | {ALT_KM} km | Equinox {SEASON_DAYS} d",
        fontsize=16,
        fontweight="bold",
        color=text_dark,
        y=0.915,
    )
    fig.text(
        0.5,
        0.055,
        f"TBD [OV-3] values require upstream traceability | ISS-like envelope uses {RAAN_STEP_DEG} deg RAAN steps",
        ha="center",
        fontsize=10,
        color=text_muted,
    )

    fig.subplots_adjust(left=0.015, right=0.985, top=0.93, bottom=0.055)

    out = Path("output/sv7_matrix.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    return fig


def print_sv7(rows: list[list[str]]) -> None:
    """Print the SV-7 table to console."""
    widths = [len(h) for h in SV7_HEADER]
    for row in rows:
        for col, cell in enumerate(row):
            widths[col] = max(widths[col], len(cell))

    def _print_row(cells: list[str]) -> None:
        parts = [cell.ljust(widths[idx]) for idx, cell in enumerate(cells)]
        print("  | ".join(parts))

    _print_row(SV7_HEADER)
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        _print_row(row)
    print()


def main() -> None:
    inst = Instrument()

    print(f"Altitude:  {ALT_KM} km")
    print(f"ISS inc:   {math.degrees(I_ISS):.1f} deg")
    print(f"SSO inc:   {math.degrees(I_SSO):.1f} deg")
    print(f"SSO RAAN0 @ epoch (LTAN {LTAN_SSO_HOURS:.1f}): {math.degrees(OMEGA_SSO_NOON):.3f} deg")
    print(f"Season:    {SEASON_DAYS} days from {EPOCH:%Y-%m-%d}")
    print(f"Readout C: {inst.readout_per_sample:.1f} s/sample")
    print()

    print(f"Sweeping ISS-like RAAN 0-360 deg ({RAAN_STEP_DEG} deg steps)...")
    iss_stats = sweep_iss(inst, raan_step_deg=RAAN_STEP_DEG)
    env = _envelope(iss_stats)
    print(
        f"  {len(iss_stats)} runs, viable range: "
        f"{env['viable_pct_min']:.0f}-{env['viable_pct_max']:.0f}%"
    )

    print("Running SSO LTAN noon...")
    sso = run_sso_noon(inst)
    if sso is None:
        print("  No viable passes for SSO noon.")
        return
    print(f"  {sso.n_viable} viable passes ({100 * sso.n_viable / sso.n_total:.0f}%)")

    print()
    rows = build_sv7_rows(env, sso, inst)
    print_sv7(rows)
    render_sv7_table(rows, inst)
    plt.show()


if __name__ == "__main__":
    main()
