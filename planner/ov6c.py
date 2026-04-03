"""OV-6c: Operational Event-Trace — Single Eclipse Pass.

Compact 1-D horizontal timeline.  Designed to be as vertically
tight as possible for paper/slide embedding.
"""

import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from constants import R_E, DEFAULT_N_MAX, DEFAULT_FOV_HALF_DEG
from optimize import Instrument
from schedule import schedule_range
from target import GALACTIC_CENTER

__all__: list[str] = []

PAL = dict(
    bg      = "#ffffff",
    bar_bg  = "#e8e8e8",
    startup = "#6b8e23",
    expose  = "#3b7fc4",
    readout = "#cc5533",
    text    = "#222222",
    sub     = "#666666",
    topen   = "#2ca02c",
    arrow   = "#c0392b",
)


def plot_ov6c(
    eclipse_s: float,
    open_sky_s: float,
    t_su: float,
    B: float,
    N: int,
    C: float,
    date: datetime,
    raan_deg: float,
    filename: str = "output/ov6c_sequence.png",
) -> plt.Figure:
    """Render OV-6c as a compact 1-D timeline."""
    t_blocked = (eclipse_s - open_sky_s) / 2.0
    t0_open = t_blocked
    t1_open = eclipse_s - t_blocked
    t0_boot = t0_open
    t1_boot = t0_boot + t_su
    t0_exp  = t1_boot
    t1_exp  = t0_exp + B
    readout_segs = [(t1_exp + k * C, t1_exp + (k + 1) * C)
                    for k in range(N)]
    ro_start = readout_segs[0][0]
    ro_end   = readout_segs[-1][1]

    # ── Figure — as compact as possible ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 2.8))
    fig.patch.set_facecolor(PAL["bg"])
    ax.set_facecolor(PAL["bg"])

    arrow_pad = eclipse_s * 0.08
    ax.set_xlim(-arrow_pad, eclipse_s + arrow_pad)
    ax.set_ylim(-1.10, 0.78)
    ax.axis("off")

    bar_y = 0.0
    bar_h = 0.40

    # ── Entry / exit arrows (black, labels beside arrow) ──────────────
    arr_y = bar_y + bar_h / 2
    arr_len = arrow_pad * 0.55

    # Entry arrow
    ax.annotate("", xy=(0, arr_y), xytext=(-arr_len, arr_y),
                arrowprops=dict(arrowstyle="-|>", color=PAL["text"],
                                lw=1.8, mutation_scale=10),
                zorder=6)
    ax.text(-arr_len - eclipse_s * 0.005, arr_y, "Umbra Entry",
            ha="right", va="center",
            fontsize=7.5, color=PAL["text"], fontweight="bold")

    # Exit arrow
    ax.annotate("", xy=(eclipse_s + arr_len, arr_y),
                xytext=(eclipse_s, arr_y),
                arrowprops=dict(arrowstyle="-|>", color=PAL["text"],
                                lw=1.8, mutation_scale=10),
                zorder=6)
    ax.text(eclipse_s + arr_len + eclipse_s * 0.005, arr_y,
            "Umbra Exit",
            ha="left", va="center",
            fontsize=7.5, color=PAL["text"], fontweight="bold")

    # ── Background bar (full eclipse) ─────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, bar_y), eclipse_s, bar_h,
        boxstyle="round,pad=0.006",
        facecolor=PAL["bar_bg"], edgecolor="#cccccc",
        linewidth=0.5, zorder=1,
    ))

    # ── Phase bars ────────────────────────────────────────────────────
    def _bar(t0, t1, color, alpha=0.92):
        ax.add_patch(mpatches.FancyBboxPatch(
            (t0, bar_y), t1 - t0, bar_h,
            boxstyle="round,pad=0.003",
            facecolor=color, edgecolor="white",
            linewidth=0.4, alpha=alpha, zorder=3,
        ))

    # Boot
    _bar(t0_boot, t1_boot, PAL["startup"])
    ax.text((t0_boot + t1_boot) / 2, bar_y + bar_h / 2,
            f"Boot\n{t_su:.0f}s",
            ha="center", va="center",
            fontsize=7, color="white", fontweight="bold", zorder=4,
            linespacing=1.0)

    # Exposure
    _bar(t0_exp, t1_exp, PAL["expose"])
    ax.text((t0_exp + t1_exp) / 2, bar_y + bar_h / 2,
            f"Exposure   B = {B/60:.1f} min",
            ha="center", va="center",
            fontsize=10, color="white", fontweight="bold", zorder=4)

    # Readout segments
    for ra, rb in readout_segs:
        _bar(ra, rb, PAL["readout"], alpha=0.88)

    # Readout label across full block
    ax.text((ro_start + ro_end) / 2, bar_y + bar_h / 2,
            f"Readout   N = {N}",
            ha="center", va="center",
            fontsize=10, color="white", fontweight="bold", zorder=4)

    # ── T_open brace (dimension line above main bar) ───────────────────
    topen_y = bar_y + bar_h + 0.03
    cap_h = 0.06
    ax.plot([t0_open, t0_open], [topen_y, topen_y + cap_h],
            color=PAL["topen"], lw=0.7)
    ax.plot([t1_open, t1_open], [topen_y, topen_y + cap_h],
            color=PAL["topen"], lw=0.7)
    ax.plot([t0_open, t1_open], [topen_y + cap_h, topen_y + cap_h],
            color=PAL["topen"], lw=0.7)
    ax.text((t0_open + t1_open) / 2, topen_y + cap_h + 0.02,
            f"T_open = {open_sky_s/60:.1f} min",
            ha="center", va="bottom",
            fontsize=7, color=PAL["topen"], fontweight="bold")

    # ── Time ticks (tight, just below bar) ────────────────────────────
    tick_y = bar_y - 0.03
    tick_step = 300
    for t in range(0, int(eclipse_s) + 1, tick_step):
        ax.plot([t, t], [tick_y, tick_y - 0.04],
                color=PAL["sub"], lw=0.4)
        ax.text(t, tick_y - 0.06, f"{int(t // 60)}m",
                ha="center", va="top",
                fontsize=20, color=PAL["sub"])

    # ── Braces (compact, single row below ticks) ──────────────────────
    br_y = tick_y - 0.26

    def _brace(t0, t1, label, color):
        ax.plot([t0, t0], [br_y, br_y - 0.06], color=color, lw=0.7)
        ax.plot([t1, t1], [br_y, br_y - 0.06], color=color, lw=0.7)
        ax.plot([t0, t1], [br_y - 0.06, br_y - 0.06], color=color, lw=0.7)
        ax.text((t0 + t1) / 2, br_y - 0.09, label,
                ha="center", va="top",
                fontsize=7, color=color)

    _brace(t0_boot, t1_boot, f"t_su = {t_su:.0f} s", PAL["startup"])
    _brace(t0_exp, t1_exp, f"B = {B:.0f} s", PAL["expose"])
    _brace(ro_start, ro_end,
           f"N\u00d7C = {N}\u00d7{C:.0f} = {N * C:.0f} s", PAL["readout"])

    # ── Budget equation (right-aligned below braces) ────────────────
    eq = (f"$t_{{su}} + B + N\\!\\cdot\\!C"
          f" = {t_su:.0f} + {B:.0f} + {N}\\times{C:.0f}"
          f" = {t_su + B + N * C:.0f}\\;\\mathrm{{s}}"
          f"\\;\\leq\\; T_{{open}}$")
    ax.text(eclipse_s, br_y - 0.22, eq,
            ha="right", va="top",
            fontsize=7, color=PAL["sub"])

    # ── Title (compact, above T_open bar) ────────────────────────────
    ax.text(eclipse_s / 2, topen_y + cap_h + 0.30,
            "OV-6c  Operational Event-Trace \u2014 Nominal Eclipse Pass",
            ha="center", va="top",
            fontsize=10, fontweight="bold", color=PAL["text"])
    ax.text(eclipse_s / 2, topen_y + cap_h + 0.20,
            f"ISS-like  |  500 km, 51.6\u00b0  |  {date:%d %b %Y}"
            f"  |  B\u00d7N = {B * N:.0f} s",
            ha="center", va="top",
            fontsize=8, color=PAL["sub"])

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    out = Path(filename)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=PAL["bg"])
    print(f"Saved: {out}")
    return fig


def main() -> None:
    A = R_E + 500e3
    I = math.radians(51.6)
    OMEGA0 = math.radians(45)
    EPOCH = datetime(2027, 3, 21, tzinfo=timezone.utc)
    FOV_HALF = math.radians(DEFAULT_FOV_HALF_DEG)
    inst = Instrument()

    results = schedule_range(
        EPOCH, 185, A, 0.0, I, OMEGA0, EPOCH, inst,
        n_max=DEFAULT_N_MAX, step_hours=24,
        target=GALACTIC_CENTER, fov_half=FOV_HALF,
    )
    viable = [r for r in results if r["solution"]]
    best = max(viable, key=lambda r: r["solution"].objective)
    sol = best["solution"]
    print(f"  {best['date']:%Y-%m-%d}  B={sol.exposure_s:.0f}s  N={sol.n_samples}")

    plot_ov6c(
        eclipse_s=best["eclipse_s"], open_sky_s=best["open_sky_s"],
        t_su=inst.startup_s, B=sol.exposure_s, N=sol.n_samples,
        C=inst.readout_per_sample, date=best["date"],
        raan_deg=best["raan_deg"],
    )
    plt.show()


if __name__ == "__main__":
    main()
