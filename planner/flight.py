"""Flight scheduler: TLE + timestamp -> optimal (N, exposure).

Vanilla Python only (stdlib + local modules). No numpy/scipy.
Designed for easy C++ translation.
"""

import argparse
import math
from datetime import datetime, timedelta, timezone

from constants import (MU, R_E, SECONDS_PER_DAY, TLE_LINE_LEN, TLE_Y2K_PIVOT,
                       DEFAULT_N_MAX, DEFAULT_FOV_HALF_DEG, DEFAULT_STARTUP_S,
                       DEFAULT_N_PIXELS, DEFAULT_READ_RATE,
                       SGR_A_RA_DEG, SGR_A_DEC_DEG)
from orbit import eclipse_duration, j2_raan_rate, propagate_raan, sun_beta_uc
from sun import sun_dist
from optimize import Instrument, optimize
from target import GALACTIC_CENTER, Target, open_sky_budget, target_beta_uc


# ── TLE parsing ──────────────────────────────────────────────────

def _tle_checksum(line: str) -> int:
    """Compute TLE line checksum (mod-10 digit sum, '-' counts as 1)."""
    s = 0
    for c in line[:68]:
        if c.isdigit():
            s += int(c)
        elif c == "-":
            s += 1
    return s % 10


def parse_tle(line1: str, line2: str) -> dict:
    """Parse a NORAD two-line element set.

    Parameters
    ----------
    line1 : str
        TLE line 1 (>=69 characters).
    line2 : str
        TLE line 2 (>=69 characters).

    Returns
    -------
    dict with keys:
        epoch : datetime (UTC)
        a     : float (m)   semi-major axis
        e     : float        eccentricity
        i     : float (rad)  inclination
        raan  : float (rad)  right ascension of ascending node
        argp  : float (rad)  argument of perigee
        ma    : float (rad)  mean anomaly
        n_rev_day : float    mean motion (rev/day)

    Raises
    ------
    ValueError
        If the TLE is malformed (wrong prefix, length, or checksum).
    """
    line1 = line1.rstrip()
    line2 = line2.rstrip()

    if len(line1) < TLE_LINE_LEN:
        raise ValueError(f"TLE line 1 too short ({len(line1)} < {TLE_LINE_LEN})")
    if len(line2) < TLE_LINE_LEN:
        raise ValueError(f"TLE line 2 too short ({len(line2)} < {TLE_LINE_LEN})")
    if line1[0] != "1":
        raise ValueError(f"TLE line 1 must start with '1', got '{line1[0]}'")
    if line2[0] != "2":
        raise ValueError(f"TLE line 2 must start with '2', got '{line2[0]}'")

    # Checksum validation
    ck1 = int(line1[68])
    if _tle_checksum(line1) != ck1:
        raise ValueError(f"TLE line 1 checksum mismatch "
                         f"(computed {_tle_checksum(line1)}, expected {ck1})")
    ck2 = int(line2[68])
    if _tle_checksum(line2) != ck2:
        raise ValueError(f"TLE line 2 checksum mismatch "
                         f"(computed {_tle_checksum(line2)}, expected {ck2})")

    # Line 1: epoch (cols 18-31, YYDDD.DDDDDDDD)
    yy = int(line1[18:20])
    year = 2000 + yy if yy < TLE_Y2K_PIVOT else 1900 + yy
    day_frac = float(line1[20:32])
    epoch = (datetime(year, 1, 1, tzinfo=timezone.utc)
             + timedelta(days=day_frac - 1.0))

    # Line 2: orbital elements
    inc_deg = float(line2[8:16])
    raan_deg = float(line2[17:25])
    ecc = float("0." + line2[26:33].strip())
    argp_deg = float(line2[34:42])
    ma_deg = float(line2[43:51])
    n_rev_day = float(line2[52:63])

    # Mean motion (rev/day) -> semi-major axis (m)
    n_rad_s = n_rev_day * 2.0 * math.pi / SECONDS_PER_DAY
    a = (MU / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)

    return {
        "epoch": epoch,
        "a": a,
        "e": ecc,
        "i": math.radians(inc_deg),
        "raan": math.radians(raan_deg),
        "argp": math.radians(argp_deg),
        "ma": math.radians(ma_deg),
        "n_rev_day": n_rev_day,
    }


# ── Main compute pipeline ────────────────────────────────────────

def compute_schedule(
    tle_line1: str,
    tle_line2: str,
    timestamp: datetime,
    target: Target,
    fov_half: float,
    inst: Instrument | None = None,
    n_max: int = DEFAULT_N_MAX,
) -> dict:
    """Compute optimal imaging schedule from TLE and timestamp.

    Parameters
    ----------
    tle_line1, tle_line2 : str
        NORAD two-line element set.
    timestamp : datetime
        UTC time for schedule computation.
    target : Target
        Inertial celestial target.
    fov_half : float
        Instrument half-cone FOV (rad).
    inst : Instrument, optional
        CCD parameters. Defaults to standard instrument.
    n_max : int
        Maximum readout samples.

    Returns
    -------
    dict with flat key-value pairs (see status for edge cases).
    """
    if inst is None:
        inst = Instrument()

    tle = parse_tle(tle_line1, tle_line2)

    # Propagate RAAN from TLE epoch to timestamp
    dt_s = (timestamp - tle["epoch"]).total_seconds()
    omega_dot = j2_raan_rate(tle["a"], tle["e"], tle["i"])
    omega = propagate_raan(tle["raan"], omega_dot, dt_s)

    # Sun geometry
    beta_sun, uc_sun = sun_beta_uc(tle["i"], omega, timestamp)
    eclipse_s = eclipse_duration(tle["a"], beta_sun, sun_dist(timestamp))

    # Target geometry
    beta_tgt, _ = target_beta_uc(tle["i"], omega, target)
    open_sky_s = open_sky_budget(
        tle["a"], tle["i"], omega, target, fov_half, beta_sun, uc_sun,
    )

    # Optimize
    solution = optimize(open_sky_s, inst, n_max=n_max)

    # Status
    if eclipse_s == 0.0:
        status = "no_eclipse"
    elif open_sky_s == 0.0:
        status = "target_obstructed"
    elif solution is None:
        status = "insufficient_budget"
    else:
        status = "ok"

    # Flat result dict (C-struct friendly)
    result = {
        "timestamp": timestamp,
        "tle_epoch": tle["epoch"],
        "a_m": tle["a"],
        "e": tle["e"],
        "i_deg": math.degrees(tle["i"]),
        "raan_deg": math.degrees(omega) % 360.0,
        "beta_sun_deg": math.degrees(beta_sun),
        "eclipse_s": eclipse_s,
        "target_name": target.name,
        "beta_target_deg": math.degrees(beta_tgt),
        "open_sky_s": open_sky_s,
        "n_samples": solution.n_samples if solution else None,
        "exposure_s": solution.exposure_s if solution else None,
        "readout_s": (inst.readout_per_sample * solution.n_samples
                      if solution else None),
        "objective": solution.objective if solution else None,
        "status": status,
    }
    return result


# ── CLI ───────────────────────────────────────────────────────────

def _fmt(label: str, value: str, width: int = 18) -> str:
    """Format a label-value pair for CLI output."""
    return f"  {label + ':':<{width}} {value}"


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description="Flight scheduler: TLE + timestamp -> optimal imaging params",
    )
    p.add_argument("tle1", help="TLE line 1 (quoted)")
    p.add_argument("tle2", help="TLE line 2 (quoted)")
    p.add_argument("-t", "--timestamp", default=None,
                   help="ISO-8601 UTC datetime (default: now)")
    p.add_argument("--target-ra", type=float, default=SGR_A_RA_DEG,
                   help=f"Target RA in degrees (default: {SGR_A_RA_DEG} = Sgr A*)")
    p.add_argument("--target-dec", type=float, default=SGR_A_DEC_DEG,
                   help=f"Target Dec in degrees (default: {SGR_A_DEC_DEG} = Sgr A*)")
    p.add_argument("--target-name", default="Galactic Center",
                   help="Target name label")
    p.add_argument("--fov-half", type=float, default=DEFAULT_FOV_HALF_DEG,
                   help=f"Half-cone FOV in degrees (default: {DEFAULT_FOV_HALF_DEG})")
    p.add_argument("--startup", type=float, default=DEFAULT_STARTUP_S,
                   help=f"Startup time in seconds (default: {DEFAULT_STARTUP_S})")
    p.add_argument("--n-pixels", type=int, default=DEFAULT_N_PIXELS,
                   help=f"CCD pixel count (default: {DEFAULT_N_PIXELS})")
    p.add_argument("--read-rate", type=int, default=DEFAULT_READ_RATE,
                   help=f"Pixel readout rate (default: {DEFAULT_READ_RATE})")

    args = p.parse_args()

    # Parse timestamp
    if args.timestamp:
        ts = datetime.fromisoformat(args.timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = datetime.now(timezone.utc)

    # Build target and instrument
    tgt = Target(
        ra_rad=math.radians(args.target_ra),
        dec_rad=math.radians(args.target_dec),
        name=args.target_name,
    )
    inst = Instrument(
        startup_s=args.startup,
        n_pixels=args.n_pixels,
        read_rate=args.read_rate,
    )

    # Compute and print
    r = compute_schedule(
        args.tle1, args.tle2, ts, tgt,
        fov_half=math.radians(args.fov_half),
        inst=inst,
    )
    _print_result(r, inst)
    print()


def _print_result(r: dict, inst: Instrument) -> None:
    """Print a formatted schedule result (shared by demo and CLI)."""
    alt_km = (r["a_m"] - R_E) / 1000.0
    print()
    print("=" * 48)
    print("  FLIGHT SCHEDULE")
    print("=" * 48)
    print(_fmt("TLE Epoch", r["tle_epoch"].strftime("%Y-%m-%d %H:%M:%S UTC")))
    print(_fmt("Compute Time", r["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")))
    print(_fmt("Altitude", f"{alt_km:.1f} km"))
    print(_fmt("Inclination", f"{r['i_deg']:.2f} deg"))
    print(_fmt("RAAN", f"{r['raan_deg']:.2f} deg"))
    print(_fmt("Sun Beta", f"{r['beta_sun_deg']:.2f} deg"))
    print(_fmt("Eclipse", f"{r['eclipse_s']:.1f} s"))
    print(_fmt("Target", r["target_name"]))
    print(_fmt("Target Beta", f"{r['beta_target_deg']:.2f} deg"))
    print(_fmt("Open Sky", f"{r['open_sky_s']:.1f} s"))
    print("-" * 48)
    print(_fmt("Status", r["status"]))
    if r["status"] == "ok":
        print(_fmt("N (samples)", str(r["n_samples"])))
        print(_fmt("Exposure", f"{r['exposure_s']:.1f} s"))
        print(_fmt("Readout", f"{r['readout_s']:.1f} s"))
        total = inst.startup_s + r["exposure_s"] + r["readout_s"]
        print(_fmt("Total", f"{total:.1f} s"))
    print("=" * 48)


def demo() -> None:
    """Run hard-coded examples covering nominal and edge cases."""
    # ── Instrument (same as main.py) ──────────────────────────────
    inst = Instrument()
    fov_half = math.radians(DEFAULT_FOV_HALF_DEG)

    # ── Representative ISS-class TLE (51.6 deg, 500 km, RAAN=45 deg) ──
    # Epoch: 2027-03-21 12:00 UTC  (day 080.5)
    TLE1 = "1 25544U 98067A   27080.50000000  .00016717  00000-0  10270-3 0  9997"
    TLE2 = "2 25544  51.6400  45.0000 0001000   0.0000   0.0000 15.50000000    09"

    cases = [
        # (label, timestamp_str)
        ("Nominal - June (good GC geometry)",  "2027-06-15T12:00:00"),
        ("Nominal - April",                    "2027-04-15T12:00:00"),
        ("Target obstructed - December",       "2027-12-15T12:00:00"),
        ("Marginal budget - January",          "2027-01-15T12:00:00"),
    ]

    print("\n" + "=" * 48)
    print("  FLIGHT.PY - DEMO MODE")
    print("  TLE: ISS-class  i=51.6 deg  h~500 km")
    print("  Target: Galactic Center (Sgr A*)")
    print("=" * 48)

    for label, ts_str in cases:
        ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        print(f"\n>>> {label}")
        r = compute_schedule(TLE1, TLE2, ts, GALACTIC_CENTER, fov_half, inst)
        _print_result(r, inst)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        demo()
    else:
        main()
