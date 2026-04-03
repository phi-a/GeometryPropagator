"""
Generate DarkNESS Constraint-Driven Operations Architecture document.
Run: python docs/generate_dao_explainer.py
Output: docs/DAO_Geometric_Constraint_Engine.pdf

Editorial architecture: prose introduces, equations express, tables collect,
boxes anchor, flow diagrams show process.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Font registration ──────────────────────────────────────────────────────────
BODY_FONT = "Helvetica"
BOLD_FONT = "Helvetica-Bold"
ITAL_FONT = "Helvetica-Oblique"
MONO_FONT = "Courier"

for reg, bld, ita, mon in [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
     "C:/Windows/Fonts/ariali.ttf", "C:/Windows/Fonts/cour.ttf"),
]:
    if os.path.exists(reg):
        try:
            pdfmetrics.registerFont(TTFont("ArialUni",      reg))
            pdfmetrics.registerFont(TTFont("ArialUni-Bold", bld))
            pdfmetrics.registerFont(TTFont("ArialUni-Ital", ita))
            pdfmetrics.registerFont(TTFont("CourierUni",    mon))
            BODY_FONT = "ArialUni"
            BOLD_FONT = "ArialUni-Bold"
            ITAL_FONT = "ArialUni-Ital"
            MONO_FONT = "CourierUni"
        except Exception:
            pass
        break

# ── Colors ─────────────────────────────────────────────────────────────────────
NAVY    = colors.HexColor("#0D1B2A")
BLUE    = colors.HexColor("#1B4F8A")
LTBLUE  = colors.HexColor("#D6E4F0")
TEAL    = colors.HexColor("#2E86AB")
RULE    = colors.HexColor("#CCCCCC")
WHITE   = colors.white
BLACK   = colors.black
GRAY    = colors.HexColor("#555555")
LTGRAY  = colors.HexColor("#F5F5F5")
GOLD    = colors.HexColor("#C8902A")
LTGOLD  = colors.HexColor("#FFF8E1")
GREEN   = colors.HexColor("#2E7D32")
RED     = colors.HexColor("#B71C1C")


# ── Helpers ────────────────────────────────────────────────────────────────────

def S(name, parent, **kw):
    return ParagraphStyle(name, parent=parent, **kw)


def ruled_table(data, col_widths, header_bg=NAVY, alt=None):
    alt = alt or [WHITE, LTGRAY]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), alt),
        ("GRID",           (0, 0), (-1, -1), 0.3, RULE),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 7),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def concept_box(text, parent_style):
    """Blue-bordered box with one conceptual statement. No equations."""
    p = Paragraph(text, parent_style)
    t = Table([[p]], colWidths=[6.3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LTBLUE),
        ("BOX",           (0, 0), (-1, -1), 1.2, BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    return t


def flow_diagram(steps, base_style):
    """Horizontal flow diagram: numbered steps with arrows."""
    arrow = Paragraph("\u2192", S("_arr", base_style, fontSize=14,
                                  textColor=GRAY, alignment=TA_CENTER))
    AW, SW = 0.25 * inch, 1.0 * inch
    cells, widths = [], []
    for i, (num, title, desc) in enumerate(steps):
        cell = Table(
            [[Paragraph(num, S(f"_fn{i}", base_style, fontSize=11,
                               textColor=WHITE, fontName=BOLD_FONT,
                               alignment=TA_CENTER))],
             [Paragraph(title, S(f"_ft{i}", base_style, fontSize=8,
                                 textColor=NAVY, fontName=BOLD_FONT,
                                 alignment=TA_CENTER, leading=10))],
             [Paragraph(desc, S(f"_fd{i}", base_style, fontSize=7.5,
                                textColor=GRAY, fontName=BODY_FONT,
                                alignment=TA_CENTER, leading=10))]],
            colWidths=[SW])
        cell.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), BLUE),
            ("BACKGROUND", (0, 1), (0, 2), LTBLUE),
            ("BOX",        (0, 0), (-1, -1), 0.5, BLUE),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 3),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
        ]))
        cells.append(cell)
        widths.append(SW)
        if i < len(steps) - 1:
            cells.append(arrow)
            widths.append(AW)
    tbl = Table([cells], colWidths=widths, rowHeights=[0.85 * inch])
    tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
    ]))
    return tbl


# ── Build ──────────────────────────────────────────────────────────────────────

def build():
    os.makedirs("docs", exist_ok=True)
    out = "docs/DAO_Geometric_Constraint_Explainer.pdf"
    doc = SimpleDocTemplate(
        out, pagesize=letter,
        leftMargin=0.85*inch, rightMargin=0.85*inch,
        topMargin=0.75*inch,  bottomMargin=0.75*inch,
    )

    base = getSampleStyleSheet()["Normal"]

    title_s   = S("T",  base, fontSize=20, textColor=NAVY, fontName=BOLD_FONT,
                  spaceAfter=4, leading=24)
    sub_s     = S("Su", base, fontSize=10, textColor=GRAY, fontName=BODY_FONT,
                  spaceAfter=2, leading=14)
    sec_s     = S("Se", base, fontSize=12, textColor=BLUE, fontName=BOLD_FONT,
                  spaceBefore=12, spaceAfter=4)
    subsec_s  = S("SS", base, fontSize=10, textColor=TEAL, fontName=BOLD_FONT,
                  spaceBefore=8, spaceAfter=3)
    body_s    = S("Bo", base, fontSize=9.5, textColor=BLACK, fontName=BODY_FONT,
                  leading=14, spaceAfter=4)
    small_s   = S("Sm", base, fontSize=8.5, textColor=GRAY, fontName=BODY_FONT,
                  leading=12)
    bold_s    = S("Bd", base, fontSize=9.5, textColor=BLACK, fontName=BOLD_FONT,
                  leading=14)
    mono_s    = S("Mo", base, fontSize=8.5, textColor=NAVY, fontName=MONO_FONT,
                  leading=12)
    mono_c    = S("Mc", base, fontSize=9, textColor=NAVY, fontName=MONO_FONT,
                  leading=13, alignment=TA_CENTER)
    cap_s     = S("Ca", base, fontSize=8, textColor=GRAY, fontName=ITAL_FONT,
                  alignment=TA_CENTER)
    callout_s = S("Co", base, fontSize=10, textColor=NAVY, fontName=BOLD_FONT,
                  alignment=TA_CENTER, leading=15)
    tag_blue  = S("TB", base, fontSize=9, textColor=BLUE,  fontName=BOLD_FONT)
    tag_teal  = S("TT", base, fontSize=9, textColor=TEAL,  fontName=BOLD_FONT)
    tag_green = S("TG", base, fontSize=9, textColor=GREEN, fontName=BOLD_FONT)
    tag_gold  = S("TGo",base, fontSize=9, textColor=GOLD,  fontName=BOLD_FONT)
    tag_red   = S("TR", base, fontSize=9, textColor=RED,   fontName=BOLD_FONT)

    story = []
    P = Paragraph

    # ══════════════════════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════════════════════
    hdr = Table([[
        P("DarkNESS  Constraint-Driven Operations", title_s),
        P("Conceptual Framework", S("R", base, fontSize=9, textColor=GRAY,
          fontName=BODY_FONT, alignment=TA_RIGHT)),
    ]], colWidths=[4.8*inch, 2.2*inch])
    hdr.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story += [
        hdr,
        P("Operational autonomy as continuous constraint evaluation \u2014 "
          "DarkNESS 6U CubeSat", sub_s),
        HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=10),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 1. OPERATIONAL PHILOSOPHY
    # ══════════════════════════════════════════════════════════════════════════
    story.append(P("1.  Operational Philosophy", sec_s))

    story.append(P(
        "DarkNESS science collection depends on coupled physical constraints "
        "that vary with every orbit. "
        "Thermal state, power margin, eclipse geometry, target visibility, "
        "and detector readiness must all be valid simultaneously. "
        "If any single condition fails, observation cannot proceed.",
        body_s))

    story.append(P(
        "Conventional CubeSat missions use procedural command scripts. "
        "Those scripts assume that conditions will be acceptable when commands execute. "
        "DarkNESS cannot rely on that assumption because its constraint coupling "
        "changes with orbit precession, seasonal Sun angle, and thermal evolution.",
        body_s))

    story.append(P(
        "The DarkNESS Architecture for Operations (DAO) encodes one rule. "
        "If all constraints are satisfied, execute the science activity. "
        "If any constraint is violated, stop. "
        "The architecture defines which constraints exist, how they are evaluated, "
        "and what operational state transitions they trigger.",
        body_s))

    story.append(Spacer(1, 4))
    story.append(concept_box(
        "Operations are not scheduled by timeline. "
        "They are permitted only when physical constraints are simultaneously satisfied.",
        callout_s))
    story.append(Spacer(1, 8))

    # ══════════════════════════════════════════════════════════════════════════
    # 2. SYMBOLS AND NOTATION
    # ══════════════════════════════════════════════════════════════════════════
    story.append(P("2.  Symbols and Notation", sec_s))
    story.append(P(
        "The following symbols appear throughout this document. "
        "Directional quantities are unit vectors on the celestial sphere "
        "or the unit sphere centered at the observer.",
        body_s))

    sym_data = [
        [P("Symbol", bold_s), P("Definition", bold_s), P("Units", bold_s)],
        # State
        [P("x(t)", mono_s), P("Spacecraft state vector at time t", body_s),
         P("\u2014", small_s)],
        [P("g\u1D62(x)", mono_s),
         P("Constraint function i; satisfied when g\u1D62 \u2265 0", body_s),
         P("dimensionless", small_s)],
        [P("F", mono_s),
         P("Feasible region: set of states where all g\u1D62 \u2265 0", body_s),
         P("subset of state space", small_s)],
        # Directions
        [P("r\u0302", mono_s), P("Observer orbital position direction", body_s),
         P("unit vector", small_s)],
        [P("s\u0302", mono_s), P("Sun direction (inertial)", body_s),
         P("unit vector", small_s)],
        [P("t\u0302", mono_s), P("Celestial target direction (inertial, J2000)", body_s),
         P("unit vector", small_s)],
        [P("b\u0302", mono_s), P("Detector boresight direction (body frame)", body_s),
         P("unit vector", small_s)],
        [P("e\u0302_earth", mono_s), P("Earth center direction from observer", body_s),
         P("unit vector", small_s)],
        [P("n\u0302", mono_s), P("Surface facet outward normal", body_s),
         P("unit vector", small_s)],
        # Angles
        [P("\u03B1_shadow", mono_s), P("Shadow cone half-angle", body_s),
         P("rad", small_s)],
        [P("\u03B1_fov", mono_s), P("Field-of-view half-angle", body_s),
         P("rad", small_s)],
        [P("\u03B1_excl", mono_s), P("Sun exclusion half-angle", body_s),
         P("rad", small_s)],
        [P("\u03B1_E", mono_s), P("Earth angular radius from orbit", body_s),
         P("rad", small_s)],
        [P("\u03B3", mono_s), P("Nadir angle: angle between n\u0302 and e\u0302_earth", body_s),
         P("rad", small_s)],
        [P("\u03B2", mono_s), P("Sun-beta angle (Sun elevation above orbit plane)", body_s),
         P("rad", small_s)],
        # View factors and thermal
        [P("F_earth", mono_s),
         P("Earth view factor (cosine-weighted solid angle fraction)", body_s),
         P("0 to 1", small_s)],
        [P("F_space", mono_s), P("Deep-space view factor", body_s),
         P("0 to 1", small_s)],
        [P("T_det", mono_s), P("Detector temperature", body_s), P("K", small_s)],
        [P("T_max", mono_s), P("Maximum allowable detector temperature", body_s),
         P("K", small_s)],
        [P("\u03B5", mono_s), P("Surface emissivity", body_s),
         P("dimensionless", small_s)],
        [P("\u03C3", mono_s), P("Stefan-Boltzmann constant", body_s),
         P("W m\u207B\u00B2 K\u207B\u2074", small_s)],
        # Optimization
        [P("N", mono_s), P("Number of exposures (frames) per pass", body_s),
         P("integer", small_s)],
        [P("t_exp", mono_s), P("Exposure time per frame", body_s), P("s", small_s)],
        [P("t_read", mono_s), P("Readout time per frame", body_s), P("s", small_s)],
        [P("\u0394t_eclipse", mono_s), P("Eclipse duration", body_s),
         P("s", small_s)],
        [P("\u0394t_obs", mono_s),
         P("Observation window (feasible sub-interval of eclipse)", body_s),
         P("s", small_s)],
        # Signal
        [P("S", mono_s), P("Source photon count per frame", body_s),
         P("photons", small_s)],
        [P("B", mono_s), P("Background photon count per frame", body_s),
         P("photons", small_s)],
        [P("SNR", mono_s), P("Signal-to-noise ratio", body_s),
         P("dimensionless", small_s)],
        # Power and data
        [P("SOC", mono_s), P("Battery state of charge", body_s),
         P("fraction", small_s)],
        [P("E_avail", mono_s), P("Available energy per pass", body_s),
         P("J", small_s)],
        [P("d_frame", mono_s), P("Data size per frame", body_s),
         P("bytes", small_s)],
    ]
    story.append(ruled_table(sym_data, [1.0*inch, 3.8*inch, 1.5*inch],
                             alt=[WHITE, LTBLUE]))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. THE OPERATIONAL FEASIBLE REGION
    # ══════════════════════════════════════════════════════════════════════════
    story.append(P("3.  The Operational Feasible Region", sec_s))

    story.append(P(
        "At any instant the spacecraft occupies a state x(t) that includes "
        "orbit position, attitude, thermal state, battery state, and target geometry. "
        "Each operational constraint defines an inequality on this state.",
        body_s))

    story.append(P(
        "g\u2081(x)  \u2265  0     eclipse constraint", mono_s))
    story.append(P(
        "g\u2082(x)  \u2265  0     target visibility constraint", mono_s))
    story.append(P(
        "g\u2083(x)  \u2265  0     sun exclusion constraint", mono_s))
    story.append(P(
        "g\u2084(x)  \u2265  0     thermal constraint", mono_s))
    story.append(P(
        "g\u2085(x)  \u2265  0     power constraint", mono_s))
    story.append(P(
        "g\u2086(x)  \u2265  0     timing / window constraint", mono_s))
    story.append(Spacer(1, 4))

    story.append(P(
        "The feasible region is the intersection of all constraint half-spaces:",
        body_s))
    story.append(P(
        "F  =  { x  |  g\u2081(x) \u2265 0  \u2227  g\u2082(x) \u2265 0  "
        "\u2227  \u2026  \u2227  g\u2086(x) \u2265 0 }",
        mono_c))
    story.append(Spacer(1, 6))

    story.append(concept_box(
        "Science operations are allowed only when x(t) \u2208 F.",
        callout_s))
    story.append(Spacer(1, 6))

    story.append(P(
        "As the spacecraft orbits, x(t) traces a path through state space. "
        "That path enters and exits F. "
        "Each entry begins a potential observation window. "
        "Each exit terminates science.",
        body_s))

    story.append(P(
        "<b>Robustness</b>: if eclipse is shorter than predicted, "
        "the window shrinks automatically. "
        "<b>Self-synchronization</b>: events occur when constraint boundaries "
        "are crossed, not at commanded times. "
        "<b>Optimizability</b>: F is a well-defined set over which "
        "science return can be maximized.",
        body_s))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════════════
    # 4. OPERATIONAL CONSTRAINT TAXONOMY
    # ══════════════════════════════════════════════════════════════════════════
    story.append(P("4.  Operational Constraint Taxonomy", sec_s))
    story.append(P(
        "Each constraint g\u1D62(x) enforces a physical requirement. "
        "All inequalities use the standard form g(x) \u2265 0 where "
        "positive means satisfied and zero is the boundary.",
        body_s))

    oc_data = [
        [P("Category", bold_s), P("Constraint", bold_s),
         P("g\u1D62(x) \u2265 0", bold_s), P("If Violated", bold_s)],

        [P("Geometry", tag_blue),
         P("Observer inside umbra", body_s),
         P("\u03B1_shadow \u2212 angle(r\u0302, \u2212s\u0302) \u2265 0", mono_s),
         P("Background too high", small_s)],
        [P("Geometry", tag_blue),
         P("Target within FOV", body_s),
         P("\u03B1_fov \u2212 angle(t\u0302, b\u0302) \u2265 0", mono_s),
         P("No observation", small_s)],
        [P("Geometry", tag_blue),
         P("Sun outside baffle", body_s),
         P("angle(s\u0302, b\u0302) \u2212 \u03B1_excl \u2265 0", mono_s),
         P("Detector damage risk", small_s)],
        [P("Geometry", tag_blue),
         P("Target above horizon", body_s),
         P("90\u00B0 \u2212 angle(t\u0302, e\u0302_earth) \u2265 0", mono_s),
         P("Occluded by Earth", small_s)],

        [P("Thermal", tag_red),
         P("Detector temperature", body_s),
         P("T_max \u2212 T_det(t) \u2265 0", mono_s),
         P("\u2192 Thermal Safe", small_s)],
        [P("Thermal", tag_red),
         P("Thermal recovery", body_s),
         P("T_ready \u2212 T_det(t) \u2265 0", mono_s),
         P("Delay science", small_s)],

        [P("Power", tag_gold),
         P("Battery charge", body_s),
         P("SOC \u2212 SOC_min \u2265 0", mono_s),
         P("\u2192 Power Safe", small_s)],
        [P("Power", tag_gold),
         P("Energy for N exposures", body_s),
         P("E_avail \u2212 N\u00B7E_exp \u2265 0", mono_s),
         P("Reduce N", small_s)],

        [P("Timing", tag_teal),
         P("Sequence fits in window", body_s),
         P("\u0394t_eclipse \u2212 N\u00B7(t_exp+t_read) \u2265 0", mono_s),
         P("Reduce N or t_exp", small_s)],
        [P("Timing", tag_teal),
         P("Exposure meets signal floor", body_s),
         P("t_exp \u2212 t_exp_min \u2265 0", mono_s),
         P("Fewer N, longer t_exp", small_s)],

        [P("Signal", tag_green),
         P("Photon count meets SNR", body_s),
         P("S\u00B7t_exp \u2212 N_ph_req \u2265 0", mono_s),
         P("Extend t_exp", small_s)],
        [P("Signal", tag_green),
         P("Background below budget", body_s),
         P("B_max \u2212 B(F_earth)\u00B7t_exp \u2265 0", mono_s),
         P("Shorten t_exp", small_s)],

        [P("Data", tag_gold),
         P("Buffer available", body_s),
         P("buf_free \u2212 N\u00B7d_frame \u2265 0", mono_s),
         P("Downlink first", small_s)],
    ]
    story.append(ruled_table(oc_data, [0.8*inch, 1.55*inch, 2.35*inch, 1.6*inch]))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    # 5. GEOMETRIC EVALUATION OF CONSTRAINTS
    # ══════════════════════════════════════════════════════════════════════════
    story.append(P("5.  Geometric Evaluation of Constraints", sec_s))

    story.append(P(
        "Several constraint functions g\u1D62(x) depend on spatial relationships "
        "that change with orbit position and time. "
        "These are evaluated using geometric primitives: vectors, cones, "
        "spherical caps, and hemispheres. "
        "As time advances, primitives move through analytic frame transforms "
        "and ephemeris lookups. No force integration is required.",
        body_s))

    story.append(P("Geometric Primitives", subsec_s))
    prim_data = [
        [P("Primitive", bold_s), P("Definition", bold_s),
         P("Inequality", bold_s), P("Evolves With", bold_s)],
        [P("Direction", body_s),
         P("v\u0302, |v\u0302| = 1", mono_s),
         P("(used in dot products)", small_s),
         P("Frame rotation, ephemeris", small_s)],
        [P("Cone", body_s),
         P("Axis a\u0302, half-angle \u03B1", mono_s),
         P("v\u0302 \u00B7 a\u0302 \u2265 cos \u03B1", mono_s),
         P("Attitude law", small_s)],
        [P("Spherical Cap", body_s),
         P("Center e\u0302, radius \u03B1", mono_s),
         P("angle(v\u0302, e\u0302) \u2264 \u03B1", mono_s),
         P("Earth direction, altitude", small_s)],
        [P("Hemisphere", body_s),
         P("Normal n\u0302", mono_s),
         P("v\u0302 \u00B7 n\u0302 \u2265 0", mono_s),
         P("Surface facet, attitude", small_s)],
        [P("Shadow Cone", body_s),
         P("Conical umbra: k=R\u2091/r, \u03B5=(R\u2609\u2212R\u2091)/D", mono_s),
         P("\u03BD=acos((k\u03B5+\u221A((1\u2212k\u00B2)(1\u2212\u03B5\u00B2)))/|cos\u03B2|)", mono_s),
         P("Orbit position, Sun distance", small_s)],
        [P("Time Interval", body_s),
         P("[T_in, T_out]", mono_s),
         P("T_in \u2264 t \u2264 T_out", mono_s),
         P("Eclipse entry/exit", small_s)],
        [P("Orbit Curve", body_s),
         P("r(t) = f(elements, t)", mono_s),
         P("(analytic, no integration)", small_s),
         P("Kepler or ephemeris", small_s)],
    ]
    story.append(ruled_table(prim_data, [1.0*inch, 1.8*inch, 1.85*inch, 1.65*inch],
                             alt=[WHITE, LTBLUE]))
    story.append(Spacer(1, 6))

    story.append(P(
        "Every geometric constraint reduces to an angle test or dot product. "
        "These operations are fast, exact, and suitable for both ground planning "
        "and onboard autonomous execution.",
        body_s))

    # ── Conical Umbra Model ──────────────────────────────────────────
    story.append(P("Conical Umbra Model", subsec_s))
    story.append(P(
        "The eclipse geometry uses an exact closed-form conical umbra model. "
        "Unlike the cylindrical approximation (shadow cylinder of radius "
        "R\u2091 aligned with the Sun-Earth axis), the conical model accounts "
        "for the Sun's finite angular size. The umbra cone narrows with "
        "distance from Earth, producing a shorter eclipse.",
        body_s))

    story.append(P(
        "For a circular orbit at radius r with Sun-beta angle \u03B2 "
        "and Earth-Sun distance D, define:",
        body_s))
    story.append(P(
        "k = R\u2091 / r          (Earth angular size ratio)", mono_s))
    story.append(P(
        "\u03B5 = (R\u2609 \u2212 R\u2091) / D   (cone taper parameter)", mono_s))
    story.append(P(
        "c = |cos \u03B2|         (orbit-plane projection)", mono_s))
    story.append(Spacer(1, 4))

    story.append(P(
        "The eclipse half-angle is:", body_s))
    story.append(P(
        "\u03BD = arccos( (k\u03B5 + \u221A((1\u2212k\u00B2)(1\u2212\u03B5\u00B2))) / c )",
        mono_c))
    story.append(Spacer(1, 4))

    story.append(P(
        "When \u03B5 \u2192 0 this collapses to the cylindrical result "
        "\u03BD = arccos(\u221A(1\u2212k\u00B2)/c). "
        "The conical formula is exact for circular orbits with no approximation.",
        body_s))

    story.append(P(
        "Earth-Sun distance D is computed from the solar mean anomaly: "
        "D = AU \u00B7 (1.00014 \u2212 0.01671 cos g \u2212 0.00014 cos 2g). "
        "This provides seasonal accuracy at zero additional cost.",
        body_s))

    story.append(Spacer(1, 6))
    story.append(P("Accuracy: Dynamic D Impact on Eclipse Duration (500 km)", subsec_s))
    story.append(P(
        "The seasonal variation in Earth-Sun distance (\u00B11.7%) has a small "
        "but measurable effect on eclipse duration, largest near grazing.",
        body_s))

    d_table = [
        [P("\u03B2", bold_s), P("D = perihelion", bold_s),
         P("D = 1 AU", bold_s), P("D = aphelion", bold_s),
         P("Spread", bold_s)],
        [P("0\u00B0",  mono_s), P("2136.7 s", mono_s), P("2136.9 s", mono_s),
         P("2137.0 s", mono_s), P("\u22120.28 s", mono_s)],
        [P("20\u00B0", mono_s), P("2089.0 s", mono_s), P("2089.2 s", mono_s),
         P("2089.3 s", mono_s), P("\u22120.30 s", mono_s)],
        [P("40\u00B0", mono_s), P("1904.2 s", mono_s), P("1904.4 s", mono_s),
         P("1904.6 s", mono_s), P("\u22120.39 s", mono_s)],
        [P("55\u00B0", mono_s), P("1535.6 s", mono_s), P("1535.9 s", mono_s),
         P("1536.2 s", mono_s), P("\u22120.60 s", mono_s)],
        [P("63\u00B0", mono_s), P("1056.0 s", mono_s), P("1056.6 s", mono_s),
         P("1057.1 s", mono_s), P("\u22121.03 s", mono_s)],
    ]
    story.append(ruled_table(d_table,
                             [0.5*inch, 1.3*inch, 1.1*inch, 1.3*inch, 1.0*inch],
                             alt=[WHITE, LTBLUE]))
    story.append(Spacer(1, 8))

    story.append(P("Accuracy: Beta-Error Timing Impact (500 km, non-SSO max drift)", subsec_s))
    story.append(P(
        "Schedule resolution determines how often \u03B2 is recomputed. "
        "Beta-error timing sensitivity dominates dynamic D by two orders of magnitude.",
        body_s))

    b_table = [
        [P("Step", bold_s), P("\u0394\u03B2 non-SSO", bold_s),
         P("\u0394t @ \u03B2=30\u00B0", bold_s),
         P("\u0394t @ \u03B2=60\u00B0", bold_s),
         P("\u0394t SSO", bold_s)],
        [P("1 min",  mono_s), P("0.006\u00B0", mono_s),
         P("0.0 s",  mono_s), P("0.2 s",   mono_s), P("0.00 s", mono_s)],
        [P("1 hr",   mono_s), P("0.375\u00B0", mono_s),
         P("1.6 s",  mono_s), P("11.6 s",  mono_s), P("0.02 s", mono_s)],
        [P("1.5 hr", mono_s), P("0.562\u00B0", mono_s),
         P("2.5 s",  mono_s), P("17.4 s",  mono_s), P("0.04 s", mono_s)],
        [P("24 hr",  mono_s), P("9.000\u00B0", mono_s),
         P("39.3 s", mono_s), P("277.9 s", mono_s), P("0.57 s", mono_s)],
    ]
    story.append(ruled_table(b_table,
                             [0.7*inch, 1.2*inch, 1.1*inch, 1.1*inch, 0.9*inch],
                             alt=[WHITE, LTBLUE]))
    story.append(Spacer(1, 6))

    story.append(concept_box(
        "Beta step resolution dominates accuracy by two orders of magnitude "
        "over dynamic D. For non-SSO orbits above \u03B2 \u2248 45\u00B0, "
        "schedule resolution is the only lever that meaningfully affects "
        "eclipse timing accuracy.",
        callout_s))
    story.append(Spacer(1, 8))

    story.append(P("Celestial Targets as Constraint Directions", subsec_s))
    story.append(P(
        "X-ray science targets are point sources at infinite distance. "
        "Each target is a fixed direction in the J2000 inertial frame. "
        "Visibility requires the target direction to lie inside the FOV cone "
        "while the observer is in eclipse.",
        body_s))

    tgt_data = [
        [P("Target", bold_s), P("Inertial Direction", bold_s),
         P("Constraint", bold_s), P("Visibility", bold_s)],
        [P("Galactic Center", body_s),
         P("RA 17h 45m, Dec \u221229\u00B0", mono_s),
         P("angle(g\u0302_gc, b\u0302) \u2264 \u03B1_fov\nAND eclipse_valid", mono_s),
         P("~May\u2013Sept", small_s)],
        [P("Cygnus X-1", body_s),
         P("RA 19h 58m, Dec +35\u00B0", mono_s),
         P("angle(g\u0302_cyg, b\u0302) \u2264 \u03B1_fov\nAND eclipse_valid", mono_s),
         P("~Jun\u2013Nov", small_s)],
        [P("Generic source", body_s),
         P("Any (RA, Dec) \u2192 t\u0302", mono_s),
         P("angle(t\u0302, b\u0302) \u2264 \u03B1_fov", mono_s),
         P("Target-agnostic", small_s)],
    ]
    story.append(ruled_table(tgt_data, [0.9*inch, 1.6*inch, 2.1*inch, 1.7*inch],
                             alt=[WHITE, LTBLUE]))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════════════════════════
    # 6. THERMAL-GEOMETRIC COUPLING
    # ══════════════════════════════════════════════════════════════════════════
    story.append(P("6.  Thermal\u2013Geometric Coupling", sec_s))

    story.append(P(
        "Detector temperature is determined by the radiative environment. "
        "The Earth subtends a spherical cap on the unit sphere around the observer. "
        "The view factor F_earth is the cosine-weighted solid angle of that cap "
        "projected onto the detector hemisphere.",
        body_s))

    story.append(P("View Factor Geometry", subsec_s))
    vf_data = [
        [P("Quantity", bold_s), P("Expression", bold_s), P("Notes", bold_s)],
        [P("Earth angular radius", body_s),
         P("\u03B1\u2091 = arcsin(R\u2091 / r)", mono_s),
         P("R\u2091 = 6371 km, r = orbital radius", small_s)],
        [P("Nadir angle", body_s),
         P("\u03B3 = angle(n\u0302, e\u0302_earth)", mono_s),
         P("n\u0302 = facet normal", small_s)],
        [P("Three regimes", body_s),
         P("\u03B3 \u2265 90\u00B0+\u03B1\u2091 \u2192 no view\n"
           "\u03B3 \u2264 90\u00B0\u2212\u03B1\u2091 \u2192 full disk\n"
           "otherwise \u2192 partial clip", mono_s),
         P("Analytic, no ray tracing", small_s)],
        [P("View factor", body_s),
         P("F = (1/\u03C0)\u222B(n\u0302\u00B7\u03C9\u0302)d\u03A9\n"
           "\u2248 sin\u00B2(\u03B1\u2091)\u00B7max(0,cos \u03B3)", mono_s),
         P("Exact: cap-hemisphere overlap\nApprox: away from 90\u00B0\u00B1\u03B1\u2091", small_s)],
        [P("Deep-space view", body_s),
         P("F_space \u2248 1 \u2212 F_earth \u2212 F_body", mono_s),
         P("Radiator (\u2212y face) sees space", small_s)],
    ]
    story.append(ruled_table(vf_data, [1.2*inch, 2.5*inch, 2.6*inch],
                             alt=[WHITE, LTGOLD]))
    story.append(Spacer(1, 6))

    story.append(P("Thermal Balance", subsec_s))
    story.append(P(
        "During eclipse the detector equilibrium temperature is set by "
        "radiative heat balance. "
        "Let \u03B5 denote emissivity, \u03C3 the Stefan-Boltzmann constant, "
        "A_det the detector area, and A_rad the radiator area.",
        body_s))

    story.append(P(
        "P_in  = \u03B5\u00B7\u03C3\u00B7T_earth\u2074\u00B7F_earth\u00B7A_det "
        " +  P_detector  +  P_electronics",
        mono_c))
    story.append(P(
        "P_out = \u03B5\u00B7\u03C3\u00B7T_det\u2074\u00B7A_rad\u00B7F_space",
        mono_c))
    story.append(P(
        "C \u00B7 dT/dt  =  P_in \u2212 P_out", mono_c))
    story.append(P(
        "g\u2084(x) = T_max \u2212 T_det(t)  \u2265  0", mono_c))
    story.append(Spacer(1, 6))

    story.append(concept_box(
        "Thermal state is geometric. Detector temperature is driven by the "
        "Earth view factor, which depends on orbit position and attitude.",
        callout_s))
    story.append(Spacer(1, 4))

    story.append(P(
        "Because F_earth depends on orbit geometry, the thermal constraint "
        "boundary shifts with altitude, inclination, and position along the orbit.",
        body_s))

    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    # 7. CONSTRAINT ALGEBRA → DODAF
    # ══════════════════════════════════════════════════════════════════════════
    story.append(P("7.  From Constraint Algebra to DoDAF", sec_s))

    story.append(P(
        "Each constraint inequality becomes an operational rule. "
        "Each boundary crossing becomes an event. "
        "The conjunction of all satisfied rules defines the operational state. "
        "This translation is mechanical.",
        body_s))

    story.append(P("Translation Pipeline", subsec_s))
    story.append(flow_diagram([
        ("1", "Constraint\nInequality", "g\u1D62(x) \u2265 0"),
        ("2", "Boolean\nFlag",          "True / False"),
        ("3", "OV-6a\nRule",            "IF flag THEN \u2026"),
        ("4", "OV-6c\nEvent",           "boundary crossing"),
        ("5", "OV-6b\nTransition",      "mode change"),
    ], base))
    story.append(Spacer(1, 8))

    story.append(P("OV-6a \u2014 Operational Rules", subsec_s))
    ov6a = [
        [P("ID", bold_s), P("Inequality", bold_s),
         P("Rule", bold_s), P("Flag", bold_s)],
        [P("R-01", mono_s),
         P("\u03B1_sh \u2212 angle(r\u0302,\u2212s\u0302) \u2265 0", mono_s),
         P("IF in shadow cone THEN eclipse valid", small_s),
         P("eclipse_valid", mono_s)],
        [P("R-02", mono_s),
         P("\u03B1_fov \u2212 angle(t\u0302,b\u0302) \u2265 0", mono_s),
         P("IF target in FOV THEN target visible", small_s),
         P("target_visible", mono_s)],
        [P("R-03", mono_s),
         P("angle(s\u0302,b\u0302) \u2212 \u03B1_excl \u2265 0", mono_s),
         P("IF Sun outside exclusion THEN pointing valid", small_s),
         P("sun_clear", mono_s)],
        [P("R-04", mono_s),
         P("T_max \u2212 T_det(F_earth) \u2265 0", mono_s),
         P("IF within thermal limit THEN thermal valid", small_s),
         P("thermal_valid", mono_s)],
        [P("R-05", mono_s),
         P("SOC \u2212 SOC_min \u2265 0", mono_s),
         P("IF battery above threshold THEN power valid", small_s),
         P("power_valid", mono_s)],
        [P("R-06", mono_s),
         P("\u0394t \u2212 N\u00B7(t_exp+t_read) \u2265 0", mono_s),
         P("IF sequence fits THEN window adequate", small_s),
         P("window_ok", mono_s)],
        [P("R-07", mono_s),
         P("R-01 \u2227 R-02 \u2227 R-03\n"
           "\u2227 R-04 \u2227 R-05 \u2227 R-06", mono_s),
         P("IF all rules satisfied THEN Observe", small_s),
         P("OBSERVE", mono_s)],
    ]
    story.append(KeepTogether([
        ruled_table(ov6a, [0.45*inch, 2.0*inch, 2.45*inch, 1.4*inch],
                    alt=[WHITE, LTBLUE]),
        Spacer(1, 8),
    ]))

    story.append(P("OV-6b \u2014 State Transitions", subsec_s))
    ov6b = [
        [P("From", bold_s), P("To", bold_s),
         P("Trigger", bold_s), P("Condition", bold_s)],
        [P("Standby",  tag_blue), P("Observe",      tag_green),
         P("R-07 \u2192 True", small_s),
         P("x(t) enters F", small_s)],
        [P("Observe",  tag_green), P("Standby",     tag_blue),
         P("eclipse_valid \u2192 False", small_s),
         P("x(t) exits shadow cone", small_s)],
        [P("Observe",  tag_green), P("Thermal Safe", tag_red),
         P("thermal_valid \u2192 False", small_s),
         P("T_det crosses T_max", small_s)],
        [P("Observe",  tag_green), P("Power Safe",   tag_red),
         P("power_valid \u2192 False", small_s),
         P("SOC crosses SOC_min", small_s)],
        [P("Any",      tag_teal), P("Downlink",      tag_gold),
         P("Contact window", small_s),
         P("Ground station in LOS", small_s)],
        [P("Safe",     tag_red),  P("Standby",       tag_blue),
         P("Recovery met", small_s),
         P("T_det \u2264 T_ready or SOC \u2265 SOC_safe", small_s)],
    ]
    story.append(KeepTogether([
        ruled_table(ov6b, [0.85*inch, 0.9*inch, 1.75*inch, 2.8*inch]),
        Spacer(1, 8),
    ]))

    story.append(P("OV-6c \u2014 Event Trace (nominal science pass)", subsec_s))
    ov6c = [
        [P("Event", bold_s), P("Boundary", bold_s),
         P("Effect", bold_s), P("Artifact", bold_s)],
        [P("Umbra entry", body_s),
         P("g\u2081 crosses 0 (+)", mono_s),
         P("eclipse_valid = True", small_s), P("OV-6c", tag_teal)],
        [P("Thermal check", body_s),
         P("g\u2084 evaluated", mono_s),
         P("thermal_valid set", small_s), P("OV-6a", tag_blue)],
        [P("Target in FOV", body_s),
         P("g\u2082 crosses 0 (+)", mono_s),
         P("target_visible = True", small_s), P("OV-6c", tag_teal)],
        [P("x(t) \u2208 F", body_s),
         P("All g\u1D62 \u2265 0", mono_s),
         P("\u2192 Observe mode", small_s), P("OV-6b", tag_green)],
        [P("N exposures", body_s),
         P("MODE == OBSERVE", mono_s),
         P("Science collected", small_s), P("OV-5b", tag_gold)],
        [P("Thermal limit", body_s),
         P("g\u2084 crosses 0 (\u2212)", mono_s),
         P("\u2192 Thermal Safe", small_s), P("OV-6b", tag_red)],
        [P("Target exits FOV", body_s),
         P("g\u2082 crosses 0 (\u2212)", mono_s),
         P("target_visible = False", small_s), P("OV-6c", tag_teal)],
        [P("Umbra exit", body_s),
         P("g\u2081 crosses 0 (\u2212)", mono_s),
         P("\u2192 Standby", small_s), P("OV-6b", tag_blue)],
    ]
    story.append(KeepTogether([
        ruled_table(ov6c, [1.05*inch, 1.65*inch, 1.85*inch, 1.15*inch],
                    alt=[WHITE, LTBLUE]),
        Spacer(1, 6),
    ]))

    story.append(concept_box(
        "Every operational rule, event, and state transition is derived "
        "from a physical constraint inequality.",
        callout_s))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    # 8. EXPOSURE OPTIMIZATION AND SLIDING WINDOW
    # ══════════════════════════════════════════════════════════════════════════
    story.append(P("8.  Exposure Optimization and the Sliding Window", sec_s))

    story.append(P(
        "Once the feasible region F is known for a given pass, the optimization "
        "selects N and t_exp to maximize science return subject to all constraints.",
        body_s))

    story.append(P("SNR Structure", subsec_s))
    story.append(P(
        "Let S denote source photon count per frame and B denote background "
        "photon count per frame. "
        "Both are proportional to exposure time.",
        body_s))

    story.append(P(
        "S  =  rate_target \u00B7 t_exp", mono_c))
    story.append(P(
        "B  =  rate_background(F_earth) \u00B7 t_exp", mono_c))
    story.append(Spacer(1, 2))
    story.append(P(
        "SNR_single  =  S / sqrt(S + B)", mono_c))
    story.append(P(
        "SNR_total   =  sqrt(N) \u00B7 SNR_single", mono_c))
    story.append(Spacer(1, 4))

    story.append(P(
        "The background rate depends on the Earth view factor F_earth. "
        "Because F_earth is geometric, SNR is coupled to orbit geometry.",
        body_s))

    story.append(concept_box(
        "SNR is coupled to orbit geometry through the background term.",
        callout_s))
    story.append(Spacer(1, 6))

    story.append(P("Optimization Problem", subsec_s))
    opt_data = [
        [P("Element", bold_s), P("Description", bold_s)],
        [P("Variables", tag_blue),
         P("N (integer \u2265 1),  t_exp (seconds)", mono_s)],
        [P("Objective", tag_green),
         P("Maximize  SNR_total = sqrt(N) \u00B7 SNR_single(t_exp)", mono_s)],
        [P("Timing", tag_teal),
         P("N\u00B7(t_exp + t_read) \u2264 \u0394t_obs", mono_s)],
        [P("Signal floor", tag_teal),
         P("t_exp \u2265 N_ph_req / rate_target", mono_s)],
        [P("Thermal", tag_red),
         P("T_det(t) \u2264 T_max  for all t in window", mono_s)],
        [P("Power", tag_gold),
         P("N \u00B7 E_exp \u2264 E_available", mono_s)],
        [P("Data", tag_gold),
         P("N \u00B7 d_frame \u2264 buffer_capacity", mono_s)],
    ]
    story.append(ruled_table(opt_data, [1.0*inch, 5.3*inch]))
    story.append(Spacer(1, 8))

    story.append(P("The Sliding Window", subsec_s))
    story.append(P(
        "The feasible region F is not static. "
        "As the orbit precesses and the Sun angle changes, "
        "the constraint boundaries shift from pass to pass.",
        body_s))

    slide_data = [
        [P("Quantity", bold_s), P("Varies With", bold_s),
         P("Effect on F", bold_s)],
        [P("\u0394t_eclipse", body_s),
         P("Sun\u2013orbit angle, season", small_s),
         P("Bounds temporal extent of F", small_s)],
        [P("F_earth(t)", body_s),
         P("Altitude, attitude, orbit position", small_s),
         P("Shifts thermal boundary", small_s)],
        [P("T_det(t)", body_s),
         P("F_earth profile, detector power", small_s),
         P("Defines thermal sub-window", small_s)],
        [P("\u0394t_target", body_s),
         P("Target direction, FOV, orbit", small_s),
         P("F = eclipse \u2229 target \u2229 thermal", small_s)],
        [P("E_avail", body_s),
         P("Eclipse fraction, \u03B2 angle", small_s),
         P("Power constraint tightens seasonally", small_s)],
    ]
    story.append(ruled_table(slide_data, [1.3*inch, 2.2*inch, 2.8*inch],
                             alt=[WHITE, LTBLUE]))
    story.append(Spacer(1, 8))

    story.append(P("Per-Pass Algorithm", subsec_s))
    story.append(P(
        "For each orbital pass, the following procedure evaluates the feasible "
        "region and solves the optimization.",
        body_s))

    story.append(flow_diagram([
        ("1", "Eclipse\nWindow", "shadow cone\ngeometry"),
        ("2", "Thermal\nEnvelope", "F_earth \u2192\nT_det(t)"),
        ("3", "Constraint\nIntersection", "eclipse \u2229\nthermal \u2229 target"),
        ("4", "Optimize\nSNR", "max SNR\n(N, t_exp)"),
    ], base))
    story.append(Spacer(1, 6))

    alg_data = [
        [P("Step", bold_s), P("Operation", bold_s)],
        [P("1", mono_s), P("Compute eclipse window [T_in, T_out] from shadow cone geometry", body_s)],
        [P("2", mono_s), P("Evaluate F_earth(t) over eclipse window (view factor geometry)", body_s)],
        [P("3", mono_s), P("Integrate T_det(t) from F_earth; find thermal-feasible sub-window", body_s)],
        [P("4", mono_s), P("Evaluate target visibility within eclipse window", body_s)],
        [P("5", mono_s), P("Observation window = eclipse \u2229 thermal_ok \u2229 target_visible", body_s)],
        [P("6", mono_s), P("Solve: maximize SNR_total subject to all constraints", body_s)],
        [P("7", mono_s), P("Output: N*, t_exp*, [T_obs_start, T_obs_end]", body_s)],
    ]
    story.append(ruled_table(alg_data, [0.5*inch, 5.8*inch], header_bg=TEAL))
    story.append(Spacer(1, 6))

    story.append(P(
        "Over the mission life this produces a family of solutions "
        "N(pass), t_exp(pass) that track the evolving constraint geometry.",
        body_s))

    # ══════════════════════════════════════════════════════════════════════════
    # CONTRIBUTIONS (unnumbered closing)
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=RULE, spaceAfter=8))
    story.append(P("Contributions", sec_s))

    contribs = [
        ("Contribution 1",
         "Mission operations formulated as a continuous constraint satisfaction "
         "problem. Science activity is enabled only while the system state lies "
         "inside a dynamically evolving feasible region."),
        ("Contribution 2",
         "Physical constraints expressed as inequalities g\u1D62(x) \u2265 0 "
         "that define a feasible operational region in system state space. "
         "The feasible region is the intersection of all constraint half-spaces."),
        ("Contribution 3",
         "Geometric primitives provide efficient constraint evaluation. "
         "The constraint algebra generates DoDAF operational rules (OV-6a), "
         "events (OV-6c), and state transitions (OV-6b) mechanically."),
    ]
    for label, text in contribs:
        row = Table(
            [[P(label, S(f"_c{label}", base, fontSize=9, textColor=WHITE,
                          fontName=BOLD_FONT, alignment=TA_CENTER)),
              P(text, body_s)]],
            colWidths=[1.0*inch, 5.3*inch])
        row.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), BLUE),
            ("BACKGROUND",    (1, 0), (1, 0), LTBLUE),
            ("BOX",           (0, 0), (-1, -1), 0.5, BLUE),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ]))
        story.append(KeepTogether([row, Spacer(1, 4)]))

    story.append(Spacer(1, 8))
    story.append(concept_box(
        "Operational philosophy \u2192  constraint manifold \u2192  "
        "geometric evaluation \u2192  DoDAF rule generation \u2192  "
        "exposure optimization.",
        callout_s))
    story.append(Spacer(1, 8))

    # ── Footer ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE, spaceAfter=4))
    story.append(P(
        "DarkNESS 6U CubeSat  \u00B7  DarkNESS Architecture for Operations (DAO)  "
        "\u00B7  Constraint-Driven Operations  \u00B7  DoDAF 2.0 Operational Viewpoint",
        cap_s))

    doc.build(story)
    print(f"Written: {out}")


if __name__ == "__main__":
    build()
