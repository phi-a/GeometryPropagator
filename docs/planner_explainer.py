"""Generate a short science-facing explainer for the imaging planner.

Run from the repository root:
    python docs/planner_explainer.py

Output:
    docs/panner_explainer.pdf
"""

from __future__ import annotations

from io import BytesIO
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
PLANNER_DIR = ROOT / "planner"
if str(PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(PLANNER_DIR))

from constants import (  # type: ignore  # noqa: E402
    DEFAULT_FOV_HALF_DEG,
    DEFAULT_N_MAX,
    DEFAULT_N_PIXELS,
    DEFAULT_READ_RATE,
    DEFAULT_STARTUP_S,
    SGR_A_DEC_DEG,
    SGR_A_RA_DEG,
)


NAVY = colors.HexColor("#102033")
BLUE = colors.HexColor("#315a7d")
TEXT = colors.HexColor("#222222")
MUTED = colors.HexColor("#666666")
RULE = colors.HexColor("#d9dee3")
PANEL = colors.HexColor("#f5f7f9")
ACCENT = colors.HexColor("#e8eef4")


def style(name: str, parent: ParagraphStyle, **kwargs) -> ParagraphStyle:
    return ParagraphStyle(name, parent=parent, **kwargs)


def paragraph_box(text: str, body_style: ParagraphStyle, width: float) -> Table:
    box = Table([[Paragraph(text, body_style)]], colWidths=[width])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PANEL),
                ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    return box


def simple_table(data: list[list[object]], widths: list[float], header_fill=ACCENT) -> Table:
    table = Table(data, colWidths=widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_fill),
                ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def numbered_flow(steps: list[tuple[str, str]], width: float) -> Table:
    rows = [[Paragraph("<b>Step</b>", BODY), Paragraph("<b>What happens</b>", BODY)]]
    for idx, text in steps:
        rows.append(
            [
                Paragraph(f"<b>{idx}</b>", BODY),
                Paragraph(text, BODY),
            ]
        )
    return simple_table(rows, [0.75 * inch, width - 0.75 * inch])


def add_page_number(canvas, doc) -> None:
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.45 * inch, f"Page {doc.page}")


STYLES = getSampleStyleSheet()
BASE = STYLES["BodyText"]
TITLE = style(
    "Title",
    BASE,
    fontName="Helvetica-Bold",
    fontSize=19,
    leading=22,
    textColor=NAVY,
    spaceAfter=2,
)
SUBTITLE = style(
    "Subtitle",
    BASE,
    fontName="Helvetica",
    fontSize=9.5,
    leading=13,
    textColor=MUTED,
    spaceAfter=8,
)
SECTION = style(
    "Section",
    BASE,
    fontName="Helvetica-Bold",
    fontSize=11.5,
    leading=14,
    textColor=BLUE,
    spaceBefore=9,
    spaceAfter=4,
)
BODY = style(
    "Body",
    BASE,
    fontName="Helvetica",
    fontSize=9.0,
    leading=12,
    textColor=TEXT,
    spaceAfter=4,
)
SMALL = style(
    "Small",
    BASE,
    fontName="Helvetica",
    fontSize=8.0,
    leading=10,
    textColor=MUTED,
    spaceAfter=4,
)
MONO = style(
    "Mono",
    BASE,
    fontName="Courier",
    fontSize=8.0,
    leading=10,
    textColor=NAVY,
    spaceAfter=3,
)


def find_result_image() -> Path | None:
    candidates = [
        ROOT / "output" / "sv7_matrix.png",
        ROOT / "outputs" / "sv7_matrix.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def render_optimization_math(max_width: float) -> Image | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig = plt.figure(figsize=(6.7, 1.95), dpi=200)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    equations = [
        r"$t_{\mathrm{startup}} + B + C(N)\leq T_{\mathrm{open}}$",
        r"$C(N)=\frac{N_{\mathrm{pix}}}{r_{\mathrm{read}}}\,N$",
        r"$B(N)=T_{\mathrm{open}}-t_{\mathrm{startup}}-C(N)$",
        r"$\mathrm{Choose}\ N\in\{1,\ldots,N_{\max}\}\ \mathrm{with}\ B(N)>0$",
        r"$\mathrm{to\ make}\ B(N)N\ \mathrm{as\ large\ as\ possible},\qquad B^\ast=B(N^\ast)$",
    ]
    y_positions = [0.86, 0.64, 0.42, 0.20, 0.03]
    for y, equation in zip(y_positions, equations):
        ax.text(0.5, y, equation, fontsize=14, color="#102033", ha="center", va="center")

    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)

    buffer.seek(0)
    img_reader = ImageReader(buffer)
    img_width_px, img_height_px = img_reader.getSize()
    scale = min(max_width / img_width_px, 1.0)
    buffer.seek(0)

    img = Image(
        buffer,
        width=img_width_px * scale,
        height=img_height_px * scale,
    )
    img.hAlign = "CENTER"
    return img


def build() -> Path:
    docs_dir = Path(__file__).resolve().parent
    out_path = docs_dir / "panner_explainer.pdf"

    readout_per_sample = DEFAULT_N_PIXELS / DEFAULT_READ_RATE

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.55 * inch,
    )

    story = []
    usable_width = letter[0] - doc.leftMargin - doc.rightMargin

    story.append(Paragraph("Planner Explainer", TITLE))
    story.append(
        Paragraph(
            "A short science-facing description of what the DarkNESS planner computes, "
            "what it needs, and how it chooses imaging settings.",
            SUBTITLE,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.2, color=BLUE, spaceAfter=10))
    story.append(
        paragraph_box(
            "In one sentence: the planner asks whether a target can be observed in the "
            "available eclipse geometry at a given time, then picks the exposure time and "
            "number of Skipper samples that fit inside that window.",
            BODY,
            usable_width,
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("1. What Problem It Solves", SECTION))
    story.append(
        Paragraph(
            "The code is a pass-level imaging planner, not a full mission scheduler. "
            "Its job is to determine whether science collection is geometrically possible "
            "for a given orbit state and target, and if it is, to recommend detector settings "
            "for that opportunity.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "For DarkNESS-like observing, science is only possible when several conditions hold "
            "at the same time: the spacecraft is in eclipse, the target is above the Earth limb, "
            "the instrument field of view is clear, and there is enough time to start the payload, "
            "integrate, and read out the detector before the window closes.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "The planner converts that physical situation into a single number called the "
            "<b>open-sky budget</b>, written here as <b>T_open</b>. That is the usable science "
            "time available for one orbit at one query time.",
            BODY,
        )
    )

    story.append(Paragraph("2. Core Logic", SECTION))
    story.append(
        numbered_flow(
            [
                (
                    "1",
                    "Start from orbit information. In the date-driven mode this is "
                    "semi-major axis, eccentricity, inclination, RAAN at epoch, and epoch. "
                    "In the flight mode it is a TLE plus a timestamp.",
                ),
                (
                    "2",
                    "Propagate the RAAN to the query time using a J2 secular drift model. "
                    "This sets the orbit plane orientation relative to the Sun and target.",
                ),
                (
                    "3",
                    "Compute eclipse geometry from the solar beta angle. This gives the "
                    "umbra duration and the orbital arc where observing could happen at all.",
                ),
                (
                    "4",
                    "Compute target visibility from the target beta angle and field-of-view "
                    "half-angle. This gives the orbital arc where the target and the full FOV "
                    "clear the Earth limb.",
                ),
                (
                    "5",
                    "Intersect those two arcs. The overlap is T_open. If there is no overlap, "
                    "the planner returns no science opportunity for that query.",
                ),
            ],
            usable_width,
        )
    )

    story.append(PageBreak())

    story.append(Paragraph("3. Inputs", SECTION))
    inputs = [
        [Paragraph("Input group", SMALL), Paragraph("What the planner uses it for", SMALL)],
        [
            Paragraph(
                "<b>Orbit state</b><br/>"
                "Either (a, e, i, RAAN at epoch, epoch) or a TLE plus a timestamp.",
                SMALL,
            ),
            Paragraph(
                "Sets the orbit plane and how it has precessed at the requested time.",
                SMALL,
            ),
        ],
        [
            Paragraph(
                "<b>Target</b><br/>"
                "Right ascension, declination, and a name label.",
                SMALL,
            ),
            Paragraph(
                "Defines the inertial science direction that must clear the Earth limb.",
                SMALL,
            ),
        ],
        [
            Paragraph(
                "<b>Field of view</b><br/>"
                "Instrument half-angle.",
                SMALL,
            ),
            Paragraph(
                "Expands the visibility requirement from target-center visibility to full-frame clearance.",
                SMALL,
            ),
        ],
        [
            Paragraph(
                "<b>Detector timing</b><br/>"
                "Startup time, number of pixels, read rate, and maximum allowed sample count.",
                SMALL,
            ),
            Paragraph(
                "Determines how much of T_open can be spent on readout versus exposure.",
                SMALL,
            ),
        ],
    ]
    story.append(simple_table(inputs, [2.2 * inch, 4.45 * inch]))
    story.append(Spacer(1, 0.1 * inch))

    defaults = [
        [Paragraph("Default DarkNESS-like settings in the current code", BODY)],
        [
            Paragraph(
                f"Startup: <b>{DEFAULT_STARTUP_S:.0f} s</b>; "
                f"pixels per frame: <b>{DEFAULT_N_PIXELS:,}</b>; "
                f"read rate: <b>{DEFAULT_READ_RATE:,} px/s</b>; "
                f"readout per sample: <b>{readout_per_sample:.1f} s</b>.<br/>"
                f"Search limit: <b>{DEFAULT_N_MAX}</b> samples; "
                f"default FOV half-angle: <b>{DEFAULT_FOV_HALF_DEG:.1f} deg</b>; "
                f"default demo target: <b>Sgr A*</b> "
                f"({SGR_A_RA_DEG:.4f} deg, {SGR_A_DEC_DEG:.4f} deg).",
                SMALL,
            )
        ],
    ]
    default_box = Table(defaults, colWidths=[usable_width])
    default_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("BACKGROUND", (0, 1), (-1, -1), PANEL),
                ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(default_box)
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("4. Outputs", SECTION))
    outputs = [
        [Paragraph("Output", SMALL), Paragraph("Meaning for science use", SMALL)],
        [
            Paragraph("<b>open_sky_s</b>", SMALL),
            Paragraph("The usable observation window after eclipse and Earth-limb/FOV limits are combined.", SMALL),
        ],
        [
            Paragraph("<b>n_samples</b>", SMALL),
            Paragraph("Recommended number of non-destructive Skipper samples for that window.", SMALL),
        ],
        [
            Paragraph("<b>exposure_s</b>", SMALL),
            Paragraph("Recommended integration time that still leaves enough room for readout.", SMALL),
        ],
        [
            Paragraph("<b>status</b>", SMALL),
            Paragraph(
                "Operational interpretation of the result: <b>ok</b>, <b>no_eclipse</b>, "
                "<b>target_obstructed</b>, or <b>insufficient_budget</b>.",
                SMALL,
            ),
        ],
    ]
    story.append(simple_table(outputs, [1.55 * inch, 5.1 * inch]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        Paragraph(
            "For science planning, <b>open_sky_s</b>, <b>n_samples</b>, <b>exposure_s</b>, and "
            "<b>status</b> are usually enough to decide whether a query time is scientifically usable.",
            BODY,
        )
    )

    story.append(PageBreak())

    story.append(Paragraph("5. How the Optimization Is Done", SECTION))
    story.append(
        Paragraph(
            "The optimization in the current implementation is intentionally small and transparent. "
            "It does not use gradients, nonlinear solvers, or machine learning. It simply searches "
            "through integer sample counts and keeps the best feasible choice.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "For this mission, the geometry stage has already done the visibility work. It returns "
            "<b>T_open</b>, the amount of time in the current orbit for which the target is usable. "
            "The optimization stage only has to decide how to spend that window across payload startup, "
            "science exposure, and detector readout.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "That leads directly to the time-budget constraint below. The first line says the full activity "
            "must fit inside the open-sky window. The second line uses the Skipper timing model: each "
            "additional sample requires one full-frame read, so readout grows linearly with sample count. "
            "Once <b>N</b> is fixed, the best exposure is simply the largest feasible one, which gives the "
            "third line. At that point the original two-variable problem collapses to a one-dimensional "
            "search over integer <b>N</b> values.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "In notation, the current code is solving:",
            BODY,
        )
    )
    math_block = render_optimization_math(usable_width * 0.88)
    if math_block is not None:
        story.append(math_block)
        story.append(
            Paragraph(
                "Here <b>C(N)</b> is total readout time, <b>B(N)</b> is the remaining feasible exposure after "
                "startup and readout are accounted for, and the final two lines restate the search in plain "
                "language rather than formal optimization notation.",
                SMALL,
            )
        )
    story.append(Spacer(1, 0.05 * inch))
    story.append(
        Paragraph(
            "Operationally, the code loops from <b>N = 1</b> to <b>N_max</b>, computes the corresponding "
            "feasible exposure <b>B(N)</b>, discards any case with non-positive exposure, and keeps the "
            "feasible pair with the largest <b>B(N) x N</b>. Every extra sample can help the detector, "
            "but it also consumes one more full-frame read, so the optimum is the balance point inside the "
            "available window.",
            BODY,
        )
    )

    result_image = find_result_image()
    if result_image is not None:
        story.append(Paragraph("6. Example Result", SECTION))
        story.append(
            Paragraph(
                "The figure below is the current SV-7 systems measures matrix generated from the planner runs. "
                "It summarizes seasonal coverage, open-sky budget, sample-count range, and exposure range.",
                BODY,
            )
        )
        img_reader = ImageReader(str(result_image))
        img_width_px, img_height_px = img_reader.getSize()
        max_width = usable_width
        scale = max_width / img_width_px
        img = Image(
            str(result_image),
            width=img_width_px * scale,
            height=img_height_px * scale,
        )
        img.hAlign = "CENTER"
        story.append(img)

    story.append(Paragraph("7. Assumptions and Limits", SECTION))
    story.append(
        Paragraph(
            "The planner assumes a near-circular LEO orbit, J2-driven RAAN precession, a fixed inertial "
            "target direction over the planning interval, and a conic instrument field of view.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "It is primarily a geometry-and-timing planner. It decides whether observing fits in the "
            "window and what detector settings fit that window. It is not yet a full end-to-end optimizer "
            "over thermal margin, power, downlink, or long-horizon scheduling. The date-driven path can "
            "attach thermal view-factor information, but those thermal quantities are not part of the current "
            "decision objective.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "For CCD science discussions, the clean summary is: geometry defines the available window, and the "
            "optimizer picks the detector operating point that best fits inside that window.",
            SMALL,
        )
    )

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return out_path


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
