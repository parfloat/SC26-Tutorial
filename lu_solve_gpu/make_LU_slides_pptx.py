#!/usr/bin/env python3
"""Build LU_Slides.pptx (native, editable shapes) for the LU-solver tutorial.

Run:  python3 make_LU_slides_pptx.py        (needs python-pptx)
Each slide is a function below; add the next slide by adding a function and
appending it to SLIDES.
"""
import re
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

DARK = RGBColor(0x23, 0x37, 0x3B)
ACCENT = RGBColor(0xEB, 0x81, 0x1B)
INK = RGBColor(0x22, 0x22, 0x22)
BLUE = RGBColor(0xE6, 0xE9, 0xFA)
GRAY = RGBColor(0xEE, 0xEE, 0xEE)
GREEN = RGBColor(0xD9, 0xF2, 0xD9)
FONT = "Calibri"
MONO = "Consolas"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_runs(par, text, size, color=INK, font=FONT, bold=False, italic=False):
    """Tiny markup: *italic* (may contain sub/sup), x_{sub}, x^{sup}."""
    pos = 0
    for m in re.finditer(r"\*([^*]+)\*|_\{([^}]*)\}|\^\{([^}]*)\}", text):
        if m.start() > pos:
            _run(par, text[pos:m.start()], size, color, font, bold, italic=italic)
        if m.group(1) is not None:
            add_runs(par, m.group(1), size, color, font, bold, italic=True)
        elif m.group(2) is not None:
            _run(par, m.group(2), size, color, font, bold, baseline=-25000, italic=italic)
        else:
            _run(par, m.group(3), size, color, font, bold, baseline=30000, italic=italic)
        pos = m.end()
    if pos < len(text):
        _run(par, text[pos:], size, color, font, bold, italic=italic)


def _run(par, text, size, color, font, bold, baseline=None, italic=False):
    r = par.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.name = font
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    if baseline is not None:
        r._r.get_or_add_rPr().set("baseline", str(baseline))


def textbox(slide, x, y, w, h, lines, size=18, color=INK, font=FONT, align=PP_ALIGN.LEFT,
            bold=False, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        add_runs(p, line, size, color, font, bold)
    return tb


def title_bar(slide, text, mono=False):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.95))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK
    bar.line.fill.background()
    tf = bar.text_frame
    tf.margin_left = Inches(0.5)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    add_runs(p, text, 28, RGBColor(0xFF, 0xFF, 0xFF), MONO if mono else FONT)


def box(slide, x, y, w, h, lines, fill=BLUE, size=11, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        mono_first=False, line_color=INK):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = line_color
    s.line.width = Pt(1.25)
    s.shadow.inherit = False
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        first_mono = mono_first and i == 0
        add_runs(p, line, size, INK, MONO if first_mono else FONT, bold=first_mono)
    return s


def arrow(slide, a, b, side_a="r", side_b="l"):
    """Connector with arrowhead from shape a to shape b (connection-site based)."""
    sites = {"t": 0, "l": 1, "b": 2, "r": 3}
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, 0, 0, 0, 0)
    c.begin_connect(a, sites[side_a])
    c.end_connect(b, sites[side_b])
    c.line.color.rgb = INK
    c.line.width = Pt(1.75)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    return c


# --------------------------------------------------------------------------
# Slide 1: cover
# --------------------------------------------------------------------------
def slide_cover():
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = DARK
    bg.line.fill.background()
    textbox(s, 0.9, 2.3, 11.5, 1.2, ["LU Solver on the GPU"], size=48,
            color=RGBColor(0xFF, 0xFF, 0xFF), bold=True)
    textbox(s, 0.9, 3.5, 11.5, 0.8,
            ["From FP64 kernels to tensor cores, watched by NixNan"], size=26,
            color=RGBColor(0xDD, 0xDD, 0xDD))
    rule = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(4.4), Inches(11.5), Inches(0.05))
    rule.fill.solid()
    rule.fill.fore_color.rgb = ACCENT
    rule.line.fill.background()
    textbox(s, 0.9, 4.7, 11.5, 1.0, ["Ganesh Gopalakrishnan", "SC26 tutorial"], size=18,
            color=RGBColor(0xDD, 0xDD, 0xDD))


# --------------------------------------------------------------------------
# Slide 2: algorithm + workflow diagram
# --------------------------------------------------------------------------
def slide_algorithm():
    s = prs.slides.add_slide(BLANK)
    title_bar(s, "Algorithm: LU with partial pivoting, solve *Ax = b*")
    textbox(s, 0.5, 1.05, 12.3, 1.2,
            ["*PA = LU*          pivot: *p = argmax_{i≥k} |U_{ik}|*          multiplier: *L_{jk} = U_{jk} / U_{kk}*",
             "eliminate: *U_{jl} −= L_{jk} U_{kl}*  (*j>k, l≥k*)          forward: *Ly = Pb*          backward: *Ux = y*"],
            size=17)

    w, h, gap = 2.1, 0.95, 0.42
    y1, y2, y3 = 2.85, 4.35, 5.85
    # Row 1
    A = box(s, 0.5, y1 - 0.65, 1.0, 0.8, ["*A*", "*m×m*"], GRAY, shape=MSO_SHAPE.RECTANGLE)
    b = box(s, 0.5, y1 + 0.45, 1.0, 0.8, ["*b*", "*m*"], GRAY, shape=MSO_SHAPE.RECTANGLE)
    init = box(s, 3.4, y1, 2.4, h, ["init_identity_kernel", "×2:  L=I, P=I"], BLUE, mono_first=True)
    st0 = box(s, 6.2, y1, 1.7, h, ["U₀ = A", "L = I,  P = I"], GRAY, shape=MSO_SHAPE.RECTANGLE)
    piv = box(s, 8.3, y1, 2.6, h, ["find_pivot_kernel", "reduction over column *k*"], BLUE, mono_first=True)
    cA = box(s, 11.4, y1 + 0.2, 0.55, 0.55, ["A"], GRAY, shape=MSO_SHAPE.OVAL)
    arrow(s, A, init)
    arrow(s, b, init)
    arrow(s, init, st0)
    arrow(s, st0, piv)
    arrow(s, piv, cA)
    # Row 2
    cA2 = box(s, 0.5, y2 + 0.2, 0.55, 0.55, ["A"], GRAY, shape=MSO_SHAPE.OVAL)
    swap = box(s, 1.6, y2, 2.9, h, ["swap_rows_kernel", "row swap in *U, L, P*"], BLUE, mono_first=True)
    mult = box(s, 4.9, y2, 3.1, h, ["compute_multipliers_kernel", "*L_{jk} = U_{jk} / U_{kk}*"], BLUE, mono_first=True)
    elim = box(s, 8.4, y2, 3.3, h, ["eliminate_kernel", "*U_{jl} −= L_{jk} U_{kl}*  (ε-clamp)"], BLUE, mono_first=True)
    cB = box(s, 12.1, y2 + 0.2, 0.55, 0.55, ["B"], GRAY, shape=MSO_SHAPE.OVAL)
    arrow(s, cA2, swap)
    arrow(s, swap, mult)
    arrow(s, mult, elim)
    arrow(s, elim, cB)
    # Row 3
    cB2 = box(s, 0.5, y3 + 0.2, 0.55, 0.55, ["B"], GRAY, shape=MSO_SHAPE.OVAL)
    perm = box(s, 1.6, y3, 2.9, h, ["apply_permutation_kernel", "*b′ = Pb*"], BLUE, mono_first=True)
    fwd = box(s, 4.9, y3, 3.2, h, ["forward_substitution_kernel", "solve *Ly = b′*"], BLUE, mono_first=True)
    bwd = box(s, 8.5, y3, 3.2, h, ["backward_substitution_kernel", "solve *Ux = y*"], BLUE, mono_first=True)
    x = box(s, 12.1, y3, 0.95, h, ["*x*", "*m*"], GREEN, shape=MSO_SHAPE.RECTANGLE)
    arrow(s, cB2, perm)
    arrow(s, perm, fwd)
    arrow(s, fwd, bwd)
    arrow(s, bwd, x)


# --------------------------------------------------------------------------
# Slide 3: NixNan environment variables
# --------------------------------------------------------------------------
LEFT = [("INSTR_BEGIN / END", "instr. index window"), ("TOOL_VERBOSE", "setup diagnostics"),
        ("ENABLE_FUN_DETAIL", "name on every rerun"), ("PRINT_ILL_INSTR", "print culprit SASS"),
        ("SAMPLING", "every N-th launch"), ("INSTRUMENT_EXCEPTIONS", "master switch"),
        ("INSTR_MEM", "also check STG (excn → mem)"), ("HISTOGRAM", "exponent histogram"),
        ("BIN_SPEC_FILE", "ranges to watch")]
RIGHT = [("NIXNAN_TIMEOUT", "self-terminate, secs"), ("LOGFILE", "redirect output"),
         ("LINE_INFO", "source file:line (YMMV)"), ("LOG_KERNELS", "log launches only"),
         ("MAX_ERRORS", "stop after N reports"), ("TIME_KERNELS", "per-kernel time"),
         ("FUNCTION_WHITELIST", "only these kernels"), ("FUNCTION_BLACKLIST", "all but these")]


def two_col_table(slide, x, y, w, rows, size=15):
    shp = slide.shapes.add_table(len(rows), 2, Inches(x), Inches(y), Inches(w), Inches(0.42 * len(rows)))
    tbl = shp.table
    # plain look: remove banding/style fills
    tblPr = shp._element.graphic.graphicData.tbl.tblPr
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    tbl.columns[0].width = Inches(w * 0.42)
    tbl.columns[1].width = Inches(w * 0.58)
    for r, (v, e) in enumerate(rows):
        for c, (txt, font) in enumerate(((v, MONO), (e, FONT))):
            cell = tbl.cell(r, c)
            cell.fill.background()
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            p = cell.text_frame.paragraphs[0]
            add_runs(p, txt, size if c else size - 1, INK, font, bold=(c == 0))


def slide_envvars():
    s = prs.slides.add_slide(BLANK)
    title_bar(s, "NixNan knobs: environment variables")
    two_col_table(s, 0.5, 1.3, 6.4, LEFT)
    two_col_table(s, 7.1, 1.3, 5.8, RIGHT)
    textbox(s, 0.5, 6.0, 12.3, 1.0,
            ["For example, ksel-autonixnan scripts (using the above env vars) the collection of kernel names "
             "and execution times, so users can then white- or black-list exactly what to observe in detail."],
            size=16)


# --------------------------------------------------------------------------
# Slide 4: TIME_KERNELS first-launch cost
# --------------------------------------------------------------------------
TK_ROWS = [("init_identity_kernel", "2", "312,451", "21", "≈ 14,900×"),
           ("find_pivot_kernel", "19", "298,100", "20", "≈ 14,900×"),
           ("swap_rows_kernel", "19", "290,205", "25", "≈ 11,600×"),
           ("compute_multipliers_kernel", "19", "295,432", "20", "≈ 14,800×"),
           ("eliminate_kernel", "19", "278,808", "20", "≈ 13,900×"),
           ("apply_permutation_kernel", "1", "300,320", "—", "—"),
           ("forward_substitution_kernel", "1", "326,736", "—", "—"),
           ("backward_substitution_kernel", "1", "324,597", "—", "—")]


def slide_time_kernels():
    s = prs.slides.add_slide(BLANK)
    title_bar(s, "TIME_KERNELS: the first-launch cost", mono=False)
    hdr = ["Kernel", "Launches", "First launch (µs)", "Median later (µs)", "Ratio"]
    shp = s.shapes.add_table(len(TK_ROWS) + 1, 5, Inches(0.7), Inches(1.2), Inches(11.9), Inches(3.4))
    tbl = shp.table
    for i, wd in enumerate((4.2, 1.4, 2.3, 2.4, 1.6)):
        tbl.columns[i].width = Inches(wd)
    for c, t in enumerate(hdr):
        cell = tbl.cell(0, c)
        add_runs(cell.text_frame.paragraphs[0], t, 14, RGBColor(0xFF, 0xFF, 0xFF), FONT, bold=True)
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK
    for r, row in enumerate(TK_ROWS, start=1):
        for c, t in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = GRAY if r % 2 else RGBColor(0xFF, 0xFF, 0xFF)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            add_runs(p, t, 13, INK, MONO if c == 0 else FONT)
    bullets = [
        "First launch of every kernel: ≈280–330 ms. Later launches: ≈20–25 µs, four to five orders of magnitude less.",
        "The timer starts before NVBit rewrites the kernel’s SASS (once per function) and stops after, so the "
        "first number is instrumentation cost, not kernel time.",
        "Kernels launched once (last three rows) only ever show this one-time cost.",
        "Library-heavy binaries pay far more: the cuSOLVER-based pytorch_slogdet run was killed at 120 s "
        "during instrumentation.",
    ]
    tb = s.shapes.add_textbox(Inches(0.7), Inches(4.95), Inches(11.9), Inches(2.3))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, t in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(4)
        add_runs(p, "•  " + t, 16)


SLIDES = [slide_cover, slide_algorithm, slide_envvars, slide_time_kernels]

if __name__ == "__main__":
    for f in SLIDES:
        f()
    prs.save("LU_Slides.pptx")
    print("wrote LU_Slides.pptx with", len(prs.slides), "slides")
