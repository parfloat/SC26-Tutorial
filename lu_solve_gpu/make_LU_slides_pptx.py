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
TITLE_FONT = "Arial"
TITLE_SIZE = 25
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
    """Standard slide title: black text on the white slide background."""
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.3), Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    add_runs(p, text, TITLE_SIZE, RGBColor(0, 0, 0), TITLE_FONT, bold=True)


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
    black = RGBColor(0, 0, 0)
    textbox(s, 0.9, 2.3, 11.5, 1.2, ["LU Solver on the GPU"], size=TITLE_SIZE, color=black, bold=True,
            font=TITLE_FONT)
    textbox(s, 0.9, 3.5, 11.5, 0.8,
            ["From FP64 kernels to tensor cores, watched by NixNan"], size=26, color=black)
    textbox(s, 0.9, 4.7, 11.5, 1.0, ["Ganesh Gopalakrishnan", "SC26 tutorial"], size=18, color=black)


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


# --------------------------------------------------------------------------
# Slide 5: Stage 1 results (report sec 4.2)
# --------------------------------------------------------------------------
STAGE1 = [("Residual ‖Ax − b‖₂ (uninstrumented)", "9.68e-16", "nan"),
          ("Binade-threshold reports", "99", "63"),
          ("FP64 NaN (repeats)", "0 (0)", "316 (2171)"),
          ("FP64 Infinity (repeats)", "0 (0)", "4 (0)"),
          ("FP64 Division-by-zero (repeats)", "0 (0)", "4 (4)")]


def slide_stage1_results():
    s = prs.slides.add_slide(BLANK)
    title_bar(s, "Stage 1 results: well-conditioned vs ill-conditioned input")
    shp = s.shapes.add_table(len(STAGE1) + 1, 3, Inches(1.2), Inches(1.4), Inches(10.9), Inches(3.3))
    tbl = shp.table
    for i, wd in enumerate((5.5, 2.6, 2.8)):
        tbl.columns[i].width = Inches(wd)
    for c, t in enumerate(["", "matrix.csv", "bad_matrix.csv"]):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK
        add_runs(cell.text_frame.paragraphs[0], t, 16, RGBColor(0xFF, 0xFF, 0xFF), MONO, bold=True)
        cell.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT if c else PP_ALIGN.LEFT
    for r, row in enumerate(STAGE1, start=1):
        for c, t in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = GRAY if r % 2 else RGBColor(0xFF, 0xFF, 0xFF)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if c else PP_ALIGN.LEFT
            add_runs(p, t, 16, INK, FONT)
    ok = box(s, 1.2, 5.2, 5.2, 1.3, ["matrix.csv", "clean solution, no exceptions"], GREEN, size=16)
    bad = box(s, 6.9, 5.2, 5.2, 1.3, ["bad_matrix.csv",
                                      "near-zero pivot → division by zero → NaN through the whole solve"],
              RGBColor(0xF9, 0xD9, 0xD9), size=16)


# --------------------------------------------------------------------------
# Slide 6: FP32 artifact of binary instrumentation (report sec 4.3)
# --------------------------------------------------------------------------
def slide_fp32_artifact():
    s = prs.slides.add_slide(BLANK)
    title_bar(s, "Binary instrumentation reveals exceptions you did not expect")
    bullets = [
        "The source computes only in double, yet bad_matrix.csv also reports 6 FP32 NaN detections (93 repeats).",
        "Cause: ptxas uses the idiom FFMA R8, RZ, R5, R3 (0 × R5 + R3 = R3) as a register move, shuffling the "
        "32-bit halves of a double. nixnan sees an f32 FFMA whose operand R5 decodes as NaN.",
        "Lesson: a SASS-level detector reports what the compiled code does, not what the source says. A nonzero "
        "count in a format the source never uses is not necessarily that format's arithmetic.",
    ]
    tb = s.shapes.add_textbox(Inches(0.7), Inches(1.3), Inches(11.9), Inches(3.0))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, t in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        add_runs(p, "•  " + t, 18)
    # small diagram: double -> two 32-bit halves -> FFMA move idiom
    d = box(s, 0.9, 5.0, 2.6, 1.1, ["double", "64 bits = hi | lo"], GRAY, size=15, shape=MSO_SHAPE.RECTANGLE)
    r5 = box(s, 4.3, 5.0, 2.4, 1.1, ["R5", "one 32-bit half"], GRAY, size=15, shape=MSO_SHAPE.RECTANGLE)
    ff = box(s, 7.5, 5.0, 3.2, 1.1, ["FFMA R8, RZ, R5, R3", "0 × R5 + R3: a register move"], BLUE,
             size=15, mono_first=True)
    fl = box(s, 11.3, 5.0, 1.5, 1.1, ["flagged", "f32 NaN"], RGBColor(0xF9, 0xD9, 0xD9), size=15,
             shape=MSO_SHAPE.RECTANGLE)
    arrow(s, d, r5)
    arrow(s, r5, ff)
    arrow(s, ff, fl)


# --------------------------------------------------------------------------
# Slide 7: TC adaptation = blocked LU with panel factorization (report 7.1 + 7.2)
# --------------------------------------------------------------------------
def slide_tc_panel():
    s = prs.slides.add_slide(BLANK)
    title_bar(s, "LU Adaptation to Tensor Cores Involves Panel Factorization")
    PANEL = RGBColor(0xB9, 0xC2, 0xF2)
    L21C = RGBColor(0xDD, 0xE2, 0xFA)
    U12C = RGBColor(0xFB, 0xD9, 0xA8)
    A22C = RGBColor(0xC9, 0xEB, 0xC9)
    TCRED = RGBColor(0xF6, 0xC3, 0xC3)
    x0, y0, side, pw = 0.8, 1.55, 3.8, 1.2      # square origin, size, panel width
    # partition of the matrix at block column kb
    box(s, x0, y0, pw, pw, ["panel", "L_{11}\\U_{11}"], PANEL, size=12, shape=MSO_SHAPE.RECTANGLE)
    box(s, x0 + pw, y0, side - pw, pw, ["U_{12}", "16 × (m−kb−16)"], U12C, size=12, shape=MSO_SHAPE.RECTANGLE)
    box(s, x0, y0 + pw, pw, side - pw, ["L_{21}"], L21C, size=12, shape=MSO_SHAPE.RECTANGLE)
    box(s, x0 + pw, y0 + pw, side - pw, side - pw,
        ["A_{22} ← A_{22} − L_{21} U_{12}", "tensor cores"], A22C, size=13, shape=MSO_SHAPE.RECTANGLE)
    textbox(s, x0, y0 - 0.4, 2.6, 0.35, ["16 columns at *kb*"], size=11)
    # numbered tags on regions
    for n, (tx, ty) in enumerate([(x0 + 0.05, y0 + 0.05), (x0 + pw + 0.05, y0 + 0.05),
                                   (x0 + pw + 0.05, y0 + pw + 0.05)], start=1):
        box(s, tx, ty, 0.28, 0.28, [str(n)], RGBColor(0xFF, 0xFF, 0xFF), size=11, shape=MSO_SHAPE.OVAL)
    # annotations (right)
    notes = [("1", "panel, FP64: pivot search, row swap, multipliers, elimination on the 16 panel columns"),
             ("2", "U_{12} = L_{11}^{-1} A_{12}, FP64 triangular solve"),
             ("3", "A_{22} −= L_{21} U_{12}: round to FP16/BF16, WMMA on tensor cores, subtract in FP64")]
    for i, (n, t) in enumerate(notes):
        yy = 1.6 + 1.1 * i
        box(s, 5.3, yy, 0.4, 0.4, [n], RGBColor(0xFF, 0xFF, 0xFF), size=13, shape=MSO_SHAPE.OVAL)
        textbox(s, 5.85, yy - 0.08, 7.0, 0.9, [t], size=16)
    textbox(s, 5.3, 4.95, 7.6, 0.4, ["Repeat for kb = 0, 16, 32, …; a last block narrower than 16 columns is panel-only."],
            size=13, color=RGBColor(0x55, 0x55, 0x55))
    # kernel pipeline for one block step
    names = [("find_pivot, swap_rows,\ncompute_multipliers,\neliminate_panel", "(×16 columns)", "FP64", BLUE),
             ("u12_trsm_kernel", "U_{12} = L_{11}^{-1} A_{12}", "FP64", BLUE),
             ("pack_tc_operands", "double → FP16/BF16, zero padding", "FP64 → FP16/BF16", BLUE),
             ("tc_gemm_tile", "mma_sync → HMMA.16816", "FP16/BF16 → FP32/16", TCRED),
             ("apply_update_kernel", "A_{22} −= P, ε-clamp", "FP64", BLUE)]
    bw, gap, by, bh = 2.3, 0.28, 5.55, 1.1
    bx = 0.45
    prev = None
    for k, (nm, sub, prec, col) in enumerate(names):
        multi = "\n" in nm
        lines = nm.split("\n") + [sub] if multi else [nm, sub]
        shp = box(s, bx + k * (bw + gap), by, bw, bh, lines, col, size=10 if multi else 11,
                  mono_first=not multi)
        textbox(s, bx + k * (bw + gap), by + bh + 0.02, bw, 0.3, [prec], size=10, align=PP_ALIGN.CENTER,
                color=RGBColor(0x55, 0x55, 0x55))
        if prev is not None:
            arrow(s, prev, shp)
        prev = shp


# --------------------------------------------------------------------------
# Slide 8: where exceptions arise (report 7.5) + numerical behaviour (report 8.1)
# --------------------------------------------------------------------------
TC_NUM = [("matrix.csv residual", "1.1e-4", "2.0e-4", "4.4e-4", "9.7e-16"),
          ("bad_matrix.csv", "NaN", "NaN", "NaN", "NaN"),
          ("rand:64:100000", "NaN", "NaN", "0.17", "–")]


def slide_tc_exceptions():
    s = prs.slides.add_slide(BLANK)
    title_bar(s, "Tensor-core LU: where exceptions arise, and what we saw")
    ORANGE = RGBColor(0xFB, 0xD9, 0xA8)
    seen = RGBColor(0xC9, 0xEB, 0xC9)
    y, h, w, gap = 1.35, 1.1, 2.15, 0.45
    xs = [0.6 + i * (w + gap) for i in range(5)]
    n1 = box(s, xs[0], y, w, h, ["L₂₁, U₁₂", "FP64"], GRAY, size=14, shape=MSO_SHAPE.RECTANGLE)
    n2 = box(s, xs[1], y, w, h, ["F2F.F16.F64", "double → half", "overflow → Inf"], ORANGE, size=13, mono_first=True)
    n3 = box(s, xs[2], y, w, h, ["HMMA", "inputs and result"], seen, size=14, mono_first=True)
    n4 = box(s, xs[3], y, w, h, ["STG", "tile store"], seen, size=14, mono_first=True)
    n5 = box(s, xs[4], y, w, h, ["DFMA / DADD", "update, FP64"], GRAY, size=14, mono_first=True)
    for a, b in ((n1, n2), (n2, n3), (n3, n4), (n4, n5)):
        arrow(s, a, b)
    textbox(s, xs[1] - 0.2, y + h + 0.05, w + 0.4, 0.5, ["no table entry: not checked"], size=11,
            align=PP_ALIGN.CENTER, color=RGBColor(0x99, 0x55, 0x00))
    textbox(s, xs[2] - 0.2, y + h + 0.05, w + 0.4, 0.5, ["Inf as input, NaN from 0 × ∞"], size=11,
            align=PP_ALIGN.CENTER, color=RGBColor(0x2E, 0x6B, 0x2E))
    textbox(s, 0.6, 3.15, 12.1, 0.8,
            ["Not all instructions are documented upfront, so we had to discover, ad hoc and whenever possible, "
             "which ones NixNan actually checks."], size=18)
    hdr = ["Input", "fp16acc32", "fp16acc16", "bf16acc32", "FP64 original"]
    shp = s.shapes.add_table(len(TC_NUM) + 1, 5, Inches(0.9), Inches(4.3), Inches(11.5), Inches(2.2))
    tbl = shp.table
    for i, wd in enumerate((3.5, 2.0, 2.0, 2.0, 2.0)):
        tbl.columns[i].width = Inches(wd)
    for c, t in enumerate(hdr):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK
        cell.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
        add_runs(cell.text_frame.paragraphs[0], t, 16, RGBColor(0xFF, 0xFF, 0xFF), FONT, bold=True)
    for r, row in enumerate(TC_NUM, start=1):
        for c, t in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = GRAY if r % 2 else RGBColor(0xFF, 0xFF, 0xFF)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            add_runs(p, t, 16, INK, MONO if c == 0 else FONT)


# --------------------------------------------------------------------------
# Slide 9: torch.where, forward fine, backward NaN (pytorch/pytorch#68425)
# --------------------------------------------------------------------------
def slide_where_backward():
    s = prs.slides.add_slide(BLANK)
    title_bar(s, "torch.where: a well-defined forward, a NaN gradient")
    RED = RGBColor(0xF9, 0xD9, 0xD9)
    textbox(s, 0.6, 1.0, 12.1, 0.4, ["b = torch.where(a <= 0, torch.exp(a), 1 + a),   a = 100 (float32, on GPU)"],
            size=15, font=MONO)
    # forward
    textbox(s, 0.6, 1.5, 2.0, 0.3, ["forward"], size=13, color=RGBColor(0x55, 0x55, 0x55), bold=True)
    fa = box(s, 0.6, 2.0, 1.5, 0.8, ["a", "100"], GRAY, size=14, shape=MSO_SHAPE.RECTANGLE)
    fe = box(s, 3.2, 1.7, 2.4, 0.8, ["exp(a)", "= inf  (overflows)"], RED, size=13)
    fp = box(s, 3.2, 2.6, 2.4, 0.8, ["1 + a", "= 101"], BLUE, size=13)
    fw = box(s, 6.4, 2.15, 2.4, 0.8, ["where(a ≤ 0, ·, ·)", "picks 1 + a"], BLUE, size=13)
    fb = box(s, 9.6, 2.15, 1.7, 0.8, ["b", "101  (fine)"], GREEN, size=13, shape=MSO_SHAPE.RECTANGLE)
    for p, q in ((fa, fe), (fa, fp), (fe, fw), (fp, fw), (fw, fb)):
        arrow(s, p, q)
    # backward
    textbox(s, 0.6, 3.75, 2.0, 0.3, ["backward"], size=13, color=RGBColor(0x55, 0x55, 0x55), bold=True)
    bg = box(s, 0.6, 4.35, 1.5, 0.8, ["grad", "1"], GRAY, size=14, shape=MSO_SHAPE.RECTANGLE)
    bw = box(s, 2.9, 4.35, 2.3, 0.8, ["where backward", "masks the grad"], BLUE, size=13)
    be = box(s, 6.2, 3.95, 2.6, 0.8, ["exp backward", "0 × exp(a) = 0 × inf = NaN"], RED, size=13)
    bp = box(s, 6.2, 4.85, 2.6, 0.8, ["(1 + a) backward", "1 × 1 = 1"], BLUE, size=13)
    ba = box(s, 9.6, 4.35, 2.4, 0.8, ["a.grad", "NaN + 1 = NaN"], RED, size=14, shape=MSO_SHAPE.RECTANGLE)
    for p, q in ((bg, bw), (bw, be), (bw, bp), (be, ba), (bp, ba)):
        arrow(s, p, q)
    textbox(s, 5.0, 3.85, 1.0, 0.3, ["grad 0"], size=11, color=RGBColor(0x55, 0x55, 0x55))
    textbox(s, 5.0, 5.35, 1.0, 0.3, ["grad 1"], size=11, color=RGBColor(0x55, 0x55, 0x55))
    # evidence
    textbox(s, 0.6, 6.0, 12.1, 1.2,
            ["NixNan on the GPU kernels: inf in an FMUL of the exp kernel; then NaN in an FMUL (0 × inf) and its STG; "
             "then NaN in an FFMA (accumulation) and its STG.",
             "Report: FP32 NaN at 5 sites, Infinity at 2 sites, FP32 memory NaN at 3 sites; all of it inside "
             "PyTorch's elementwise kernels, none in user code."], size=14)


# --------------------------------------------------------------------------
# Slide 10: why it stays, and the workaround
# --------------------------------------------------------------------------
WORKAROUNDS = [("issue: where(a<=0, exp(a), 1+a)", "101", "nan"),
               ("mask multiply: m × exp(a) + (1+a) × ~m", "nan", "nan"),
               ("double where: exp(a.clamp_max(0))", "101", "1.0"),
               ("double where: exp(where(m, a, 0))", "101", "1.0")]


def slide_where_why():
    s = prs.slides.add_slide(BLANK)
    title_bar(s, "torch.where: why it stays, and the workaround")
    bullets = [
        "Issue #68425: closed “not planned” (2022-07-25), as a duplicate of the still-open #4132. Reproduces "
        "unchanged on torch 2.3.1.",
        "Stated reason, from a maintainer: “it’ll stay like it is and nans propagate … It breaks IEEE754 simply "
        "because we don’t have the metadata available to track invalid elements.” Autograd has no “undefined grad”.",
        "Efficiency came up only as the cost of a fix: tracking a mask in forward “will require the CPU-GPU sync”. "
        "No maintainer says it is left for speed.",
        "The fix is the user’s: keep the unselected branch finite (“double where”). Thread: the same applies to log.",
    ]
    tb = s.shapes.add_textbox(Inches(0.6), Inches(1.15), Inches(12.1), Inches(2.9))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, t in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        add_runs(p, "•  " + t, 16)
    shp = s.shapes.add_table(len(WORKAROUNDS) + 1, 3, Inches(1.0), Inches(4.45), Inches(11.3), Inches(2.1))
    tbl = shp.table
    for i, wd in enumerate((6.8, 2.0, 2.5)):
        tbl.columns[i].width = Inches(wd)
    for c, t in enumerate(["form (a = 100, float32, CUDA)", "b", "a.grad"]):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK
        cell.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
        add_runs(cell.text_frame.paragraphs[0], t, 15, RGBColor(0xFF, 0xFF, 0xFF), FONT, bold=True)
    for r, row in enumerate(WORKAROUNDS, start=1):
        for c, t in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = GRAY if r % 2 else RGBColor(0xFF, 0xFF, 0xFF)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            add_runs(p, t, 15, INK, MONO if c == 0 else FONT)
    textbox(s, 1.0, 6.65, 11.3, 0.5,
            ["Sources: github.com/pytorch/pytorch/issues/68425; native runs, torch_where/workarounds.py."],
            size=11, color=RGBColor(0x55, 0x55, 0x55))


SLIDES = [slide_cover, slide_algorithm, slide_envvars, slide_time_kernels,
          slide_stage1_results, slide_fp32_artifact, slide_tc_panel,
          slide_tc_exceptions, slide_where_backward, slide_where_why]

if __name__ == "__main__":
    for f in SLIDES:
        f()
    prs.save("LU_Slides.pptx")
    print("wrote LU_Slides.pptx with", len(prs.slides), "slides")
