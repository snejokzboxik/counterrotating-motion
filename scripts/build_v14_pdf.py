"""Build the V14 Russian report as a one-column, journal-style PDF.

The Markdown report remains the source of truth. This formatter supports the
headings, paragraphs, lists, tables, links, inline code, emphasis and figures
used by that report and writes a single-column A4 PDF beside the V14 results.
"""
from __future__ import annotations

import html
import os
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs/octupole_comparison_home/mass_filter_v14"
SOURCE = REPORT_DIR / "REPORT_RU.md"
OUTPUT = REPORT_DIR / "REPORT_V14_PR_ONE_COLUMN_RU.pdf"
if os.environ.get("V14_FONT_DIR"):
    FONT_DIR = Path(os.environ["V14_FONT_DIR"])
else:
    try:
        import matplotlib
        FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    except ImportError:
        FONT_DIR = ROOT / ".venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf"


def normalize_dashes(text: str) -> str:
    return text.replace("\u2010", "-").replace("\u2011", "-").replace(
        "\u2012", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")


def inline_markup(text: str) -> str:
    text = normalize_dashes(text)
    replacements: list[tuple[str, str]] = []

    def stash(value: str) -> str:
        token = f"V14TOKEN{len(replacements)}END"
        replacements.append((token, value))
        return token

    def code_repl(match: re.Match[str]) -> str:
        value = html.escape(match.group(1))
        return stash(f'<font name="V14Mono">{value}</font>')

    text = re.sub(r"`([^`]+)`", code_repl, text)

    def link_repl(match: re.Match[str]) -> str:
        label = html.escape(normalize_dashes(match.group(1)))
        href = html.escape(match.group(2), quote=True)
        return stash(f'<link href="{href}" color="#174a72">{label}</link>')

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_repl, text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    for token, value in replacements:
        text = text.replace(token, value)
    return text


def register_fonts() -> None:
    pairs = {
        "V14Serif": "DejaVuSerif.ttf",
        "V14Serif-Bold": "DejaVuSerif-Bold.ttf",
        "V14Serif-Italic": "DejaVuSerif-Italic.ttf",
        "V14Serif-BoldItalic": "DejaVuSerif-BoldItalic.ttf",
        "V14Mono": "DejaVuSansMono.ttf",
    }
    for name, filename in pairs.items():
        path = FONT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(path)
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        "V14Serif", normal="V14Serif", bold="V14Serif-Bold",
        italic="V14Serif-Italic", boldItalic="V14Serif-BoldItalic",
    )


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["BodyText"]
    return {
        "title": ParagraphStyle(
            "V14Title", parent=base, fontName="V14Serif-Bold", fontSize=18,
            leading=22, alignment=TA_CENTER, textColor=colors.HexColor("#172d3d"),
            spaceAfter=5 * mm,
        ),
        "date": ParagraphStyle(
            "V14Date", parent=base, fontName="V14Serif-Italic", fontSize=9,
            leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#596c79"),
            spaceAfter=5 * mm,
        ),
        "h1": ParagraphStyle(
            "V14H1", parent=base, fontName="V14Serif-Bold", fontSize=14,
            leading=17, textColor=colors.HexColor("#16384f"), spaceBefore=5 * mm,
            spaceAfter=2.2 * mm, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "V14H2", parent=base, fontName="V14Serif-Bold", fontSize=11.3,
            leading=14, textColor=colors.HexColor("#244b63"), spaceBefore=3.4 * mm,
            spaceAfter=1.5 * mm, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "V14Body", parent=base, fontName="V14Serif", fontSize=9.25,
            leading=12.4, alignment=TA_JUSTIFY, textColor=colors.HexColor("#202a32"),
            spaceAfter=2.3 * mm, allowWidows=0, allowOrphans=0,
        ),
        "list": ParagraphStyle(
            "V14List", parent=base, fontName="V14Serif", fontSize=9.1,
            leading=12, leftIndent=13, firstLineIndent=-8, alignment=TA_LEFT,
            textColor=colors.HexColor("#202a32"), spaceAfter=1.2 * mm,
        ),
        "cell": ParagraphStyle(
            "V14Cell", parent=base, fontName="V14Serif", fontSize=7.1,
            leading=8.6, alignment=TA_LEFT, textColor=colors.HexColor("#202a32"),
        ),
        "cell_header": ParagraphStyle(
            "V14CellHeader", parent=base, fontName="V14Serif-Bold", fontSize=7.0,
            leading=8.5, alignment=TA_LEFT, textColor=colors.white,
        ),
        "caption": ParagraphStyle(
            "V14Caption", parent=base, fontName="V14Serif-Italic", fontSize=8.0,
            leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#52616b"),
            spaceBefore=1 * mm, spaceAfter=3 * mm,
        ),
    }


def parse_table(block: list[str], styles: dict[str, ParagraphStyle], width: float) -> Table:
    rows: list[list[str]] = []
    for line in block:
        if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line.strip()):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return Table([[""]])
    ncols = max(map(len, rows))
    rows = [row + [""] * (ncols - len(row)) for row in rows]
    weights = []
    for col in range(ncols):
        longest = max((len(re.sub(r"[`*_]", "", row[col])) for row in rows), default=8)
        weights.append(max(5.0, min(23.0, float(longest))))
    total = sum(weights)
    col_widths = [width * value / total for value in weights]
    data = []
    for ri, row in enumerate(rows):
        sty = styles["cell_header"] if ri == 0 else styles["cell"]
        data.append([Paragraph(inline_markup(cell), sty) for cell in row])
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#234b63")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f7")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c5cfd5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def is_table_line(line: str) -> bool:
    return line.strip().startswith("|") and "|" in line.strip()[1:]


class V14DocTemplate(BaseDocTemplate):
    def __init__(self, filename: Path, **kwargs):
        super().__init__(str(filename), **kwargs)
        frame = Frame(
            self.leftMargin, self.bottomMargin, self.width, self.height,
            id="one-column", leftPadding=0, rightPadding=0,
            topPadding=0, bottomPadding=0,
        )
        self.addPageTemplates([PageTemplate(id="V14", frames=[frame], onPage=self.decorate)])

    def decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setFont("V14Serif-Italic", 7.2)
            canvas.setFillColor(colors.HexColor("#667984"))
            canvas.drawString(doc.leftMargin, A4[1] - 13 * mm, "V14 | нелинейная RF-ловушка")
            canvas.setStrokeColor(colors.HexColor("#bcc7ce"))
            canvas.setLineWidth(0.45)
            canvas.line(doc.leftMargin, A4[1] - 15 * mm, A4[0] - doc.rightMargin, A4[1] - 15 * mm)
        canvas.setFont("V14Serif", 8)
        canvas.setFillColor(colors.HexColor("#667984"))
        canvas.drawCentredString(A4[0] / 2, 11 * mm, str(doc.page))
        canvas.restoreState()


def parse_markdown(styles: dict[str, ParagraphStyle], body_width: float):
    lines = SOURCE.read_text(encoding="utf8").splitlines()
    story = []
    i = 0
    title_done = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("!["):
            match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            if match:
                caption, target = match.groups()
                image_path = REPORT_DIR / target
                if not image_path.exists():
                    raise FileNotFoundError(image_path)
                img = Image(str(image_path))
                scale = min(body_width / img.imageWidth, 118 * mm / img.imageHeight, 1.0)
                img.drawWidth = img.imageWidth * scale
                img.drawHeight = img.imageHeight * scale
                story.append(KeepTogether([Spacer(1, 2 * mm), img,
                                          Paragraph(inline_markup(caption), styles["caption"])]))
                i += 1
                continue
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            block = []
            while i < len(lines) and is_table_line(lines[i]):
                block.append(lines[i])
                i += 1
            story.extend([Spacer(1, 1.2 * mm), parse_table(block, styles, body_width), Spacer(1, 2.2 * mm)])
            continue
        if line.startswith("# "):
            if not title_done:
                story.append(Paragraph(inline_markup(line[2:]), styles["title"]))
                title_done = True
            else:
                story.append(Paragraph(inline_markup(line[2:]), styles["h1"]))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), styles["h1"]))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), styles["h2"]))
            i += 1
            continue
        if line.startswith("*") and line.endswith("*") and len(line) > 2:
            story.append(Paragraph(inline_markup(line), styles["date"]))
            i += 1
            continue
        if line.startswith("- ") or re.match(r"\d+\. ", line):
            prefix = "•" if line.startswith("- ") else "•"
            text = line[2:] if line.startswith("- ") else re.sub(r"^\d+\.\s+", "", line)
            story.append(Paragraph(f"{prefix} {inline_markup(text)}", styles["list"]))
            i += 1
            continue

        paragraph = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("# ", "## ", "### ", "![", "|", "- "))
                    or re.match(r"\d+\. ", nxt)):
                break
            paragraph.append(nxt)
            i += 1
        block = Paragraph(inline_markup(" ".join(paragraph)), styles["body"])
        # Keep the report's final, numbered scientific conclusions together;
        # splitting that paragraph across pages made the last page start mid-claim.
        if paragraph[0].startswith("В статье можно осторожно утверждать:"):
            conclusion = [block]
            if (story and isinstance(story[-1], Paragraph)
                    and story[-1].getPlainText() == "Краткая формулировка результата V14"):
                conclusion.insert(0, story.pop())
            story.append(KeepTogether(conclusion))
        else:
            story.append(block)
    return story


def main() -> None:
    register_fonts()
    styles = build_styles()
    page_w, page_h = A4
    left = right = 21 * mm
    top, bottom = 23 * mm, 19 * mm
    doc = V14DocTemplate(
        OUTPUT, pagesize=A4, leftMargin=left, rightMargin=right,
        topMargin=top, bottomMargin=bottom, title="V14: нелинейное удержание за границей Флоке",
        author="Исследовательский отчёт проекта 12-electrode RF trap",
        subject="Сравнение Q, Q+SO и Q+RO на идеальной модели и полном P4 FEM",
    )
    story = parse_markdown(styles, page_w - left - right)
    doc.build(story)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
