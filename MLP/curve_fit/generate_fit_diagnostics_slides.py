"""Generate engineer-facing slides for the q1 fit diagnostics.

The repository environment used for this project does not always have
PowerPoint, LaTeX, or python-pptx available. This script therefore writes a
minimal PPTX package directly with the Python standard library and also writes
an HTML slide deck that can be opened in a browser.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import struct
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent
MLP_DIR = THIS_DIR.parent
PROJECT_ROOT = MLP_DIR.parent
DIAG_DIR = MLP_DIR / "synthetic_data" / "fit_diagnostics"
SYN_DIR = MLP_DIR / "synthetic_data"
OUT_DIR = DIAG_DIR / "slides_mechanical_engineers"

SLIDE_W = 12_192_000
SLIDE_H = 6_858_000
EMU_PER_INCH = 914_400


def emu(inches: float) -> int:
    return int(round(inches * EMU_PER_INCH))


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else float("nan")


def parse_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = (len(values) - 1) * q / 100.0
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def median(values: list[float]) -> float:
    return quantile(values, 50.0)


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_int(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{int(round(value)):,}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nozzle_label(dataset_name: str) -> str:
    if dataset_name == "Nozzle0":
        return "Nozzle0"
    for token in dataset_name.lower().split("_"):
        if token.startswith("nozzle"):
            return token.capitalize()
    return dataset_name


def collect_stats() -> dict:
    summary_path = DIAG_DIR / "fit_diagnostics_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    rmse: list[float] = []
    r2: list[float] = []
    k_quarter: list[float] = []
    t0_ms: list[float] = []
    s_ms: list[float] = []
    by_nozzle: dict[str, list[float]] = defaultdict(list)
    by_condition: dict[tuple[float, float], list[float]] = defaultdict(list)

    for path in sorted(SYN_DIR.glob("*/cdf/clean/*.csv")):
        dataset = path.parts[-4]
        nozzle = nozzle_label(dataset)
        for row in read_csv_rows(path):
            v = parse_float(row.get("rmse"))
            if v is not None:
                rmse.append(v)
                by_nozzle[nozzle].append(v)
                pinj = parse_float(row.get("injection_pressure_bar"))
                pch = parse_float(row.get("chamber_pressure_bar"))
                if pinj is not None and pch is not None:
                    by_condition[(pinj, pch)].append(v)
            v = parse_float(row.get("r2"))
            if v is not None:
                r2.append(v)
            v = parse_float(row.get("k_quarter"))
            if v is not None:
                k_quarter.append(v)
            v = parse_float(row.get("t0"))
            if v is not None:
                t0_ms.append(v * 1e3)
            v = parse_float(row.get("s"))
            if v is not None:
                s_ms.append(v * 1e3)

    report_totals = Counter()
    for row in read_csv_rows(SYN_DIR / "fit_report.csv"):
        if row.get("penetration_source") != "cdf":
            continue
        report_totals["n_total"] += int(float(row.get("n_total") or 0))
        report_totals["n_clean"] += int(float(row.get("n_clean") or 0))
        report_totals["n_flagged"] += int(float(row.get("n_flagged") or 0))
        report_totals["success_main"] += int(float(row.get("success_main") or 0))

    flag_reasons = Counter()
    for path in sorted(SYN_DIR.glob("*/cdf/all/*.csv")):
        if path.name.endswith("_flagged.csv"):
            continue
        for row in read_csv_rows(path):
            if not parse_bool(row.get("flag_bad_fit")):
                continue
            if not parse_bool(row.get("mask_basic")):
                flag_reasons["basic gate failed"] += 1
            if not parse_bool(row.get("mask_penetration_far")):
                flag_reasons["5 ms penetration out of range"] += 1
            if parse_bool(row.get("mask_outlier")):
                flag_reasons["robust outlier"] += 1

    condition_summaries = []
    for condition, values in by_condition.items():
        if len(values) < 50:
            continue
        condition_summaries.append(
            {
                "pinj": condition[0],
                "pch": condition[1],
                "count": len(values),
                "median": median(values),
                "p95": quantile(values, 95),
            }
        )
    condition_summaries.sort(key=lambda r: r["median"])

    by_nozzle_summary = [
        {
            "nozzle": nozzle,
            "count": len(values),
            "median": median(values),
            "p95": quantile(values, 95),
        }
        for nozzle, values in sorted(by_nozzle.items())
    ]

    regression = read_csv_rows(DIAG_DIR / "param_distributions" / "scaling_regression.csv")
    convergence = read_csv_rows(DIAG_DIR / "convergence" / "convergence_by_status.csv")
    identifiability = read_csv_rows(DIAG_DIR / "identifiability" / "identifiability_summary.csv")
    residual = read_csv_rows(DIAG_DIR / "residual_structure" / "residual_vs_time.csv")

    return {
        "summary": summary,
        "rmse": {
            "n": len(rmse),
            "median": median(rmse),
            "q25": quantile(rmse, 25),
            "q75": quantile(rmse, 75),
            "p95": quantile(rmse, 95),
        },
        "r2": {
            "median": median(r2),
            "q25": quantile(r2, 25),
            "q75": quantile(r2, 75),
        },
        "params": {
            "k_median": median(k_quarter),
            "t0_ms_median": median(t0_ms),
            "s_ms_median": median(s_ms),
            "t0_ms_p25": quantile(t0_ms, 25),
            "t0_ms_p75": quantile(t0_ms, 75),
            "s_ms_p25": quantile(s_ms, 25),
            "s_ms_p75": quantile(s_ms, 75),
        },
        "report_totals": report_totals,
        "flag_reasons": flag_reasons,
        "by_nozzle": by_nozzle_summary,
        "conditions_low": condition_summaries[:3],
        "conditions_high": condition_summaries[-3:],
        "regression": regression,
        "convergence": convergence,
        "identifiability": identifiability,
        "residual": residual,
    }


def xml_escape(text: str) -> str:
    return html.escape(str(text), quote=False)


def ppt_run(text: str, size: int, color: str = "1F2937", bold: bool = False) -> str:
    b = ' b="1"' if bold else ""
    return (
        f'<a:r><a:rPr lang="en-US" sz="{size * 100}"{b} dirty="0">'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        '<a:latin typeface="Arial"/></a:rPr>'
        f"<a:t>{xml_escape(text)}</a:t></a:r>"
    )


def ppt_paragraph(text: str, size: int, color: str = "1F2937", bold: bool = False) -> str:
    return (
        '<a:p><a:pPr marL="0" indent="0"><a:lnSpc><a:spcPct val="105000"/>'
        "</a:lnSpc></a:pPr>"
        f"{ppt_run(text, size, color, bold)}"
        f'<a:endParaRPr lang="en-US" sz="{size * 100}"/></a:p>'
    )


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return (1600, 900)
    return struct.unpack(">II", data[16:24])


class Slide:
    def __init__(self, title: str = "") -> None:
        self.title = title
        self.shapes: list[str] = []
        self.rels: list[tuple[str, str, str]] = []
        self.shape_id = 2

    def next_id(self) -> int:
        out = self.shape_id
        self.shape_id += 1
        return out

    def rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        fill: str = "FFFFFF",
        line: str | None = "D1D5DB",
        text: Iterable[str] | None = None,
        size: int = 18,
        color: str = "1F2937",
        bold: bool = False,
        name: str = "Box",
    ) -> None:
        sid = self.next_id()
        line_xml = (
            f'<a:ln w="8000"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
            if line
            else '<a:ln><a:noFill/></a:ln>'
        )
        tx = ""
        if text is not None:
            paras = "".join(ppt_paragraph(t, size, color, bold and i == 0) for i, t in enumerate(text))
            tx = f'<p:txBody><a:bodyPr wrap="square" lIns="120000" rIns="120000" tIns="90000" bIns="90000"/><a:lstStyle/>{paras}</p:txBody>'
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{xml_escape(name)} {sid}"/>'
            '<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{line_xml}</p:spPr>{tx}</p:sp>'
        )

    def text(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        lines: Iterable[str],
        size: int = 20,
        color: str = "1F2937",
        bold_first: bool = False,
        name: str = "Text",
    ) -> None:
        self.rect(x, y, w, h, fill="FFFFFF", line=None, text=lines, size=size, color=color, bold=bold_first, name=name)

    def image(self, deck: "PptxDeck", path: Path, x: int, y: int, w: int, h: int, name: str = "Image") -> None:
        if not path.exists():
            self.text(x, y, w, h, [f"Missing image: {path.name}"], size=18, color="9B1C1C")
            return
        iw, ih = png_size(path)
        box_ratio = w / h
        img_ratio = iw / ih if ih else box_ratio
        if img_ratio > box_ratio:
            ph = int(w / img_ratio)
            px, py, pw = x, y + (h - ph) // 2, w
        else:
            pw = int(h * img_ratio)
            px, py, ph = x + (w - pw) // 2, y, h
        target = deck.media_target(path)
        rid = f"rId{len(self.rels) + 2}"
        self.rels.append((rid, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"../media/{target}"))
        sid = self.next_id()
        self.shapes.append(
            f'<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="{xml_escape(name)} {sid}"/>'
            '<p:cNvPicPr/><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr><a:xfrm><a:off x="{px}" y="{py}"/><a:ext cx="{pw}" cy="{ph}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
        )

    def title_bar(self, title: str, kicker: str = "") -> None:
        self.rect(0, 0, SLIDE_W, emu(0.62), fill="F3F6FA", line=None)
        self.text(emu(0.48), emu(0.12), emu(9.9), emu(0.42), [title], size=25, color="0F172A", bold_first=True)
        if kicker:
            self.text(emu(10.4), emu(0.18), emu(2.45), emu(0.32), [kicker], size=10, color="64748B")

    def footer(self) -> None:
        self.text(
            emu(0.48),
            emu(7.16),
            emu(12.5),
            emu(0.2),
            ["Source: MLP/synthetic_data/fit_diagnostics and MLP/curve_fit/fit_raw_data.py"],
            size=8,
            color="64748B",
        )

    def xml(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            '<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>'
            '<a:effectLst/></p:bgPr></p:bg><p:spTree>'
            '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            + "".join(self.shapes)
            + '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
        )

    def rels_xml(self) -> str:
        rels = [
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        ]
        for rid, rtype, target in self.rels:
            rels.append(f'<Relationship Id="{rid}" Type="{rtype}" Target="{xml_escape(target)}"/>')
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(rels)
            + "</Relationships>"
        )


class PptxDeck:
    def __init__(self) -> None:
        self.slides: list[Slide] = []
        self.media: dict[Path, str] = {}

    def media_target(self, path: Path) -> str:
        path = path.resolve()
        if path not in self.media:
            self.media[path] = f"image{len(self.media) + 1}{path.suffix.lower()}"
        return self.media[path]

    def add(self, slide: Slide) -> None:
        self.slides.append(slide)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", content_types_xml(len(self.slides)))
            z.writestr("_rels/.rels", package_rels_xml())
            z.writestr("docProps/core.xml", core_xml())
            z.writestr("docProps/app.xml", app_xml(len(self.slides)))
            z.writestr("ppt/presentation.xml", presentation_xml(len(self.slides)))
            z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels_xml(len(self.slides)))
            z.writestr("ppt/presProps.xml", pres_props_xml())
            z.writestr("ppt/viewProps.xml", view_props_xml())
            z.writestr("ppt/tableStyles.xml", table_styles_xml())
            z.writestr("ppt/theme/theme1.xml", theme_xml())
            z.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
            z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels_xml())
            z.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
            z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels_xml())
            for idx, slide in enumerate(self.slides, start=1):
                z.writestr(f"ppt/slides/slide{idx}.xml", slide.xml())
                z.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", slide.rels_xml())
            for source, target in self.media.items():
                z.write(source, f"ppt/media/{target}")


def content_types_xml(n_slides: int) -> str:
    slides = "".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, n_slides + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>'
        '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>'
        '<Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        + slides
        + "</Types>"
    )


def package_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def core_xml() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:title>Q1 Fit Diagnostics for Spray Penetration</dc:title>"
        "<dc:creator>Mie Spray Postprocessing Project</dc:creator>"
        "<cp:lastModifiedBy>Codex</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def app_xml(n_slides: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Codex standard-library PPTX writer</Application>"
        f"<Slides>{n_slides}</Slides>"
        "</Properties>"
    )


def presentation_xml(n_slides: int) -> str:
    slide_ids = "".join(
        f'<p:sldId id="{255 + i}" r:id="rId{i + 1}"/>' for i in range(1, n_slides + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" saveSubsetFonts="1">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        f"<p:sldIdLst>{slide_ids}</p:sldIdLst>"
        f'<p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>'
        '<p:notesSz cx="6858000" cy="9144000"/>'
        '<p:defaultTextStyle><a:defPPr><a:defRPr lang="en-US"/></a:defPPr></p:defaultTextStyle>'
        "</p:presentation>"
    )


def presentation_rels_xml(n_slides: int) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    ]
    rels.extend(
        f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, n_slides + 1)
    )
    base = n_slides + 2
    rels.extend(
        [
            f'<Relationship Id="rId{base}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>',
            f'<Relationship Id="rId{base + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>',
            f'<Relationship Id="rId{base + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>',
            f'<Relationship Id="rId{base + 3}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles" Target="tableStyles.xml"/>',
        ]
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def pres_props_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentationPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>'
    )


def view_props_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:viewPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:normalViewPr><p:restoredLeft sz="15620"/><p:restoredTop sz="94660"/></p:normalViewPr>'
        '<p:slideViewPr/><p:notesTextViewPr/><p:gridSpacing cx="72008" cy="72008"/></p:viewPr>'
    )


def table_styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>'
    )


def slide_master_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>'
        '<a:effectLst/></p:bgPr></p:bg><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '</p:spTree></p:cSld>'
        '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
        'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
        '<p:txStyles><p:titleStyle><a:lvl1pPr algn="l"><a:defRPr sz="3200"/></a:lvl1pPr></p:titleStyle>'
        '<p:bodyStyle><a:lvl1pPr algn="l"><a:defRPr sz="2000"/></a:lvl1pPr></p:bodyStyle>'
        '<p:otherStyle><a:lvl1pPr algn="l"><a:defRPr sz="1800"/></a:lvl1pPr></p:otherStyle></p:txStyles>'
        "</p:sldMaster>"
    )


def slide_master_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
        "</Relationships>"
    )


def slide_layout_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">'
        '<p:cSld name="Blank"><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
    )


def slide_layout_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
        "</Relationships>"
    )


def theme_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office">'
        '<a:themeElements><a:clrScheme name="Office">'
        '<a:dk1><a:srgbClr val="1F2937"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>'
        '<a:dk2><a:srgbClr val="334155"/></a:dk2><a:lt2><a:srgbClr val="F3F6FA"/></a:lt2>'
        '<a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="DC2626"/></a:accent2>'
        '<a:accent3><a:srgbClr val="059669"/></a:accent3><a:accent4><a:srgbClr val="D97706"/></a:accent4>'
        '<a:accent5><a:srgbClr val="7C3AED"/></a:accent5><a:accent6><a:srgbClr val="0891B2"/></a:accent6>'
        '<a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>'
        '</a:clrScheme><a:fontScheme name="Office"><a:majorFont><a:latin typeface="Arial"/></a:majorFont>'
        '<a:minorFont><a:latin typeface="Arial"/></a:minorFont></a:fontScheme>'
        '<a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
        '<a:gradFill rotWithShape="1"><a:gsLst><a:gs pos="0"><a:schemeClr val="phClr"/></a:gs>'
        '<a:gs pos="100000"><a:schemeClr val="phClr"/></a:gs></a:gsLst><a:lin ang="5400000" scaled="0"/></a:gradFill>'
        '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
        '<a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
        '<a:ln w="25400"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
        '<a:ln w="38100"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>'
        '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle>'
        '<a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>'
        '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
        '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
        '</a:bgFillStyleLst></a:fmtScheme></a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>'
    )


def stat_box(slide: Slide, x: float, y: float, w: float, h: float, value: str, label: str, accent: str = "2563EB") -> None:
    slide.rect(emu(x), emu(y), emu(w), emu(h), fill="F8FAFC", line="D9E2EC")
    slide.text(emu(x + 0.18), emu(y + 0.13), emu(w - 0.36), emu(0.32), [value], size=23, color=accent, bold_first=True)
    slide.text(emu(x + 0.18), emu(y + 0.53), emu(w - 0.36), emu(h - 0.58), [label], size=10, color="475569")


def add_standard(slide: Slide, title: str, kicker: str = "q1 fit diagnostics") -> None:
    slide.title_bar(title, kicker)
    slide.footer()


def build_pptx(stats: dict, pptx_path: Path) -> None:
    deck = PptxDeck()
    totals = stats["report_totals"]
    rmse = stats["rmse"]
    params = stats["params"]
    clean_pct = pct(totals["n_clean"], totals["n_total"])
    ok_pct = pct(totals["success_main"], totals["n_total"])

    s = Slide()
    s.rect(0, 0, SLIDE_W, SLIDE_H, fill="FFFFFF", line=None)
    s.text(emu(0.55), emu(0.55), emu(11.8), emu(0.65), ["Q1 Fit Diagnostics"], size=34, color="0F172A", bold_first=True)
    s.text(
        emu(0.58),
        emu(1.22),
        emu(11.4),
        emu(0.55),
        ["Spray penetration curve-fit quality, translated for mechanical engineering review"],
        size=18,
        color="475569",
    )
    s.rect(emu(0.55), emu(1.95), emu(12.2), emu(0.82), fill="EFF6FF", line="BFDBFE")
    s.text(
        emu(0.78),
        emu(2.09),
        emu(11.65),
        emu(0.5),
        [f"Main message: the clean CDF fits are generally usable for synthetic data generation. Median RMSE is {fmt(rmse['median'])} mm and median R2 is {fmt(stats['r2']['median'], 4)}."],
        size=18,
        color="1E3A8A",
        bold_first=True,
    )
    stat_box(s, 0.65, 3.15, 2.35, 1.0, fmt_int(totals["n_total"]), "candidate CDF plume traces")
    stat_box(s, 3.2, 3.15, 2.35, 1.0, fmt_int(totals["n_clean"]), f"clean accepted fits ({fmt(clean_pct, 1)}%)")
    stat_box(s, 5.75, 3.15, 2.35, 1.0, fmt(rmse["median"], 2), "median RMSE [mm]")
    stat_box(s, 8.3, 3.15, 2.35, 1.0, fmt(rmse["p95"], 2), "95th percentile RMSE [mm]", accent="DC2626")
    stat_box(s, 10.85, 3.15, 1.9, 1.0, fmt(ok_pct, 1) + "%", "clean and RMSE < 3 mm", accent="059669")
    s.text(
        emu(0.65),
        emu(4.65),
        emu(12.1),
        emu(1.2),
        [
            "What is being checked:",
            "- Does the simplified curve follow measured spray penetration closely enough?",
            "- Do the fitted constants move in physically sensible directions with pressure and nozzle geometry?",
            "- Are optimization failures or parameter ambiguity likely to contaminate the exported data?",
        ],
        size=17,
        color="334155",
        bold_first=True,
    )
    s.footer()
    deck.add(s)

    s = Slide()
    add_standard(s, "From Video Measurements To Diagnostics")
    steps = [
        ("1. Raw plume series", "Penetration in mm vs time for each plume"),
        ("2. Clean and align", "Remove early zeros, bad jumps, negative values, and align onset"),
        ("3. Fit q1 curve", "Least-squares fit of onset-ramped t^(1/4) penetration"),
        ("4. Quality gates", "Keep plausible fits, reject outliers and weak traces"),
        ("5. Aggregate", "Pool clean rows and per-frame residuals"),
        ("6. Diagnose", "RMSE, residuals, convergence, parameter uncertainty"),
    ]
    x0, y0 = 0.62, 1.15
    for i, (head, body) in enumerate(steps):
        row = i // 3
        col = i % 3
        x = x0 + col * 4.2
        y = y0 + row * 2.0
        s.rect(emu(x), emu(y), emu(3.75), emu(1.35), fill="F8FAFC", line="CBD5E1")
        s.text(emu(x + 0.18), emu(y + 0.15), emu(3.4), emu(0.35), [head], size=17, color="0F172A", bold_first=True)
        s.text(emu(x + 0.18), emu(y + 0.55), emu(3.35), emu(0.62), [body], size=12, color="475569")
        if col < 2:
            s.text(emu(x + 3.73), emu(y + 0.48), emu(0.35), emu(0.3), ["->"], size=18, color="94A3B8")
    s.text(
        emu(0.65),
        emu(5.35),
        emu(11.9),
        emu(0.75),
        ["Code map: prepare_cleaned_series -> fit_quarter_only -> apply_filter_masking -> fit_diagnostics.run_all"],
        size=18,
        color="1E3A8A",
        bold_first=True,
    )
    deck.add(s)

    s = Slide()
    add_standard(s, "The Fitted Curve In Plain Terms")
    s.rect(emu(0.75), emu(1.08), emu(12.0), emu(1.05), fill="F8FAFC", line="CBD5E1")
    s.text(
        emu(1.0),
        emu(1.28),
        emu(11.55),
        emu(0.45),
        ["S(t) = sigmoid((t - t0) / s) * k * t^(1/4)"],
        size=25,
        color="0F172A",
        bold_first=True,
    )
    s.text(
        emu(1.0),
        emu(1.77),
        emu(11.2),
        emu(0.25),
        ["The sigmoid is a smooth valve: before onset penetration is near zero, after onset the curve follows a quarter-root growth law."],
        size=12,
        color="475569",
    )
    stat_box(s, 0.82, 2.65, 3.8, 1.2, "k", f"growth scale. Median fitted k = {fmt(params['k_median'], 0)} mm/s^(1/4)")
    stat_box(s, 4.82, 2.65, 3.8, 1.2, "t0", f"onset time. Median = {fmt(params['t0_ms_median'], 3)} ms")
    stat_box(s, 8.82, 2.65, 3.8, 1.2, "s", f"transition width. Median = {fmt(params['s_ms_median'], 3)} ms")
    s.text(
        emu(0.85),
        emu(4.45),
        emu(11.8),
        emu(1.0),
        [
            "Mechanical interpretation:",
            "- k says how aggressively the visible penetration grows after onset.",
            "- t0 says when the plume becomes visible after hydraulic delay correction.",
            "- s says how sharply the plume turns on. Very large or nonphysical values are rejected.",
        ],
        size=16,
        color="334155",
        bold_first=True,
    )
    deck.add(s)

    s = Slide()
    add_standard(s, "Quality Gates Keep The Exported Curves Conservative")
    stat_box(s, 0.7, 1.05, 2.4, 1.0, fmt_int(totals["n_flagged"]), "flagged CDF fits")
    stat_box(s, 3.35, 1.05, 2.4, 1.0, fmt(clean_pct, 1) + "%", "clean survival rate")
    stat_box(s, 6.0, 1.05, 2.4, 1.0, fmt(rmse["q25"], 2) + "-" + fmt(rmse["q75"], 2), "middle 50% RMSE [mm]")
    stat_box(s, 8.65, 1.05, 2.4, 1.0, fmt_int(stats["summary"].get("n_series_rows", 0)), "clean per-frame points")
    reasons = stats["flag_reasons"].most_common()
    s.text(
        emu(0.75),
        emu(2.55),
        emu(5.75),
        emu(1.7),
        [
            "Most rejected fits are statistical outliers:",
            *[f"- {name}: {fmt_int(count)}" for name, count in reasons],
        ],
        size=16,
        color="334155",
        bold_first=True,
    )
    s.text(
        emu(6.85),
        emu(2.55),
        emu(5.75),
        emu(1.8),
        [
            "Hard plausibility checks include:",
            "- solver success and finite RMSE",
            "- at least 10 valid points",
            "- 0 < t0 < measured time range",
            "- s < 1 ms and t0 < 0.8 ms",
            "- penetration at 5 ms between 18 and 300 mm",
        ],
        size=15,
        color="334155",
        bold_first=True,
    )
    s.rect(emu(0.75), emu(5.05), emu(11.8), emu(0.75), fill="ECFDF5", line="BBF7D0")
    s.text(emu(1.0), emu(5.22), emu(11.3), emu(0.35), ["Engineering read: the deck focuses on clean rows; flagged rows are kept separately for audit, not mixed into training-quality data."], size=15, color="065F46", bold_first=True)
    deck.add(s)

    s = Slide()
    add_standard(s, "Fit Error Across Operating Conditions")
    s.image(deck, DIAG_DIR / "error_by_condition" / "rmse_heatmap_pinj_pch.png", emu(0.6), emu(1.0), emu(6.7), emu(4.8))
    low = stats["conditions_low"]
    high = stats["conditions_high"]
    s.text(
        emu(7.45),
        emu(1.08),
        emu(5.25),
        emu(2.0),
        [
            "What to look for:",
            "- Lower values mean the fitted curve is closer to the measured plume.",
            f"- Overall median RMSE is {fmt(rmse['median'])} mm; 95% are below {fmt(rmse['p95'])} mm.",
            "- Error is not uniform: the pressure condition matters.",
        ],
        size=15,
        color="334155",
        bold_first=True,
    )
    s.text(
        emu(7.45),
        emu(3.35),
        emu(5.2),
        emu(1.9),
        [
            "Lowest median-RMSE cells:",
            *[f"- Pinj {fmt(c['pinj'],0)} bar, Pch {fmt(c['pch'],2)} bar: {fmt(c['median'])} mm" for c in low],
            "Highest median-RMSE cells:",
            *[f"- Pinj {fmt(c['pinj'],0)} bar, Pch {fmt(c['pch'],2)} bar: {fmt(c['median'])} mm" for c in reversed(high)],
        ],
        size=12,
        color="475569",
        bold_first=True,
    )
    deck.add(s)

    s = Slide()
    add_standard(s, "Per-Nozzle Fit Error")
    s.image(deck, DIAG_DIR / "error_by_condition" / "rmse_by_nozzle.png", emu(0.55), emu(0.95), emu(7.35), emu(5.3))
    rows = sorted(stats["by_nozzle"], key=lambda r: r["median"])
    s.text(
        emu(8.1),
        emu(1.0),
        emu(4.7),
        emu(1.2),
        [
            "Nozzle comparison:",
            "- Boxes summarize the spread of RMSE by nozzle.",
            "- The fitted model performs best when the curve shape is repeatable within the nozzle and condition.",
        ],
        size=15,
        color="334155",
        bold_first=True,
    )
    s.text(
        emu(8.1),
        emu(2.55),
        emu(4.7),
        emu(2.4),
        [
            "Median RMSE by nozzle:",
            *[f"- {r['nozzle']}: {fmt(r['median'])} mm, p95 {fmt(r['p95'])} mm" for r in rows],
        ],
        size=11,
        color="475569",
        bold_first=True,
    )
    deck.add(s)

    s = Slide()
    add_standard(s, "Do The Parameters Move Like Physics Suggests?")
    s.image(deck, DIAG_DIR / "param_distributions" / "param_vs_conditions.png", emu(0.45), emu(0.98), emu(8.2), emu(4.7))
    reg = {r.get("target"): r for r in stats["regression"]}
    kreg = reg.get("k_quarter", {})
    s.text(
        emu(8.8),
        emu(1.05),
        emu(4.15),
        emu(1.6),
        [
            "Most important result:",
            f"- k scales as DeltaP^{fmt(parse_float(kreg.get('exp_delta_p')), 2)}, rho_air^{fmt(parse_float(kreg.get('exp_rho_air')), 2)}, diameter^{fmt(parse_float(kreg.get('exp_diameter')), 2)}.",
            "- Signs match the expected trend: higher pressure drop and larger holes increase penetration; higher chamber density reduces it.",
        ],
        size=13,
        color="334155",
        bold_first=True,
    )
    s.text(
        emu(8.8),
        emu(3.15),
        emu(4.15),
        emu(1.55),
        [
            "Caution:",
            f"- Simple pressure-density-diameter scaling explains only R2 = {fmt(parse_float(kreg.get('r2')), 2)} for k.",
            "- t0 and s have lower R2, so onset and transition width also depend on imaging, plume interaction, and cleaning details.",
        ],
        size=13,
        color="475569",
        bold_first=True,
    )
    deck.add(s)

    s = Slide()
    add_standard(s, "Residuals Show Where The Curve Is Biased")
    s.image(deck, DIAG_DIR / "residual_structure" / "residual_vs_time.png", emu(0.55), emu(0.92), emu(7.0), emu(5.45))
    residual = stats["residual"]
    def residual_at(time_text: str) -> dict[str, str]:
        return next((r for r in residual if str(r.get("time_ms", "")).startswith(time_text)), {})
    r005 = residual_at("0.05")
    r095 = residual_at("0.95")
    r155 = residual_at("1.55")
    s.text(
        emu(7.85),
        emu(1.0),
        emu(4.9),
        emu(2.0),
        [
            "Residual = fitted penetration - measured penetration.",
            f"- At 0.05 ms: median residual {fmt(parse_float(r005.get('median')))} mm.",
            f"- Around 0.95 ms: median residual {fmt(parse_float(r095.get('median')))} mm.",
            f"- At 1.55 ms: median residual {fmt(parse_float(r155.get('median')))} mm, but only {r155.get('n', 'few')} points.",
        ],
        size=14,
        color="334155",
        bold_first=True,
    )
    s.text(
        emu(7.85),
        emu(3.6),
        emu(4.9),
        emu(1.25),
        [
            "Engineering read:",
            "- The curve is close to unbiased after the initial transient.",
            "- Early time is the hardest region because onset timing and image thresholding dominate.",
            "- Late-time bins have low sample count, so do not over-interpret the tail.",
        ],
        size=13,
        color="475569",
        bold_first=True,
    )
    deck.add(s)

    s = Slide()
    add_standard(s, "The Solver Is Not The Bottleneck")
    s.image(deck, DIAG_DIR / "convergence" / "convergence_diagnostics.png", emu(0.5), emu(1.05), emu(8.0), emu(4.75))
    conv_lines = ["Termination summary:"]
    for row in stats["convergence"]:
        conv_lines.append(
            f"- {row.get('status_name')}: {fmt_int(float(row.get('count') or 0))} fits, median nfev {fmt(parse_float(row.get('median_nfev')), 0)}"
        )
    s.text(emu(8.75), emu(1.15), emu(4.1), emu(2.0), conv_lines, size=13, color="334155", bold_first=True)
    s.rect(emu(8.75), emu(3.75), emu(4.0), emu(1.0), fill="ECFDF5", line="BBF7D0")
    s.text(emu(8.95), emu(3.9), emu(3.6), emu(0.55), ["No max-iteration failure appears in the clean-row summary. Most fits stop because further improvement in cost becomes tiny."], size=13, color="065F46", bold_first=True)
    deck.add(s)

    s = Slide()
    add_standard(s, "Parameter Identifiability")
    s.image(deck, DIAG_DIR / "identifiability" / "identifiability.png", emu(0.55), emu(1.0), emu(7.7), emu(5.0))
    id_rows = {r.get("metric"): r for r in stats["identifiability"]}
    s.text(
        emu(8.45),
        emu(1.08),
        emu(4.35),
        emu(1.8),
        [
            "What this means:",
            f"- Typical log-space std: k {fmt(parse_float(id_rows.get('std_log_k_quarter', {}).get('median')), 3)}, t0 {fmt(parse_float(id_rows.get('std_log_t0', {}).get('median')), 3)}, s {fmt(parse_float(id_rows.get('std_log_s', {}).get('median')), 3)}.",
            "- These are small enough that the fitted curves are stable.",
        ],
        size=13,
        color="334155",
        bold_first=True,
    )
    s.text(
        emu(8.45),
        emu(3.3),
        emu(4.35),
        emu(1.65),
        [
            "Important caveat:",
            "- k, t0, and s are correlated, especially k with s.",
            "- Use the full fitted curve for engineering comparison. Avoid over-interpreting a single parameter by itself.",
        ],
        size=13,
        color="475569",
        bold_first=True,
    )
    deck.add(s)

    s = Slide()
    add_standard(s, "How To Use These Results")
    s.text(
        emu(0.8),
        emu(1.1),
        emu(5.8),
        emu(2.0),
        [
            "Use clean fits for:",
            "- generating synthetic penetration trajectories",
            "- training downstream MLP models",
            "- comparing conditions with a compact curve representation",
            "- estimating time-dependent residual spread",
        ],
        size=17,
        color="334155",
        bold_first=True,
    )
    s.text(
        emu(6.95),
        emu(1.1),
        emu(5.65),
        emu(2.0),
        [
            "Do not use them blindly for:",
            "- diagnosing individual failed plumes without checking flagged plots",
            "- claiming t0 or s are pure physical delays",
            "- extrapolating after the measured time range",
            "- treating late residual bins with low counts as final truth",
        ],
        size=17,
        color="334155",
        bold_first=True,
    )
    s.rect(emu(0.8), emu(4.1), emu(11.8), emu(1.2), fill="FEFCE8", line="FDE68A")
    s.text(
        emu(1.05),
        emu(4.28),
        emu(11.2),
        emu(0.68),
        ["Recommended next checks: inspect the highest-RMSE conditions, tighten early-onset alignment if needed, and report RMSE plus residual sigma(t) when using synthetic data."],
        size=17,
        color="854D0E",
        bold_first=True,
    )
    deck.add(s)

    s = Slide()
    add_standard(s, "File And Code Map")
    s.text(
        emu(0.8),
        emu(1.0),
        emu(11.7),
        emu(4.7),
        [
            "Primary fitting code:",
            "- MLP/curve_fit/fit_raw_data.py",
            "- Model: spray_penetration_model_quarter_only",
            "- Fit: fit_quarter_only uses scipy least_squares with Huber loss",
            "- Filtering: apply_filter_masking creates clean and flagged rows",
            "- End of script: summarize_dataset, summarize_filter_survival, fit_diagnostics",
            "",
            "Diagnostics used in this deck:",
            "- MLP/synthetic_data/fit_diagnostics/error_by_condition",
            "- MLP/synthetic_data/fit_diagnostics/param_distributions",
            "- MLP/synthetic_data/fit_diagnostics/residual_structure",
            "- MLP/synthetic_data/fit_diagnostics/convergence",
            "- MLP/synthetic_data/fit_diagnostics/identifiability",
        ],
        size=16,
        color="334155",
        bold_first=True,
    )
    deck.add(s)

    s = Slide()
    add_standard(s, "Bottom Line For Review")
    s.rect(emu(0.75), emu(1.05), emu(12.0), emu(1.0), fill="EFF6FF", line="BFDBFE")
    s.text(
        emu(1.0),
        emu(1.22),
        emu(11.45),
        emu(0.55),
        [f"Accepted q1 fits are accurate enough for the intended surrogate-data role: median error {fmt(rmse['median'])} mm, p95 {fmt(rmse['p95'])} mm, and {fmt(clean_pct, 1)}% of CDF candidates survive cleaning."],
        size=18,
        color="1E3A8A",
        bold_first=True,
    )
    s.text(
        emu(0.95),
        emu(2.6),
        emu(11.5),
        emu(2.45),
        [
            "Decision points:",
            "- Green light for using clean q1 CDF rows as compact synthetic data.",
            "- Keep flagged rows out of training unless manually reviewed.",
            "- If early-time accuracy matters most, improve onset alignment before changing the curve law.",
            "- If individual parameter interpretation matters, add a parameter-correlation caveat to reports.",
        ],
        size=18,
        color="334155",
        bold_first=True,
    )
    deck.add(s)

    deck.save(pptx_path)


def html_path(path: Path, base: Path) -> str:
    return os.path.relpath(path, start=base).replace(os.sep, "/")


def build_html(stats: dict, html_out: Path) -> None:
    totals = stats["report_totals"]
    rmse = stats["rmse"]
    clean_pct = pct(totals["n_clean"], totals["n_total"])
    ok_pct = pct(totals["success_main"], totals["n_total"])
    base = html_out.parent
    image = lambda p: html_path(p, base)
    slides = [
        {
            "title": "Q1 Fit Diagnostics",
            "body": f"""
            <p class="lead">Spray penetration curve-fit quality, translated for mechanical engineering review.</p>
            <div class="callout">Main message: the clean CDF fits are generally usable for synthetic data generation. Median RMSE is {fmt(rmse['median'])} mm and median R2 is {fmt(stats['r2']['median'], 4)}.</div>
            <div class="stats">
              <div><strong>{fmt_int(totals['n_total'])}</strong><span>candidate CDF plume traces</span></div>
              <div><strong>{fmt_int(totals['n_clean'])}</strong><span>clean accepted fits ({fmt(clean_pct, 1)}%)</span></div>
              <div><strong>{fmt(rmse['median'])}</strong><span>median RMSE [mm]</span></div>
              <div><strong>{fmt(rmse['p95'])}</strong><span>95th percentile RMSE [mm]</span></div>
              <div><strong>{fmt(ok_pct, 1)}%</strong><span>clean and RMSE &lt; 3 mm</span></div>
            </div>
            """,
        },
        {
            "title": "From Video Measurements To Diagnostics",
            "body": """
            <div class="grid3">
              <div><b>1. Raw plume series</b><br>Penetration in mm vs time for each plume</div>
              <div><b>2. Clean and align</b><br>Remove early zeros, bad jumps, negative values, and align onset</div>
              <div><b>3. Fit q1 curve</b><br>Least-squares fit of onset-ramped t^(1/4) penetration</div>
              <div><b>4. Quality gates</b><br>Keep plausible fits, reject outliers and weak traces</div>
              <div><b>5. Aggregate</b><br>Pool clean rows and per-frame residuals</div>
              <div><b>6. Diagnose</b><br>RMSE, residuals, convergence, parameter uncertainty</div>
            </div>
            <p class="callout">Code map: prepare_cleaned_series -> fit_quarter_only -> apply_filter_masking -> fit_diagnostics.run_all</p>
            """,
        },
        {
            "title": "The Fitted Curve In Plain Terms",
            "body": f"""
            <div class="equation">S(t) = sigmoid((t - t0) / s) * k * t^(1/4)</div>
            <p>The sigmoid is a smooth valve: before onset penetration is near zero, after onset the curve follows a quarter-root growth law.</p>
            <div class="grid3">
              <div><b>k</b><br>growth scale. Median fitted k = {fmt(stats['params']['k_median'], 0)} mm/s^(1/4)</div>
              <div><b>t0</b><br>onset time. Median = {fmt(stats['params']['t0_ms_median'], 3)} ms</div>
              <div><b>s</b><br>transition width. Median = {fmt(stats['params']['s_ms_median'], 3)} ms</div>
            </div>
            """,
        },
        {
            "title": "Quality Gates Keep The Exported Curves Conservative",
            "body": f"""
            <div class="stats">
              <div><strong>{fmt_int(totals['n_flagged'])}</strong><span>flagged CDF fits</span></div>
              <div><strong>{fmt(clean_pct, 1)}%</strong><span>clean survival rate</span></div>
              <div><strong>{fmt(rmse['q25'])}-{fmt(rmse['q75'])}</strong><span>middle 50% RMSE [mm]</span></div>
              <div><strong>{fmt_int(stats['summary'].get('n_series_rows', 0))}</strong><span>clean per-frame points</span></div>
            </div>
            <div class="twocol">
              <div><h3>Rejected mainly as outliers</h3>{'<br>'.join(f'- {html.escape(k)}: {fmt_int(v)}' for k, v in stats['flag_reasons'].most_common())}</div>
              <div><h3>Hard gates</h3>- solver success and finite RMSE<br>- at least 10 valid points<br>- 0 &lt; t0 &lt; measured range<br>- s &lt; 1 ms and t0 &lt; 0.8 ms<br>- 5 ms penetration between 18 and 300 mm</div>
            </div>
            """,
        },
        {
            "title": "Fit Error Across Operating Conditions",
            "body": f"""
            <div class="twocol wideleft">
              <img src="{image(DIAG_DIR / 'error_by_condition' / 'rmse_heatmap_pinj_pch.png')}">
              <div><h3>Engineering read</h3>Lower values mean a closer curve. Overall median RMSE is {fmt(rmse['median'])} mm; 95% are below {fmt(rmse['p95'])} mm. Pressure condition visibly changes the error level.</div>
            </div>
            """,
        },
        {
            "title": "Per-Nozzle Fit Error",
            "body": f"""
            <div class="twocol wideleft">
              <img src="{image(DIAG_DIR / 'error_by_condition' / 'rmse_by_nozzle.png')}">
              <div><h3>Nozzle comparison</h3>{'<br>'.join(f'- {r["nozzle"]}: median {fmt(r["median"])} mm, p95 {fmt(r["p95"])} mm' for r in sorted(stats['by_nozzle'], key=lambda r: r['median']))}</div>
            </div>
            """,
        },
        {
            "title": "Do The Parameters Move Like Physics Suggests?",
            "body": f"""
            <img class="full" src="{image(DIAG_DIR / 'param_distributions' / 'param_vs_conditions.png')}">
            <p class="callout">The k parameter has physically sensible signs: higher pressure drop and larger holes increase penetration; higher chamber density reduces it. The simple scaling is useful, but not the full story.</p>
            """,
        },
        {
            "title": "Residuals Show Where The Curve Is Biased",
            "body": f"""
            <div class="twocol wideleft">
              <img src="{image(DIAG_DIR / 'residual_structure' / 'residual_vs_time.png')}">
              <div><h3>Engineering read</h3>The curve is close to unbiased after the initial transient. Early time is hardest because onset timing and image thresholding dominate. Late bins have low count, so do not over-interpret the tail.</div>
            </div>
            """,
        },
        {
            "title": "The Solver Is Not The Bottleneck",
            "body": f"""
            <div class="twocol wideleft">
              <img src="{image(DIAG_DIR / 'convergence' / 'convergence_diagnostics.png')}">
              <div><h3>Termination summary</h3>{'<br>'.join(f'- {r.get("status_name")}: {fmt_int(float(r.get("count") or 0))} fits' for r in stats['convergence'])}<p class="callout">No max-iteration failure appears in the clean-row summary.</p></div>
            </div>
            """,
        },
        {
            "title": "Parameter Identifiability",
            "body": f"""
            <div class="twocol wideleft">
              <img src="{image(DIAG_DIR / 'identifiability' / 'identifiability.png')}">
              <div><h3>Interpretation</h3>The fitted curves are stable, but k, t0, and s co-move. Use the full curve for comparison and avoid over-interpreting a single parameter by itself.</div>
            </div>
            """,
        },
        {
            "title": "How To Use These Results",
            "body": """
            <div class="twocol">
              <div><h3>Use clean fits for</h3>- synthetic penetration trajectories<br>- downstream MLP training<br>- comparing conditions with a compact curve<br>- estimating residual spread over time</div>
              <div><h3>Do not use blindly for</h3>- diagnosing failed plumes without flagged plots<br>- claiming t0 or s are pure physical delays<br>- extrapolating after measured time<br>- treating sparse late residual bins as final truth</div>
            </div>
            <p class="callout warn">Recommended next checks: inspect highest-RMSE conditions, tighten early-onset alignment if needed, and report RMSE plus residual sigma(t) when using synthetic data.</p>
            """,
        },
        {
            "title": "File And Code Map",
            "body": """
            <div class="twocol">
              <div><h3>Primary fitting code</h3>
                - MLP/curve_fit/fit_raw_data.py<br>
                - Model: spray_penetration_model_quarter_only<br>
                - Fit: fit_quarter_only uses scipy least_squares with Huber loss<br>
                - Filtering: apply_filter_masking creates clean and flagged rows<br>
                - End of script: summarize_dataset, summarize_filter_survival, fit_diagnostics
              </div>
              <div><h3>Diagnostics used here</h3>
                - error_by_condition<br>
                - param_distributions<br>
                - residual_structure<br>
                - convergence<br>
                - identifiability
              </div>
            </div>
            """,
        },
        {
            "title": "Bottom Line For Review",
            "body": f"""
            <div class="callout">Accepted q1 fits are accurate enough for the intended surrogate-data role: median error {fmt(rmse['median'])} mm, p95 {fmt(rmse['p95'])} mm, and {fmt(clean_pct, 1)}% of CDF candidates survive cleaning.</div>
            <ul>
              <li>Green light for using clean q1 CDF rows as compact synthetic data.</li>
              <li>Keep flagged rows out of training unless manually reviewed.</li>
              <li>If early-time accuracy matters most, improve onset alignment before changing the curve law.</li>
              <li>If individual parameter interpretation matters, add a parameter-correlation caveat to reports.</li>
            </ul>
            """,
        },
    ]

    css = """
    body { margin: 0; background: #e5e7eb; font-family: Arial, sans-serif; color: #1f2937; }
    .slide { width: 1280px; height: 720px; box-sizing: border-box; margin: 24px auto; background: white; padding: 56px 64px 44px; position: relative; box-shadow: 0 8px 24px rgba(15,23,42,.16); page-break-after: always; }
    .slide:before { content: attr(data-title); position: absolute; left: 0; top: 0; right: 0; height: 48px; background: #f3f6fa; padding: 14px 64px; box-sizing: border-box; font-weight: 700; font-size: 24px; color: #0f172a; }
    .slide:after { content: "Source: MLP/synthetic_data/fit_diagnostics and MLP/curve_fit/fit_raw_data.py"; position: absolute; left: 64px; right: 64px; bottom: 18px; color: #64748b; font-size: 11px; }
    h1 { font-size: 42px; margin: 0 0 16px; }
    h3 { margin: 0 0 12px; color: #0f172a; }
    p, li { font-size: 22px; line-height: 1.35; }
    .lead { font-size: 26px; color: #475569; }
    .callout { background: #eff6ff; border: 1px solid #bfdbfe; color: #1e3a8a; padding: 18px 22px; font-size: 22px; font-weight: 700; margin: 20px 0; }
    .callout.warn { background: #fefce8; border-color: #fde68a; color: #854d0e; }
    .stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin: 28px 0; }
    .stats div, .grid3 div, .twocol > div { background: #f8fafc; border: 1px solid #d9e2ec; padding: 16px; box-sizing: border-box; }
    .stats strong { display: block; color: #2563eb; font-size: 32px; margin-bottom: 8px; }
    .stats span { color: #475569; font-size: 15px; line-height: 1.2; }
    .grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin: 22px 0; }
    .grid3 div, .twocol div { font-size: 20px; line-height: 1.35; }
    .twocol { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }
    .twocol.wideleft { grid-template-columns: 2fr 1fr; }
    img { max-width: 100%; max-height: 560px; object-fit: contain; }
    img.full { width: 100%; height: 480px; object-fit: contain; }
    .equation { font-size: 34px; font-weight: 700; color: #0f172a; background: #f8fafc; border: 1px solid #d9e2ec; padding: 22px; margin: 22px 0; text-align: center; }
    @media print { body { background: white; } .slide { margin: 0; box-shadow: none; width: 100vw; height: 56.25vw; } }
    """
    body = "\n".join(f'<section class="slide" data-title="{html.escape(s["title"])}">{s["body"]}</section>' for s in slides)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Q1 Fit Diagnostics</title>"
        f"<style>{css}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )


def validate_xml_in_pptx(path: Path) -> None:
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith(".xml") or name.endswith(".rels"):
                ET.fromstring(z.read(name))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = collect_stats()
    pptx_path = OUT_DIR / "fit_diagnostics_mechanical_engineers.pptx"
    html_out = OUT_DIR / "fit_diagnostics_mechanical_engineers.html"
    build_pptx(stats, pptx_path)
    validate_xml_in_pptx(pptx_path)
    build_html(stats, html_out)
    print(f"Wrote {pptx_path}")
    print(f"Wrote {html_out}")


if __name__ == "__main__":
    main()
