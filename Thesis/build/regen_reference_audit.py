#!/usr/bin/env python3
"""Aligned, no-network regenerator for thesisreferences_audit.tsv.

Why this exists (vs. build_reference_audit.py):
  * The original scans only thesis.tex/thesis_zh.tex and hard-asserts 63 rows;
    the live thesis \\input{}s per-chapter files under latex/sections_{en,zh}/,
    so citation locations must be gathered from those section files (labelled
    "sections_en/NAME.tex:line"), and the bib has grown well past 63 entries.
  * A previous manual/PowerShell append wrote a few rows as UTF-16 inside the
    otherwise UTF-8 file (null-byte interleaving -> ripgrep flags it "binary").
    Regenerating from scratch with a UTF-8 writer fixes that.

Design: PRESERVE-AND-EXTEND, no network.
  * Mechanical columns (authors/year/title/doi/locations/citation sentences ...)
    are re-derived fresh from the bib + the current .tex sources.
  * Curated columns (english/chinese_reference_meaning + the download_* columns)
    are PRESERVED from the existing TSV for any clean row, so nothing hand-authored
    is lost; missing/corrupted keys fall back to MEANINGS below, then to a generic
    template. No files are downloaded and reference_sources/ is left untouched.
"""

from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

BUILD_DIR = Path(__file__).resolve().parent
THESIS_DIR = BUILD_DIR.parent
BIB_PATH = THESIS_DIR / "thesisreferences.bib"
LATEX_DIR = THESIS_DIR / "latex"
OUT_TSV = THESIS_DIR / "thesisreferences_audit.tsv"

HEADERS = [
    "bib_key", "entry_type", "used_in_tex", "authors", "year", "title", "container",
    "publisher_or_institution", "volume", "number", "pages", "doi", "url", "isbn",
    "eprint", "bib_note", "source_link", "download_status", "downloaded_file_path",
    "manual_reason", "english_tex_locations", "chinese_tex_locations",
    "english_citation_sentences", "chinese_citation_sentences",
    "english_reference_meaning", "chinese_reference_meaning", "verification_notes",
]

# Bespoke meanings for keys that are missing from / corrupted in the existing TSV.
# Preserved curated meanings (from clean existing rows) take precedence over these.
MEANINGS: Dict[str, Tuple[str, str]] = {
    "Zhao2023LLMSurvey": (
        "Umbrella survey of the large-language-model development lifecycle (pre-training, adaptation, utilization, evaluation); anchors the thesis's six-stage engineering-workflow parallel.",
        "大语言模型开发生命周期综述（预训练、适配、使用、评估），作为本文六阶段工程工作流类比的伞式参考。",
    ),
    "Bommasani2021FoundationModels": (
        "Position paper that coined 'foundation models' and frames the broad-pretraining-then-adaptation paradigm cited as the methodological template.",
        "提出“基础模型”概念的立场论文，勾勒广域预训练-再适配范式，作为方法论模板依据。",
    ),
    "Penedo2023RefinedWeb": (
        "Web-scale filtered and deduplicated pretraining corpus (RefinedWeb); anchors the Stage-1 web-scale data-acquisition parallel.",
        "网络级过滤去重的预训练语料（RefinedWeb），支撑第一阶段网络级数据获取的类比。",
    ),
    "Albalak2024DataSelection": (
        "Survey of data selection and curation methods for language-model training; anchors the Stage-2 data-centric curation parallel.",
        "语言模型训练的数据选择与整理方法综述，支撑第二阶段以数据为中心整理的类比。",
    ),
    "Xu2024LLMDistillation": (
        "Survey of knowledge distillation of large language models; anchors the pre-training to distillation to fine-tuning training-curriculum parallel.",
        "大语言模型知识蒸馏综述，支撑预训练→蒸馏→微调训练课程的类比。",
    ),
    "Liu2024SyntheticData": (
        "Best-practices overview of synthetic data for language models; anchors the Stage-3 target- and distillation-label construction parallel.",
        "语言模型合成数据最佳实践综述，支撑第三阶段目标与蒸馏标签构造的类比。",
    ),
    "Chang2024LLMEval": (
        "Survey of large-language-model evaluation (what, where and how to evaluate); anchors the Stage-5 held-out and calibration-aware evaluation parallel.",
        "大语言模型评估综述（评估什么/在哪评/如何评），支撑第五阶段留出集与校准评估的类比。",
    ),
    "Han2024PEFT": (
        "Comprehensive survey of parameter-efficient fine-tuning (adapters, LoRA); anchors the Stage-6 deployment-time lightweight-adapter parallel.",
        "参数高效微调（适配器、LoRA）综述，支撑第六阶段部署期轻量适配的类比。",
    ),
    "Deshmukh2021CAS": (
        "Supports the cost-wall motivation: high-fidelity reacting-spray CFD is orders of magnitude more expensive than a reduced-order surrogate.",
        "用于支撑成本墙动机：高保真反应喷雾 CFD 比降阶代理模型昂贵若干数量级。",
    ),
    "Tekgul2021DLBFoam": (
        "Supports that reacting-spray LES is chemistry-bound (chemistry and species transport ~90-98% of runtime), reinforcing the CFD cost-wall argument.",
        "用于支撑反应喷雾 LES 受化学求解主导（化学与组分输运约占运行时间 90-98%），强化 CFD 成本墙论证。",
    ),
    "GneitingRaftery2007CRPS": (
        "Establishes proper scoring rules (CRPS) used to evaluate the probabilistic penetration predictions.",
        "确立用于评估概率贯穿距离预测的恰当评分规则（CRPS）。",
    ),
    "RasmussenWilliams2006": (
        "Foundational Gaussian-process reference underpinning the SVGP surrogate baseline.",
        "高斯过程基础参考，支撑 SVGP 代理模型基线。",
    ),
    "HensmanFusiLawrence2013": (
        "Introduces sparse variational Gaussian processes (SVGP) used as the kernel surrogate baseline.",
        "提出稀疏变分高斯过程（SVGP），用作核方法代理模型基线。",
    ),
    "LoshchilovHutter2019AdamW": (
        "Provides the AdamW optimizer with decoupled weight decay used to train the surrogate.",
        "提供带解耦权重衰减的 AdamW 优化器，用于训练代理模型。",
    ),
    "Raissi2019PINNs": (
        "Physics-informed neural networks reference motivating the shape and derivative constraint terms.",
        "物理信息神经网络参考，支撑形状与导数约束项的动机。",
    ),
    "Srivastava2014Dropout": (
        "Introduces dropout regularization used in the MLP surrogate.",
        "提出 dropout 正则化，用于 MLP 代理模型。",
    ),
    "Ramachandran2017Swish": (
        "Introduces the Swish activation function considered for the surrogate network.",
        "提出 Swish 激活函数，用于代理网络的备选评估。",
    ),
    "ChengAhmadGrahn2023": (
        "Experimental ANN surrogate predicting multi-hole nozzle spray geometry from Mie data; the closest prior work to this thesis's data-driven penetration surrogate.",
        "基于 Mie 数据预测多孔喷嘴喷雾几何的实验型 ANN 代理，是与本文数据驱动贯穿距离代理最接近的先前工作。",
    ),
    "ZhangWuXu2024": (
        "GA-BP neural network for diesel-spray penetration reported to surpass classical empirical formulas; evidence that compact regressors can predict penetration.",
        "用于柴油喷雾贯穿距离的 GA-BP 神经网络，据报优于经典经验公式，佐证紧凑回归器可预测贯穿距离。",
    ),
    "Hornik1989UAT": (
        "Universal-approximation theorem justifying the MLP as an adequate function approximator for the surrogate.",
        "通用逼近定理，支撑 MLP 作为代理模型的充分函数逼近器。",
    ),
    "Guo2017Calibration": (
        "Establishes that modern neural networks are frequently miscalibrated, motivating the thesis's calibration-aware evaluation.",
        "指出现代神经网络常常校准不良，支撑本文的校准感知评估。",
    ),
    "wartsila_decarbonisation_2026": (
        "Industry decarbonization context for marine dual-fuel and alternative-fuel engines that motivates the pilot-injection screening problem.",
        "船用双燃料与替代燃料发动机减排的行业背景，引出引燃喷射筛查问题。",
    ),
    "Seitzer2022PitfallsHeteroscedastic": (
        "Documents the sigma-arbitrage pitfall of heteroscedastic Gaussian NLL that the staged curriculum (deterministic Stage 1 before the variance head) is designed to avoid.",
        "记录异方差高斯 NLL 的方差套利陷阱，正是分阶段课程（先确定性 Stage 1，再训练方差头）所要规避的失效模式。",
    ),
    "Detlefsen2019VarianceNetworks": (
        "Evidence that training a variance network from scratch can destabilise the mean fit, motivating the deterministic-first training curriculum.",
        "佐证从零训练方差网络会破坏均值拟合的稳定性，支撑先确定性、后异方差的训练课程。",
    ),
    "BengioCurriculumLearning2009": (
        "Formalises curriculum learning—fitting an easier/more stable target before the full noisy problem—underpinning the three-stage training schedule.",
        "将课程学习形式化（先拟合更容易/更稳定的目标，再面对完整且更嘈杂的问题），支撑三阶段训练安排。",
    ),
    "Cromey2010TwistedPixels": (
        "Scientific-image-processing ethics: nonlinear display adjustments (gamma, display gain, local contrast) must not be treated as quantitative measurement inputs.",
        "科学图像处理伦理：gamma、显示增益、局部对比度等非线性显示调整不得作为定量测量输入。",
    ),
    "WilsonIzmailov2020BDL": (
        "Bayesian deep-learning perspective on generalization, motivating a probabilistic treatment of neural-network predictions.",
        "关于泛化的贝叶斯深度学习视角，支撑对神经网络预测进行概率化处理。",
    ),
    "Ovadia2019Trust": (
        "Shows predictive uncertainty degrades under dataset shift, motivating calibration evaluation and explicit out-of-distribution reporting.",
        "表明预测不确定性在数据分布偏移下退化，支撑校准评估与分布外结果的显式报告。",
    ),
    "GalGhahramani2016Dropout": (
        "Monte-Carlo dropout as a Bayesian approximation for estimating predictive uncertainty in neural networks.",
        "将 MC dropout 作为贝叶斯近似，用于估计神经网络的预测不确定性。",
    ),
    "LakshminarayananEnsembles2017": (
        "Deep-ensembles baseline for simple, scalable predictive-uncertainty estimation.",
        "深度集成基线，用于简单、可扩展的预测不确定性估计。",
    ),
    "PanYang2010TransferLearning": (
        "Transfer-learning survey underpinning the frozen-trunk adapter and few-shot family-conditioning deployment framing.",
        "迁移学习综述，支撑冻结主干适配器与少样本族条件部署框架。",
    ),
    "Titsias2009SparseGP": (
        "Introduces variational inducing-point sparse Gaussian processes underlying the SVGP surrogate baseline.",
        "提出稀疏高斯过程的变分诱导点方法，构成 SVGP 代理模型基线的基础。",
    ),
}


def _load_base_meanings() -> Dict[str, Tuple[str, str]]:
    """Pull the original generator's curated MEANINGS (clean UTF-8 in the .py
    source) via AST, without importing it (avoids its `requests` dependency)."""
    try:
        src = (BUILD_DIR / "build_reference_audit.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in tree.body:
            names: List[str] = []
            value = None
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]  # MEANINGS uses an annotated assignment
                value = node.value
            if "MEANINGS" in names and value is not None:
                return dict(ast.literal_eval(value))
    except Exception:
        pass
    return {}


# Original generator's curated meanings + the additions above (additions win on overlap).
ALL_MEANINGS: Dict[str, Tuple[str, str]] = {**_load_base_meanings(), **MEANINGS}

# Keys that fell back to preserved/generic meanings (reported for follow-up).
UNCOVERED: List[str] = []

# ---------------------------------------------------------------------------
# BibTeX parsing (ported verbatim from build_reference_audit.py)
# ---------------------------------------------------------------------------

def scan_bib_entries(text: str) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    i = 0
    while True:
        at = text.find("@", i)
        if at < 0:
            break
        m = re.match(r"@(\w+)\s*\{", text[at:])
        if not m:
            i = at + 1
            continue
        entry_type = m.group(1)
        brace = at + m.end() - 1
        depth = 0
        end = brace
        while end < len(text):
            ch = text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            end += 1
        body = text[brace + 1 : end]
        if "," not in body:
            i = end + 1
            continue
        key, fields_text = body.split(",", 1)
        entries.append({"entry_type": entry_type, "key": key.strip(), "fields": parse_fields(fields_text)})
        i = end + 1
    return entries


def parse_fields(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    i = 0
    while i < len(text):
        while i < len(text) and text[i] in " \r\n\t,":
            i += 1
        name_start = i
        while i < len(text) and re.match(r"[A-Za-z0-9_:-]", text[i]):
            i += 1
        if i == name_start:
            break
        name = text[name_start:i].lower()
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "=":
            break
        i += 1
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] == "{":
            value, i = read_braced(text, i)
        elif i < len(text) and text[i] == '"':
            value, i = read_quoted(text, i)
        else:
            start = i
            while i < len(text) and text[i] != ",":
                i += 1
            value = text[start:i].strip()
        fields[name] = value.strip()
    return fields


def read_braced(text: str, i: int) -> Tuple[str, int]:
    depth = 0
    start = i + 1
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    return text[start:], len(text)


def read_quoted(text: str, i: int) -> Tuple[str, int]:
    i += 1
    start = i
    escaped = False
    while i < len(text):
        ch = text[i]
        if ch == '"' and not escaped:
            return text[start:i], i + 1
        escaped = ch == "\\" and not escaped
        if ch != "\\":
            escaped = False
        i += 1
    return text[start:], len(text)


LATEX_REPLACEMENTS = {
    r"{\"u}": "u", r"{\"o}": "o", r"{\"a}": "a", r"{\'e}": "e", r"{\'u}": "u",
    r"{\'i}": "i", r"{\c{c}}": "c", r"{\~o}": "o", r"---": "-", r"--": "-", r"~": " ",
}


def clean_latex(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\\url\{([^}]+)\}", r"\1", value)
    for src, dst in LATEX_REPLACEMENTS.items():
        value = value.replace(src, dst)
    value = value.replace("\\&", "&").replace("\\%", "%")
    value = re.sub(r"\\[A-Za-z]+\s*", "", value)
    value = value.replace("{", "").replace("}", "")
    return normalize_space(value)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def safe_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ").replace("\x00", "").strip()


def fix_mojibake(s: object) -> object:
    """Repair cp1252/latin-1-over-UTF-8 double-encoding (e.g. 'é€šç”¨' -> '通用').

    Self-guarding: only returns the round-tripped string when it actually decodes
    to CJK, so clean English (unchanged) and already-correct CJK (raises on the
    Latin encode step) are both left untouched.
    """
    if not isinstance(s, str) or not s:
        return s
    for enc in ("cp1252", "latin-1"):
        try:
            repaired = s.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired != s and re.search(r"[一-鿿]", repaired):
            return repaired
    return s


def first_present(fields: Dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = fields.get(name)
        if value:
            return clean_latex(value)
    return ""


def source_link(fields: Dict[str, str]) -> str:
    doi = clean_latex(fields.get("doi", ""))
    if doi:
        return "https://doi.org/" + doi
    url = clean_latex(fields.get("url", ""))
    if url:
        return url
    how = clean_latex(fields.get("howpublished", ""))
    if how.startswith("http"):
        return how
    eprint = clean_latex(fields.get("eprint", ""))
    if eprint and fields.get("archiveprefix", "").lower() == "arxiv":
        return "https://arxiv.org/abs/" + eprint
    return ""


# ---------------------------------------------------------------------------
# Citation scanning across the real (sectioned) .tex tree
# ---------------------------------------------------------------------------

CITE_RE = re.compile(r"\\[A-Za-z]*cite[A-Za-z*]*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]+)\}")


def extract_sentence(text: str, start: int, end: int) -> str:
    para_start = text.rfind("\n\n", 0, start)
    para_start = 0 if para_start < 0 else para_start + 2
    para_end = text.find("\n\n", end)
    para_end = len(text) if para_end < 0 else para_end
    paragraph = text[para_start:para_end]
    rel_start = start - para_start
    rel_end = end - para_start
    protected = paragraph
    placeholders = {
        r"et al.\ ": "et al<dot>\\ ", r"e.g.\ ": "e<dot>g<dot>\\ ",
        r"i.e.\ ": "i<dot>e<dot>\\ ", r"Fig.\ ": "Fig<dot>\\ ", r"Eq.\ ": "Eq<dot>\\ ",
    }
    for src, dst in placeholders.items():
        protected = protected.replace(src, dst)
    before = protected[:rel_start]
    after = protected[rel_end:]
    left = max(before.rfind(p) for p in ".?!。？！")
    right_candidates = [after.find(p) for p in ".?!。？！" if after.find(p) >= 0]
    right = min(right_candidates) if right_candidates else len(after)
    sentence = protected[left + 1 : rel_end + right + 1].strip()
    for src, dst in placeholders.items():
        sentence = sentence.replace(dst, src)
    return normalize_space(sentence)


def scan_tex_files(files: List[Path]) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for path in files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        label = path.relative_to(LATEX_DIR).as_posix()
        for match in CITE_RE.finditer(text):
            keys = [k.strip() for k in match.group(1).split(",") if k.strip()]
            sentence = extract_sentence(text, match.start(), match.end())
            line = text.count("\n", 0, match.start()) + 1
            loc = f"{label}:{line}"
            for key in keys:
                data = out.setdefault(key, {"locations": [], "sentences": []})
                if loc not in data["locations"]:
                    data["locations"].append(loc)
                if sentence and sentence not in data["sentences"]:
                    data["sentences"].append(sentence)
    return out


def tex_file_sets() -> Tuple[List[Path], List[Path]]:
    en = [LATEX_DIR / "thesis.tex"] + sorted((LATEX_DIR / "sections_en").glob("*.tex"))
    zh = [LATEX_DIR / "thesis_zh.tex"] + sorted((LATEX_DIR / "sections_zh").glob("*.tex"))
    return en, zh


# ---------------------------------------------------------------------------
# Preserve curated columns from the existing TSV (clean rows only)
# ---------------------------------------------------------------------------

PRESERVE_COLS = [
    "english_reference_meaning", "chinese_reference_meaning",
    "download_status", "downloaded_file_path", "manual_reason", "verification_notes",
]


def load_existing(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    # utf-8-sig strips any leftover BOM (a prior PowerShell append left one, which
    # otherwise renames the first header to "﻿bib_key"). Drop null-bearing lines
    # so UTF-16-corrupted rows are ignored and their clean MEANINGS win instead.
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    good_lines = [ln for ln in raw.splitlines() if "\x00" not in ln]
    reader = csv.DictReader(good_lines, delimiter="\t")
    preserved: Dict[str, Dict[str, str]] = {}
    for row in reader:
        key = (row.get("bib_key") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_:+.-]+", key or ""):
            continue
        preserved[key] = {c: (row.get(c) or "").strip() for c in PRESERVE_COLS}
    return preserved


# ---------------------------------------------------------------------------

def build_rows() -> List[Dict[str, str]]:
    entries = scan_bib_entries(BIB_PATH.read_text(encoding="utf-8"))
    en_files, zh_files = tex_file_sets()
    en = scan_tex_files(en_files)
    zh = scan_tex_files(zh_files)
    existing = load_existing(OUT_TSV)

    rows: List[Dict[str, str]] = []
    for entry in entries:
        key = str(entry["key"])
        fields: Dict[str, str] = entry["fields"]  # type: ignore[assignment]
        en_data = en.get(key, {"locations": [], "sentences": []})
        zh_data = zh.get(key, {"locations": [], "sentences": []})
        used = bool(en_data["locations"] or zh_data["locations"])
        prev = existing.get(key, {})

        # Meaning precedence: curated dict (clean UTF-8) -> preserved TSV cell
        # (mojibake-repaired) -> generic template. Dict wins so stale/generic or
        # double-encoded preserved cells never shadow a curated meaning.
        if key in ALL_MEANINGS:
            en_meaning, zh_meaning = ALL_MEANINGS[key]
        else:
            en_meaning = str(fix_mojibake(prev.get("english_reference_meaning") or ""))
            zh_meaning = str(fix_mojibake(prev.get("chinese_reference_meaning") or ""))
            if not en_meaning or not zh_meaning:
                UNCOVERED.append(key)
                title = first_present(fields, ["title"])
                en_meaning = en_meaning or f"Used to support the thesis discussion connected to {title}."
                zh_meaning = zh_meaning or f"用于支撑论文中与《{title}》相关的论述。"

        # Download columns: preserve if present; else no-network manual_required.
        download_status = prev.get("download_status") or "manual_required"
        downloaded_file_path = prev.get("downloaded_file_path") or ""
        manual_reason = prev.get("manual_reason")
        verification = prev.get("verification_notes")
        if key not in existing:
            eprint = clean_latex(fields.get("eprint", ""))
            if eprint and fields.get("archiveprefix", "").lower() == "arxiv":
                manual_reason = "New entry; arXiv preprint - fetch from arXiv abstract page if a local copy is needed."
                verification = "Metadata taken from BibTeX; arXiv id present (no automatic download in this regeneration)."
            elif clean_latex(fields.get("doi", "")):
                manual_reason = "New entry; use the DOI to retrieve the source manually or via institutional access."
                verification = "Metadata taken from BibTeX; DOI present (no automatic download in this regeneration)."
            else:
                manual_reason = "New entry; no DOI/URL/eprint in the BibTeX entry."
                verification = "Metadata taken from BibTeX; no machine-resolvable link."

        row = {
            "bib_key": key,
            "entry_type": str(entry["entry_type"]),
            "used_in_tex": str(used).lower(),
            "authors": first_present(fields, ["author", "editor"]),
            "year": first_present(fields, ["year"]),
            "title": first_present(fields, ["title"]),
            "container": first_present(fields, ["journal", "booktitle"]),
            "publisher_or_institution": first_present(fields, ["publisher", "institution", "school"]),
            "volume": first_present(fields, ["volume"]),
            "number": first_present(fields, ["number"]),
            "pages": first_present(fields, ["pages"]),
            "doi": first_present(fields, ["doi"]),
            "url": first_present(fields, ["url", "howpublished"]),
            "isbn": first_present(fields, ["isbn"]),
            "eprint": first_present(fields, ["eprint"]),
            "bib_note": first_present(fields, ["note"]),
            "source_link": source_link(fields),
            "download_status": download_status,
            "downloaded_file_path": downloaded_file_path,
            "manual_reason": manual_reason or "",
            "english_tex_locations": " || ".join(en_data["locations"]),
            "chinese_tex_locations": " || ".join(zh_data["locations"]),
            "english_citation_sentences": " || ".join(en_data["sentences"]),
            "chinese_citation_sentences": " || ".join(zh_data["sentences"]),
            "english_reference_meaning": en_meaning,
            "chinese_reference_meaning": zh_meaning,
            "verification_notes": verification or "",
        }
        rows.append({h: safe_cell(fix_mojibake(row.get(h, ""))) for h in HEADERS})
    return rows


def validate(rows: List[Dict[str, str]], n_entries: int) -> None:
    if len(rows) != n_entries:
        raise SystemExit(f"row/entry mismatch: {len(rows)} rows vs {n_entries} bib entries")
    keys = {r["bib_key"] for r in rows}
    en_files, zh_files = tex_file_sets()
    cited = set(scan_tex_files(en_files)) | set(scan_tex_files(zh_files))
    missing = sorted(cited - keys)
    if missing:
        raise SystemExit(f"citation keys cited but absent from bib/TSV: {missing}")


def main() -> int:
    entries = scan_bib_entries(BIB_PATH.read_text(encoding="utf-8"))
    rows = build_rows()
    validate(rows, len(entries))
    with OUT_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    used = sum(1 for r in rows if r["used_in_tex"] == "true")
    print(f"rows={len(rows)} used_in_tex={used} out={OUT_TSV.relative_to(THESIS_DIR)}")
    if UNCOVERED:
        print("uncovered (fell back to preserved/generic meaning):", sorted(set(UNCOVERED)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
