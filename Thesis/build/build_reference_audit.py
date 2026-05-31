#!/usr/bin/env python3
"""Build a TSV audit twin for thesisreferences.bib.

The script keeps the original BibTeX file unchanged, extracts citation context
from the English and Chinese thesis files, and downloads only clearly legal open
sources into reference_sources/.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, urljoin

import requests


ROOT = Path(__file__).resolve().parent
BIB_PATH = ROOT / "thesisreferences.bib"
EN_TEX = ROOT / "thesis.tex"
ZH_TEX = ROOT / "thesis_zh.tex"
OUT_TSV = ROOT / "thesisreferences_audit.tsv"
SOURCE_DIR = ROOT / "reference_sources"

HEADERS = [
    "bib_key",
    "entry_type",
    "used_in_tex",
    "authors",
    "year",
    "title",
    "container",
    "publisher_or_institution",
    "volume",
    "number",
    "pages",
    "doi",
    "url",
    "isbn",
    "eprint",
    "bib_note",
    "source_link",
    "download_status",
    "downloaded_file_path",
    "manual_reason",
    "english_tex_locations",
    "chinese_tex_locations",
    "english_citation_sentences",
    "chinese_citation_sentences",
    "english_reference_meaning",
    "chinese_reference_meaning",
    "verification_notes",
]

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (compatible; Mie-Postprocessing-thesis-reference-audit/1.0; legal-open-source-check)",
        "Accept": "application/pdf,text/html,application/xhtml+xml,*/*",
    }
)

LEGAL_OPEN_URL_OVERRIDES: Dict[str, List[Tuple[str, str]]] = {
    # Official author page for the first-edition draft matching the cited 2011 Springer book.
    "Szeliski2010Vision": [
        ("https://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf", "pdf")
    ],
    # Public article PDFs exposed by the article pages / repository pages.
    "Schindelin2012Fiji": [
        ("https://pdfs.semanticscholar.org/dea5/dcac6fe02b4c3f81371e5da5be9f99832ba2.pdf", "pdf")
    ],
    "VanDerWalt2014ScikitImage": [
        ("https://peerj.com/articles/453.pdf?download=1", "pdf")
    ],
    "ThielickeSonntag2021PIVlab": [
        (
            "https://openresearchsoftware.metajnl.com/articles/334/files/submission/proof/334-1-4984-1-10-20210531.pdf",
            "pdf",
        )
    ],
}


MEANINGS: Dict[str, Tuple[str, str]] = {
    "HeywoodICE": (
        "General internal-combustion-engine reference; specifically cited for the slider-crank piston-kinematics idealisation used in the prototype wall-impingement geometry.",
        "通用内燃机基础参考；本文具体用于原型撞壁几何中活塞滑块--曲柄运动学近似的依据。",
    ),
    "Linne2013SprayImaging": (
        "Supports the limits and usefulness of optical spray imaging and Mie-scattering diagnostics for extracting liquid-envelope observables.",
        "用于支撑光学喷雾成像和 Mie 散射诊断的适用性及局限，说明液相包络只能作为操作性观测量解释。",
    ),
    "BohrenHuffman2004": (
        "Provides the scattering-physics basis for explaining why Mie intensity is not a direct liquid-volume-fraction measurement.",
        "用于说明散射物理基础，支撑 Mie 强度不能直接等同于液体体积分数的论述。",
    ),
    "LefebvreMcDonell2017": (
        "Supports the use of penetration, cone angle, projected area, and boundary location as practical spray descriptors.",
        "用于支持把穿深、锥角、投影面积和边界位置作为喷雾工程描述量的做法。",
    ),
    "Dent1971Penetration": (
        "Anchors the thesis use of spray penetration as a standard diesel-spray comparison metric.",
        "用于支撑柴油喷雾研究中以穿深作为标准比较指标的传统依据。",
    ),
    "HiroyasuArai1990": (
        "Provides the classical two-stage diesel-spray scaling law used as a physical baseline and as motivation for the surrogate scaling group.",
        "提供经典两阶段柴油喷雾缩放关系，作为物理基线并支撑代理模型中的缩放群构造。",
    ),
    "ZhouLiYi2021": (
        "Supports the claim that start-of-injection transients can deviate from a rigid classical penetration law during short pilot injections.",
        "用于说明短预喷中的起始注入瞬态会偏离单一经典穿深定律。",
    ),
    "ZhouLiLaiWei2019": (
        "Supports the discussion that end-of-injection penetration behavior changes time variables and regimes after injection stops.",
        "用于支撑结束注入后穿深阶段和时间变量会发生变化的论述。",
    ),
    "NaberSiebers1996": (
        "Supports gas-density and vaporization effects on diesel-spray penetration and the use of non-reacting penetration as a conservative screening quantity.",
        "用于支撑环境密度和蒸发对穿深的影响，以及非反应穿深作为保守筛查量的合理性。",
    ),
    "Pratt1987CircleFit": (
        "Supports least-squares circle fitting as a standard basis for manual nozzle-center calibration.",
        "用于支撑手动喷嘴中心标定中的最小二乘圆拟合方法。",
    ),
    "Taubin1991CircleFit": (
        "Supports algebraic curve and circle fitting as a standard approach for noisy image-coordinate calibration.",
        "用于支撑在有噪声图像坐标中进行圆/曲线拟合标定的标准性。",
    ),
    "GonzalezWoodsDIP": (
        "Supports the image-processing choices: log-domain preprocessing, gradients, thresholding, morphology, and boundary extraction.",
        "用于支撑对数预处理、梯度算子、阈值、形态学和边界提取等图像处理步骤。",
    ),
    "Szeliski2010Vision": (
        "Supports computer-vision operations such as polar resampling, inverse warping, image primitives, and geometric remapping.",
        "用于支撑极坐标重采样、逆映射、图像处理原语和几何重映射等计算机视觉操作。",
    ),
    "CooleyTukey1965FFT": (
        "Supports using Fourier harmonic phase to estimate the rotational offset of repeated multi-hole plume patterns.",
        "用于支撑通过傅里叶谐波相位估计多孔喷雾重复结构的旋转偏移。",
    ),
    "Huber1964": (
        "Supports robust fitting with Huber loss for noisy or partially contaminated penetration traces.",
        "用于支撑在噪声或局部污染的穿深轨迹上使用 Huber 损失进行稳健拟合。",
    ),
    "OppenheimSchaferStockham1968": (
        "Supports the homomorphic-filtering motivation behind log-ratio preprocessing for multiplicative illumination variation.",
        "用于支撑对数比预处理的同态滤波动机，即将乘性照明变化转为更易处理的加性成分。",
    ),
    "ZackRogersLatt1977": (
        "Supports automatic triangle thresholding as part of the segmentation and occupancy pipeline.",
        "用于支撑分割和占用分析流程中的自动三角阈值方法。",
    ),
    "MuenchLeipertz1992": (
        "Supports using two-dimensional Mie-scattering images to report practical jet-tip penetration and cone-angle quantities.",
        "用于支撑二维 Mie 散射图像可用于报告射流尖端穿深和锥角等实用量。",
    ),
    "JohnsonNaberLee2012": (
        "Supports the caution that threshold-based cone-angle and boundary estimates are sensitive to the chosen image metric.",
        "用于说明阈值式锥角和边界估计对图像度量选择敏感，需要谨慎解释。",
    ),
    "MagnottiGenzale2013": (
        "Supports the warning that different liquid-penetration metrics can lead to different model-validation conclusions.",
        "用于支撑不同液相穿深定义会导致不同验证结论的警示。",
    ),
    "JungManinSkeenPickett2015": (
        "Supports the importance of transient spreading angle and penetration measurements in diesel spray studies.",
        "用于支撑柴油喷雾研究中瞬态扩散角和穿深测量的重要性。",
    ),
    "NocedalWright2006": (
        "Supports nonlinear least-squares optimization and robust parameter fitting in the reduced-order penetration model.",
        "用于支撑降阶穿深模型中的非线性最小二乘优化和参数拟合。",
    ),
    "NickollsBuckGarlandSkadron2008": (
        "Supports the computational argument that dense image operations are well suited to GPU/CUDA acceleration.",
        "用于支撑稠密图像运算适合 GPU/CUDA 并行加速的计算论证。",
    ),
    "Tobin1958": (
        "Supports the statistical analogy between field-of-view-limited penetration traces and limited dependent variables.",
        "用于支撑有限视场穿深轨迹与受限因变量问题之间的统计类比。",
    ),
    "BuckleyJames1979": (
        "Supports the censored-regression analogy used to interpret truncated late-time penetration observations.",
        "用于支撑将后期穿深截断观测理解为删失回归问题的统计类比。",
    ),
    "HintonVinyalsDean2015": (
        "Supports knowledge distillation as the general teacher-student mechanism used in the refinement stage.",
        "用于支撑精化阶段采用教师-学生知识蒸馏机制的基本方法论。",
    ),
    "TarvainenValpola2017": (
        "Supports teacher-student learning when labels are limited, noisy, or incomplete.",
        "用于支撑在标签有限、噪声较大或不完整时使用教师-学生学习框架。",
    ),
    "MopuriGrahnSedarskyHyvonen2024": (
        "Supports the multi-hole spray context and the need to analyze isolated plume shape and penetration consistently.",
        "用于支撑多孔喷雾背景，以及对独立 plume 形状和穿深进行一致分析的必要性。",
    ),
    "JinKimKakamiNishidaOgataLuo2020": (
        "Supports the point that multi-hole injector behavior is not a trivial repetition of single-hole spray behavior.",
        "用于支撑多孔喷嘴行为不能简单视为单孔喷雾重复叠加的论点。",
    ),
    "imo2023ghgstrategy": (
        "Supports the maritime decarbonization context and 2030 targets motivating alternative-fuel marine engines.",
        "用于支撑海运减排背景和 2030 目标，从而引出替代燃料船用发动机需求。",
    ),
    "classnk2025alternativefuels": (
        "Supports the claim that alternative fuels have moved into concrete vessel and orderbook decisions.",
        "用于支撑替代燃料已经进入实际船舶和订单决策阶段的行业背景。",
    ),
    "wingd_xdf_manual_2021": (
        "Supports the description of WinGD X-DF operation and pilot-fuel ignition in low-pressure gas mode.",
        "用于支撑 WinGD X-DF 低压燃气运行和 pilot fuel 点火机制的描述。",
    ),
    "wingd_lng_faq_2025": (
        "Supports the practical LNG dual-fuel engine context and fallback/pilot-fuel operation.",
        "用于支撑 LNG 双燃料发动机的实际运行背景、pilot fuel 和模式切换说明。",
    ),
    "classnk_methanol_2024": (
        "Supports the discussion that methanol concepts still require ignition support or combustion-system adaptation.",
        "用于支撑甲醇燃料方案仍需要点火支持或燃烧系统适配的论述。",
    ),
    "wingd_ammonia_faq_2025": (
        "Supports the discussion that ammonia dual-fuel concepts require pilot fuel because ammonia is difficult to auto-ignite.",
        "用于支撑氨双燃料概念因自燃困难而仍需 pilot fuel 的论述。",
    ),
    "ecn_liquid_penetration": (
        "Supports the ECN definition and method-dependence of liquid-penetration measurements.",
        "用于支撑 ECN 对液相穿深定义及其方法依赖性的说明。",
    ),
    "ecn_liquid_penetration_accessory": (
        "Supports the ECN caution that optical liquid boundaries depend on diagnostic and post-processing definitions.",
        "用于支撑 ECN 关于光学液相边界依赖诊断和后处理定义的警示。",
    ),
    "liu2020sprayreview": (
        "Supports the review of phenomenological spray-penetration models and their limitations for advanced injection strategies.",
        "用于支撑现象学穿深模型综述及其在先进喷射策略下的局限。",
    ),
    "peraza2022swi": (
        "Supports the physical importance of spray-wall interaction and penetration-to-wall-distance reasoning.",
        "用于支撑喷雾撞壁问题的重要性，以及以穿深和壁距关系进行风险判断的思路。",
    ),
    "liu2024marine_impingement": (
        "Supports the marine-engine wall-impingement context and the role of optical diagnostics under large-bore conditions.",
        "用于支撑船用发动机撞壁背景，以及大缸径条件下光学诊断的重要性。",
    ),
    "ahmad2025fueldilution": (
        "Supports the marine-engine fuel-dilution and wall-approach motivation linked to Aalto work.",
        "用于支撑船用发动机燃油稀释和液体燃料接近缸壁问题的 Aalto 背景。",
    ),
    "BishopPRML2006": (
        "Supports the regression, Gaussian likelihood, and probabilistic-learning assumptions behind the surrogate model.",
        "用于支撑代理模型中的回归、高斯似然和概率学习假设。",
    ),
    "KendallGal2017": (
        "Supports heteroscedastic uncertainty estimation for input-dependent predictive variance.",
        "用于支撑输入相关预测方差的异方差不确定性建模。",
    ),
    "Schindelin2012Fiji": (
        "Supports the argument that mature scientific image-processing workflows should be scriptable and inspectable.",
        "用于支撑科学图像处理流程应可脚本化、可检查和可复用的观点。",
    ),
    "VanDerWalt2014ScikitImage": (
        "Supports the open-source, reusable image-processing ecosystem argument.",
        "用于支撑开源、可复用图像处理生态在科学工作流中的作用。",
    ),
    "ThielickeSonntag2021PIVlab": (
        "Supports the comparison to mature MATLAB-based PIV tooling while distinguishing tool maturity from local batch architecture.",
        "用于支撑与成熟 MATLAB PIV 工具的比较，同时区分工具本身和本地批处理架构限制。",
    ),
    "OpenPIV": (
        "Supports the broader context of open, scriptable particle-image and flow-analysis tools.",
        "用于支撑开放、可脚本化的粒子图像和流场分析工具背景。",
    ),
    "CurtisNiemeyerSung2017": (
        "Supports the high computational cost example for reacting diesel-spray simulation.",
        "用于支撑反应柴油喷雾仿真计算成本极高的例子。",
    ),
    "Segatori2023DFILES": (
        "Supports the claim that LES spray studies treat computational cost as a central design constraint.",
        "用于支撑 LES 喷雾研究中计算成本是核心设计约束的论述。",
    ),
    "MeijerECN2012": (
        "Supports ECN benchmark context and cross-facility comparison of spray boundary conditions.",
        "用于支撑 ECN 基准实验背景和跨机构喷雾边界条件比较。",
    ),
    "StumppRicco1996": (
        "Supports the injector-hardware transient discussion for common-rail needle and flow-area dynamics.",
        "用于支撑共轨喷射系统中针阀和有效流通截面瞬态的讨论。",
    ),
    "VazKarathanasis2026": (
        "Supports the feasibility of data-driven neural-network prediction of marine-injector spray macroscopic characteristics.",
        "用于支撑神经网络预测船用喷嘴宏观喷雾特性的可行性。",
    ),
    "KangKang2024KDRegression": (
        "Supports knowledge distillation specifically in regression with insufficient training data.",
        "用于支撑训练数据不足时在回归任务中采用知识蒸馏。",
    ),
    "Baumgarten2006MixFormation": (
        "Supports the first-level wall-impingement criterion that compares spray-tip penetration with wall distance.",
        "用于支撑将喷雾尖端穿深与壁距比较作为一级撞壁筛查准则。",
    ),
    "MorieiraMotaPanao2010": (
        "Supports the broader spray-impingement physics context and the difficulty of scaling droplet-impact knowledge to full sprays.",
        "用于支撑喷雾撞壁物理背景，以及从单液滴碰壁推广到整体喷雾的困难。",
    ),
    "Park2018WallImpingement": (
        "Supports image-based wall-impingement prediction as a lower-cost alternative to full CFD.",
        "用于支撑基于图像分析进行撞壁预测可以作为低成本工程估计路径。",
    ),
    "Karim2015DualFuel": (
        "Foundational textbook reference linking pilot-spray characteristics in dual-fuel diesel-gas engines to ignition-kernel formation and downstream main-fuel flame propagation.",
        "用于支撑在双燃料柴油-气体发动机中，pilot 喷雾特性决定点火核形成与主燃料火焰传播的论述基础。",
    ),
    "WeiGeng2016DualFuelReview": (
        "Peer-reviewed review on how pilot diesel injection parameters (timing, quantity, spray characteristics) affect combustion efficiency and unburned-methane emissions in natural-gas/diesel dual-fuel engines.",
        "用于支撑 pilot 柴油喷射参数（定时、油量、喷雾特性）对天然气/柴油双燃料燃烧效率及未燃甲烷排放的影响。",
    ),
    "Sahoo2009DualFuelReview": (
        "Critical review establishing pilot fuel quantity and spray characteristics as design parameters that influence overall combustion efficiency in dual-fuel gas-diesel engines.",
        "用于支撑 pilot 燃油量与喷雾特性是双燃料燃烧效率的关键设计变量这一综述性结论。",
    ),
}


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
        key, fields_text = body.split(",", 1)
        entries.append(
            {
                "entry_type": entry_type,
                "key": key.strip(),
                "fields": parse_fields(fields_text),
            }
        )
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
    r"{\"u}": "u",
    r"{\"o}": "o",
    r"{\"a}": "a",
    r"{\'e}": "e",
    r"{\'u}": "u",
    r"{\'i}": "i",
    r"{\c{c}}": "c",
    r"{\~o}": "o",
    r"---": "-",
    r"--": "-",
    r"~": " ",
}


def clean_latex(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\\url\{([^}]+)\}", r"\1", value)
    for src, dst in LATEX_REPLACEMENTS.items():
        value = value.replace(src, dst)
    value = value.replace("\\&", "&")
    value = value.replace("\\%", "%")
    value = re.sub(r"\\[A-Za-z]+\s*", "", value)
    value = value.replace("{", "").replace("}", "")
    return normalize_space(value)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def safe_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def citation_occurrences(path: Path) -> Dict[str, Dict[str, List[str]]]:
    text = path.read_text(encoding="utf-8")
    out: Dict[str, Dict[str, List[str]]] = {}
    cite_re = re.compile(r"\\[A-Za-z]*cite[A-Za-z*]*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]+)\}")
    for match in cite_re.finditer(text):
        keys = [k.strip() for k in match.group(1).split(",") if k.strip()]
        sentence = extract_sentence(text, match.start(), match.end())
        line = text.count("\n", 0, match.start()) + 1
        loc = f"{path.name}:{line}"
        for key in keys:
            data = out.setdefault(key, {"locations": [], "sentences": []})
            if loc not in data["locations"]:
                data["locations"].append(loc)
            if sentence not in data["sentences"]:
                data["sentences"].append(sentence)
    return out


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
        r"et al.\ ": "et al<dot>\\ ",
        r"e.g.\ ": "e<dot>g<dot>\\ ",
        r"i.e.\ ": "i<dot>e<dot>\\ ",
        r"Fig.\ ": "Fig<dot>\\ ",
        r"Eq.\ ": "Eq<dot>\\ ",
    }
    for src, dst in placeholders.items():
        protected = protected.replace(src, dst)

    before = protected[:rel_start]
    after = protected[rel_end:]
    left_candidates = [before.rfind(p) for p in ".?!。？！"]
    left = max(left_candidates)
    right_candidates = [after.find(p) for p in ".?!。？！" if after.find(p) >= 0]
    right = min(right_candidates) if right_candidates else len(after)
    sentence = protected[left + 1 : rel_end + right + 1].strip()
    for src, dst in placeholders.items():
        sentence = sentence.replace(dst, src)
    return normalize_space(sentence)


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
    return ""


def download_for_entry(key: str, fields: Dict[str, str]) -> Tuple[str, str, str, str]:
    SOURCE_DIR.mkdir(exist_ok=True)
    doi = clean_latex(fields.get("doi", ""))
    eprint = clean_latex(fields.get("eprint", ""))
    url = clean_latex(fields.get("url", "")) or clean_latex(fields.get("howpublished", ""))

    candidates: List[Tuple[str, str]] = []
    candidates.extend(LEGAL_OPEN_URL_OVERRIDES.get(key, []))
    if url:
        candidates.extend(url_candidates(url))
    if eprint and ("arxiv" in doi.lower() or fields.get("archiveprefix", "").lower() == "arxiv"):
        candidates.append((f"https://arxiv.org/pdf/{eprint}", "pdf"))
    elif doi.lower().startswith("10.48550/arxiv."):
        arxiv_id = re.split(r"arxiv\.", doi, flags=re.I)[-1]
        candidates.append((f"https://arxiv.org/pdf/{arxiv_id}", "pdf"))
    if doi:
        candidates.extend(openalex_candidates(doi))

    seen = set()
    for candidate_url, expected in candidates:
        if not candidate_url or candidate_url in seen:
            continue
        seen.add(candidate_url)
        result = try_download(key, candidate_url, expected)
        if result:
            path, status_note = result
            return "downloaded", str(path.relative_to(ROOT).as_posix()), "", status_note

    if url and is_public_html_url(url):
        result = try_download(key, url, "html")
        if result:
            path, status_note = result
            return "downloaded_html", str(path.relative_to(ROOT).as_posix()), "", status_note

    if doi or url:
        return (
            "manual_required",
            "",
            "No clearly legal open PDF/HTML download was found automatically; use the DOI/URL manually or through institutional access if needed.",
            "Checked direct URL, arXiv when applicable, and OpenAlex OA metadata.",
        )
    return (
        "manual_required",
        "",
        "No DOI or URL is present in the BibTeX entry; manual lookup is required.",
        "No machine-resolvable link in BibTeX.",
    )


def url_candidates(url: str) -> List[Tuple[str, str]]:
    out = []
    lower = url.lower()
    if lower.endswith(".pdf"):
        out.append((url, "pdf"))
    else:
        out.append((url, "html"))
    if not lower.endswith(".pdf"):
        try:
            html = SESSION.get(url, timeout=20).text
            pdf_like = re.findall(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', html, flags=re.I)
            pdf_like.extend(
                re.findall(r'href=["\']([^"\']*/bitstreams/[^"\']*/download)["\']', html, flags=re.I)
            )
            for href in pdf_like:
                out.insert(0, (urljoin(url, href), "pdf"))
        except Exception:
            pass
    return out


def openalex_candidates(doi: str) -> List[Tuple[str, str]]:
    url = f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='')}"
    try:
        response = SESSION.get(url, timeout=20)
        if response.status_code != 200:
            return []
        data = response.json()
    except Exception:
        return []

    candidates: List[Tuple[str, str]] = []
    locations = []
    if data.get("primary_location"):
        locations.append(data["primary_location"])
    locations.extend(data.get("locations") or [])
    for loc in locations:
        if not loc:
            continue
        pdf_url = loc.get("pdf_url")
        if pdf_url:
            candidates.append((pdf_url, "pdf"))
    oa_url = data.get("open_access", {}).get("oa_url")
    if oa_url and str(oa_url).lower().endswith(".pdf"):
        candidates.append((oa_url, "pdf"))
    return candidates


def is_public_html_url(url: str) -> bool:
    return url.lower().startswith(("http://", "https://")) and not url.lower().endswith(".pdf")


def try_download(key: str, url: str, expected: str) -> Optional[Tuple[Path, str]]:
    try:
        response = SESSION.get(url, timeout=40, allow_redirects=True)
    except Exception as exc:
        return None
    if response.status_code >= 400:
        return None

    content_type = response.headers.get("content-type", "").lower()
    content = response.content
    if expected == "pdf" or content[:4] == b"%PDF":
        if content[:4] != b"%PDF":
            return None
        path = SOURCE_DIR / f"{key}.pdf"
        path.write_bytes(content)
        return path, f"Downloaded legal open PDF from {response.url}"

    if expected == "html" and ("text/html" in content_type or content.strip().startswith(b"<!DOCTYPE") or b"<html" in content[:500].lower()):
        path = SOURCE_DIR / f"{key}.html"
        path.write_text(response.text, encoding=response.encoding or "utf-8", errors="replace")
        return path, f"Saved public HTML page from {response.url}"
    return None


def build_rows() -> List[Dict[str, str]]:
    entries = scan_bib_entries(BIB_PATH.read_text(encoding="utf-8"))
    en = citation_occurrences(EN_TEX)
    zh = citation_occurrences(ZH_TEX)
    rows: List[Dict[str, str]] = []

    for entry in entries:
        key = str(entry["key"])
        fields: Dict[str, str] = entry["fields"]  # type: ignore[assignment]
        en_data = en.get(key, {"locations": [], "sentences": []})
        zh_data = zh.get(key, {"locations": [], "sentences": []})
        used = bool(en_data["locations"] or zh_data["locations"])
        status, downloaded_path, manual_reason, verification = download_for_entry(key, fields)
        en_meaning, zh_meaning = MEANINGS.get(
            key,
            (
                f"Used to support the thesis discussion connected to {first_present(fields, ['title'])}.",
                f"用于支撑论文中与《{first_present(fields, ['title'])}》相关的论述。",
            ),
        )
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
            "download_status": status,
            "downloaded_file_path": downloaded_path,
            "manual_reason": manual_reason,
            "english_tex_locations": " || ".join(en_data["locations"]),
            "chinese_tex_locations": " || ".join(zh_data["locations"]),
            "english_citation_sentences": " || ".join(en_data["sentences"]),
            "chinese_citation_sentences": " || ".join(zh_data["sentences"]),
            "english_reference_meaning": en_meaning,
            "chinese_reference_meaning": zh_meaning,
            "verification_notes": verification,
        }
        rows.append({h: safe_cell(row.get(h, "")) for h in HEADERS})
    return rows


def validate(rows: List[Dict[str, str]]) -> None:
    bib_keys = {row["bib_key"] for row in rows}
    cited_keys = set(citation_occurrences(EN_TEX)) | set(citation_occurrences(ZH_TEX))
    missing = sorted(cited_keys - bib_keys)
    if missing:
        raise SystemExit(f"citation keys missing from TSV: {missing}")
    if len(rows) != 63:
        raise SystemExit(f"expected 63 BibTeX rows, got {len(rows)}")
    heywood = next(row for row in rows if row["bib_key"] == "HeywoodICE")
    if heywood["used_in_tex"] != "true":
        raise SystemExit("HeywoodICE should be marked used_in_tex=true (cited in piston-kinematics section)")
    for row in rows:
        path = row["downloaded_file_path"]
        if path and not (ROOT / path).exists():
            raise SystemExit(f"downloaded file missing: {path}")


def main() -> int:
    SOURCE_DIR.mkdir(exist_ok=True)
    for stale in SOURCE_DIR.iterdir():
        if stale.is_file():
            stale.unlink()
    rows = build_rows()
    validate(rows)
    with OUT_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "rows": len(rows),
        "downloaded": sum(1 for row in rows if row["downloaded_file_path"]),
        "manual_required": sum(1 for row in rows if row["download_status"] == "manual_required"),
        "output": str(OUT_TSV.relative_to(ROOT)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
