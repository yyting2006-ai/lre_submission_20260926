#!/usr/bin/env python3
"""Align all 572 GF0025 grammar points to the three teaching handbooks.

The handbooks explicitly state that standard points may be split or merged.
Consequently this alignment never projects a standard ordinal onto a handbook
ordinal or page.  It ranks real handbook pages using only page-local evidence:
normalised grammar labels, section labels, literal cues recovered from the
legacy draft regex, and (when present) the printed standard identifier.

The released CSV is copyright-safe: it stores identifiers, page numbers,
short matched grammar keywords, numeric scores, and a SHA-256 digest of the
selected page.  It never stores handbook prose or snippets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_DIR.parent
DEFAULT_POINTS = PROJECT_DIR / "kg" / "standard_grammar_points_gf0025_v2.csv"
DEFAULT_PAGES = WORKSPACE / "grammar_teaching_dataset" / "data" / "source_pages_raw.jsonl"
DEFAULT_CORRECTIONS = PROJECT_DIR / "kg" / "standard_catalog_corrections_v3.csv"
DEFAULT_ALIGNMENT = PROJECT_DIR / "kg" / "standard_handbook_alignment_v3.csv"
DEFAULT_COVERAGE = PROJECT_DIR / "kg" / "full_inventory_handbook_coverage_v3.json"

SOURCE_BY_LEVEL = {
    "1": "SRC-GRAMMAR-HANDBOOK-ELEM-2022",
    "2": "SRC-GRAMMAR-HANDBOOK-ELEM-2022",
    "3": "SRC-GRAMMAR-HANDBOOK-ELEM-2022",
    "4": "SRC-GRAMMAR-HANDBOOK-INTER-2022",
    "5": "SRC-GRAMMAR-HANDBOOK-INTER-2022",
    "6": "SRC-GRAMMAR-HANDBOOK-INTER-2022",
    "7-9": "SRC-GRAMMAR-HANDBOOK-ADV-2022",
}

EXPECTED_POINT_COUNT = 572
OUTPUT_FIELDS = [
    "standard_point_id",
    "handbook_source_id",
    "primary_page",
    "candidate_pages",
    "matched_source_derived_keywords",
    "alignment_score",
    "score_margin",
    "alignment_method",
    "alignment_status",
    "confidence",
    "page_text_sha256",
]

CONTENT_MARKERS = (
    "基本语义及用法",
    "典型例句和对话",
    "结构特点",
    "补充例句",
    "小提示",
)
NAVIGATION_MARKERS = ("目录", "索引", "语法术语缩略形式一览表")


@dataclass(frozen=True)
class Cue:
    text: str
    compact: str
    origin: str
    weight: float


@dataclass
class PageEvidence:
    page: int
    score: float
    matched: list[Cue]
    official_id_match: bool
    full_grammar_match: bool
    heading_match: bool
    content_page: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--catalog-corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--alignment-out", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--coverage-out", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--low-confidence-limit", type=int, default=30)
    return parser.parse_args()


def normalise_display(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text).strip()


def compact(value: str) -> str:
    """Normalise OCR variants while retaining Chinese, letters and digits."""

    text = normalise_display(value)
    replacements = {
        "七一九": "七九",
        "七—九": "七九",
        "七–九": "七九",
        "七-九": "七九",
        "七至九": "七九",
        "×": "x",
        "〇": "0",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return "".join(char.lower() for char in text if char.isalnum() or "\u4e00" <= char <= "\u9fff")


def page_hash(text: str) -> str:
    return hashlib.sha256(normalise_display(text).encode("utf-8")).hexdigest()


def read_points(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        points = list(csv.DictReader(handle))
    if len(points) != EXPECTED_POINT_COUNT:
        raise ValueError(f"Expected {EXPECTED_POINT_COUNT} points, found {len(points)} in {path}")
    ids = [row["standard_point_id"] for row in points]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate standard_point_id values")
    unknown = sorted({row["level"] for row in points} - set(SOURCE_BY_LEVEL))
    if unknown:
        raise ValueError(f"Unsupported levels: {unknown}")
    return points


def apply_catalog_corrections(
    points: list[dict[str, str]], path: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing verified catalog-correction file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        corrections = list(csv.DictReader(handle))
    by_id = {point["standard_point_id"]: dict(point) for point in points}
    seen: set[tuple[str, str]] = set()
    applied: list[dict[str, str]] = []
    for correction in corrections:
        point_id = correction.get("standard_point_id", "")
        field_name = correction.get("field_name", "")
        key = (point_id, field_name)
        if key in seen:
            raise ValueError(f"Duplicate catalog correction: {key}")
        seen.add(key)
        if point_id not in by_id:
            raise ValueError(f"Correction references unknown point: {point_id}")
        if field_name not in by_id[point_id]:
            raise ValueError(f"Correction references unknown field {field_name!r} for {point_id}")
        original = correction.get("original_value", "")
        if by_id[point_id][field_name] != original:
            raise ValueError(
                f"Correction original mismatch for {point_id}.{field_name}: "
                f"catalog={by_id[point_id][field_name]!r}, correction={original!r}"
            )
        if correction.get("verification_status") != "manually_verified_against_official_page":
            raise ValueError(f"Unverified correction for {point_id}.{field_name}")
        corrected = correction.get("corrected_value", "")
        if not corrected:
            raise ValueError(f"Empty corrected value for {point_id}.{field_name}")
        by_id[point_id][field_name] = corrected
        applied.append(dict(correction))
    return [by_id[point["standard_point_id"]] for point in points], applied


def read_handbook_pages(path: Path) -> dict[str, list[dict[str, object]]]:
    requested = set(SOURCE_BY_LEVEL.values())
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            source_id = payload.get("source_id")
            if source_id not in requested:
                continue
            if "page" not in payload or "text" not in payload:
                raise ValueError(f"Missing page/text at JSONL line {line_number}")
            text = normalise_display(str(payload.get("text", "")))
            text_compact = compact(text)
            by_source[source_id].append(
                {
                    "page": int(payload["page"]),
                    "text": text,
                    "compact": text_compact,
                    "hash": page_hash(text),
                    "content": any(marker in text for marker in CONTENT_MARKERS),
                    "navigation": any(compact(marker) in text_compact for marker in NAVIGATION_MARKERS),
                }
            )
    missing = sorted(requested - set(by_source))
    if missing:
        raise ValueError(f"Missing handbook sources: {missing}")
    for source_id, pages in by_source.items():
        pages.sort(key=lambda item: int(item["page"]))
        page_numbers = [int(item["page"]) for item in pages]
        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError(f"Duplicate pages in {source_id}")
        # Derive book-body bounds from source structure, not from a standard
        # ordinal.  The first real entry has both a printed bracketed ID and a
        # semantic-description marker.  The final index is detected in the
        # last quarter of the volume.  This prevents table-of-contents and
        # index continuations from becoming primary evidence pages.
        body_candidates = [
            int(item["page"])
            for item in pages
            if bool(item["content"])
            and re.search(
                r"[【\[]\s*(?:一|二|三|四|五|六|七\s*[一—–\-至]\s*九)\s*\d{1,3}",
                str(item["text"]),
            )
        ]
        if not body_candidates:
            raise ValueError(f"Could not derive handbook body start for {source_id}")
        body_start = min(body_candidates)
        max_page = max(page_numbers)
        index_candidates = [
            int(item["page"])
            for item in pages
            if int(item["page"]) >= math.floor(max_page * 0.75) and "索引" in str(item["compact"])
        ]
        index_start = min(index_candidates) if index_candidates else max_page + 1
        for item in pages:
            page_number = int(item["page"])
            item["body_start"] = body_start
            item["index_start"] = index_start
            item["navigation"] = bool(item["navigation"]) or page_number < body_start or page_number >= index_start
    return dict(by_source)


def split_components(value: str) -> list[str]:
    """Extract short source-derived terms without copying handbook prose."""

    text = normalise_display(value)
    text = re.sub(r"[（(]\s*\d+\s*[）)]", " ", text)
    text = text.replace("⋯⋯", " ").replace("……", " ")
    components = re.split(r"[、，,；;。/|：:\n（）()\"“”'‘’]+", text)
    result: list[str] = []
    for component in components:
        component = component.strip(" +?？！!。.\t")
        if not component:
            continue
        component = re.sub(r"^(或|和|与|以及)", "", component).strip()
        if component and compact(component):
            result.append(component)
    return result


def regex_literals(value: str) -> list[str]:
    if not value:
        return []
    # Literal runs only; regex syntax is not executed and never becomes a rule.
    candidates = re.findall(r"[\u4e00-\u9fff]{1,16}|[A-Za-z]{2,16}", normalise_display(value))
    stop = {"P", "S", "NP", "VP", "Adj", "Adv", "N", "V"}
    return [candidate for candidate in candidates if candidate not in stop]


def cues_for_point(point: dict[str, str]) -> list[Cue]:
    raw: list[tuple[str, str, float]] = []
    grammar = normalise_display(point.get("grammar_point", ""))
    canonical = normalise_display(point.get("canonical_label", ""))
    section = normalise_display(point.get("section_item", ""))
    subcategory = normalise_display(point.get("subcategory", ""))

    if grammar:
        raw.append((grammar, "grammar_full", 25.0))
    if canonical and compact(canonical) != compact(grammar):
        raw.append((canonical, "canonical", 21.0))
    for component in split_components(grammar):
        length = len(compact(component))
        raw.append((component, "grammar_component", 4.0 + min(length, 10) * 1.15))
    if section:
        raw.append((section, "section_item", 8.0 + min(len(compact(section)), 8) * 0.6))
    if subcategory and compact(subcategory) != compact(section):
        raw.append((subcategory, "subcategory", 4.0 + min(len(compact(subcategory)), 8) * 0.35))
    for literal in regex_literals(point.get("legacy_regex", "")):
        raw.append((literal, "legacy_regex_literal", 2.0 + min(len(compact(literal)), 8) * 0.3))

    # Keep the strongest provenance for duplicate normalised cues.
    best: dict[str, Cue] = {}
    for text, origin, weight in raw:
        cue_compact = compact(text)
        if not cue_compact:
            continue
        cue = Cue(text=text, compact=cue_compact, origin=origin, weight=weight)
        if cue_compact not in best or weight > best[cue_compact].weight:
            best[cue_compact] = cue
    return sorted(best.values(), key=lambda cue: (-cue.weight, -len(cue.compact), cue.text))


def official_id_forms(label: str) -> set[str]:
    """Return OCR-tolerant printed-ID forms, never a page/ordinal projection."""

    value = compact(label)
    forms = {value}
    match = re.search(r"(一|二|三|四|五|六|七九)(\d+)$", value)
    if match:
        level, ordinal = match.groups()
        # OCR may retain or drop leading zeroes.  Matching either is page-local
        # evidence; it is not used to calculate a page number.
        forms.add(f"{level}{int(ordinal)}")
        forms.add(f"{level}{int(ordinal):03d}")
    return {form for form in forms if form}


def printed_official_id_match(point: dict[str, str], page_compact: str) -> bool:
    # The advanced handbook reorders, splits, and merges the 7--9 inventory;
    # its local printed numbers are therefore not treated as standard-ID
    # evidence.  For levels 1--6, accept only a digit-bounded ID actually
    # printed on the page.  No ordinal is ever converted into a page.
    if point["level"] == "7-9":
        return False
    return any(re.search(re.escape(form) + r"(?!\d)", page_compact) for form in official_id_forms(point["official_label"]))


def cue_heading_match(cue: Cue, page_text: str) -> bool:
    if len(cue.compact) < 2:
        return False
    for line in page_text.splitlines():
        line_compact = compact(line)
        if cue.compact not in line_compact:
            continue
        # Handbook item headings are short and usually numbered or followed by
        # the semantic-description marker.  This deliberately ignores ordinals.
        if len(line_compact) <= len(cue.compact) + 24 and (
            re.match(r"^\d{1,3}", line_compact) or "词" in line_compact or "句" in line_compact or "结构" in line_compact
        ):
            return True
    cue_position = page_text.find(cue.text)
    if cue_position >= 0:
        window = page_text[cue_position : cue_position + len(cue.text) + 100]
        return "基本语义及用法" in window
    return False


def joint_heading_match(point: dict[str, str], page_text: str) -> bool:
    """Match short grammar cues only when paired with a taxonomy heading."""

    grammar = compact(point.get("grammar_point", ""))
    if not grammar or grammar in {"name", "xname"}:
        return False
    taxonomy = [
        compact(point.get(field, ""))
        for field in ("section_item", "subcategory", "category")
        if compact(point.get(field, ""))
    ]
    for line in page_text.splitlines():
        line_compact = compact(line)
        if grammar not in line_compact:
            continue
        short_heading = len(line_compact) <= len(grammar) + 40 and re.match(r"^\d{1,3}", line_compact)
        taxonomy_match = any(term in line_compact for term in taxonomy if len(term) >= 2)
        colon_ending = bool(re.search(r"[：:]\s*" + re.escape(normalise_display(point.get("grammar_point", ""))) + r"(?:\s|$)", line))
        if short_heading and (taxonomy_match or colon_ending):
            return True
    return False


def score_page(point: dict[str, str], cues: list[Cue], page: dict[str, object]) -> PageEvidence:
    text = str(page["text"])
    text_compact = str(page["compact"])
    content_page = bool(page["content"])
    navigation_page = bool(page["navigation"])
    matched = [cue for cue in cues if cue.compact in text_compact]
    matched_compacts = {cue.compact for cue in matched}
    grammar_compact = compact(point.get("grammar_point", ""))
    full_grammar_match = bool(grammar_compact and grammar_compact in text_compact)
    official_id_match = printed_official_id_match(point, text_compact)

    heading_cues = [
        cue
        for cue in matched
        if cue.origin in {"grammar_full", "canonical", "grammar_component", "section_item"}
        and cue_heading_match(cue, text)
    ]
    joint_heading = joint_heading_match(point, text)
    heading_match = bool(heading_cues) or joint_heading

    score = 0.0
    if content_page:
        score += 2.0
    if navigation_page and not content_page:
        score -= 80.0
    if official_id_match:
        score += 48.0
    if full_grammar_match:
        score += 22.0 + min(len(grammar_compact), 18) * 0.7

    # Multiple matching components are useful for split/merged handbook items.
    component_score = 0.0
    for cue in matched:
        length = len(cue.compact)
        if cue.origin == "grammar_full":
            continue
        if length == 1:
            contribution = min(cue.weight, 1.0)
        elif length == 2:
            contribution = min(cue.weight, 5.0)
        else:
            contribution = cue.weight
        component_score += contribution
    score += min(component_score, 38.0)

    grammar_heading_cues = [
        cue
        for cue in heading_cues
        if cue.origin in {"grammar_full", "canonical", "grammar_component"}
    ]
    if grammar_heading_cues:
        strongest = max(grammar_heading_cues, key=lambda cue: (len(cue.compact), cue.weight))
        score += 42.0 + min(len(strongest.compact), 15) * 1.2
    elif heading_cues:
        strongest = max(heading_cues, key=lambda cue: (len(cue.compact), cue.weight))
        score += 15.0 + min(len(strongest.compact), 15) * 0.8
    elif joint_heading:
        score += 34.0 + min(len(grammar_compact), 8) * 0.8
    if any(cue.origin == "section_item" for cue in matched) and any(
        cue.origin in {"grammar_full", "grammar_component", "canonical"} for cue in matched
    ):
        score += 9.0
    if any(cue.origin == "legacy_regex_literal" for cue in matched):
        score += 1.5

    # A lone one-character match in prose is weak evidence.
    substantive = [cue for cue in matched if len(cue.compact) >= 2]
    if not official_id_match and not heading_match and not substantive:
        score = min(score, 3.0)

    return PageEvidence(
        page=int(page["page"]),
        score=round(max(score, 0.0), 3),
        matched=matched,
        official_id_match=official_id_match,
        full_grammar_match=full_grammar_match,
        heading_match=heading_match,
        content_page=content_page,
    )


def keyword_release_value(evidence: PageEvidence) -> str:
    # Short standard/regex-derived terms only.  No handbook prose or snippets.
    preferred_origins = {
        "grammar_full": 0,
        "canonical": 1,
        "grammar_component": 2,
        "section_item": 3,
        "subcategory": 4,
        "legacy_regex_literal": 5,
    }
    selected: list[str] = []
    seen: set[str] = set()
    for cue in sorted(
        evidence.matched,
        key=lambda cue: (preferred_origins.get(cue.origin, 9), -len(cue.compact), cue.text),
    ):
        # Long enumerations are not useful as release keywords; their shorter
        # components remain available instead.
        if len(cue.text) > 20 or cue.compact in seen:
            continue
        selected.append(cue.text)
        seen.add(cue.compact)
        if len(selected) >= 8 or sum(len(item) for item in selected) >= 72:
            break
    return " | ".join(selected)


def classify_alignment(primary: PageEvidence, margin: float) -> tuple[str, str, str]:
    substantive = [cue for cue in primary.matched if len(cue.compact) >= 2]
    grammar_evidence = [
        cue
        for cue in substantive
        if cue.origin in {"grammar_full", "canonical", "grammar_component"}
    ]
    grammar_evidence_any = [
        cue
        for cue in primary.matched
        if cue.origin in {"grammar_full", "canonical", "grammar_component"}
    ]
    if primary.official_id_match and (primary.heading_match or grammar_evidence_any):
        return "printed_id_plus_lexical", "aligned", "high"
    if primary.official_id_match and len(substantive) >= 2 and primary.score >= 30.0:
        return "printed_id_plus_taxonomy", "aligned", "medium"
    if primary.heading_match and (primary.full_grammar_match or grammar_evidence_any):
        confidence = "high" if primary.score >= 55.0 else "medium"
        return "handbook_heading_lexical", "aligned", confidence
    if primary.full_grammar_match and len(compact(grammar_evidence[0].text if grammar_evidence else "")) >= 2:
        confidence = "medium" if primary.score >= 30.0 else "low"
        return "full_label_lexical", "aligned" if confidence == "medium" else "low_confidence", confidence
    if len(grammar_evidence) >= 2 and primary.score >= 25.0:
        return "multi_cue_lexical", "aligned", "medium"
    if grammar_evidence and primary.score >= 12.0:
        return "single_cue_lexical", "low_confidence", "low"
    if substantive and primary.score >= 8.0:
        return "taxonomy_lexical", "low_confidence", "low"
    return "best_available_page_local_evidence", "low_confidence", "low"


def candidate_page_string(ranked: list[PageEvidence]) -> str:
    if not ranked:
        return ""
    best = ranked[0].score
    # Include near ties and component-specific alternatives, capped for a
    # compact release artifact.  This is important when one standard point was
    # split into several handbook entries.
    chosen: list[PageEvidence] = []
    keyword_coverage: set[str] = set()
    for evidence in ranked:
        if len(chosen) >= 8:
            break
        evidence_keywords = {
            cue.compact
            for cue in evidence.matched
            if cue.origin in {"grammar_full", "canonical", "grammar_component"} and len(cue.compact) >= 2
        }
        near_best = evidence.score >= max(8.0, best * 0.62)
        adds_keyword = bool(evidence_keywords - keyword_coverage) and evidence.score >= 8.0
        if near_best or adds_keyword or not chosen:
            chosen.append(evidence)
            keyword_coverage.update(evidence_keywords)
    return ";".join(str(item.page) for item in chosen)


def toc_reference_evidence(
    point: dict[str, str], cues: list[Cue], pages: list[dict[str, object]]
) -> PageEvidence | None:
    """Resolve an OCR-empty page from a lexical TOC reference.

    This is deliberately not an ordinal projection.  The point's grammar and
    taxonomy cues must match a real handbook contents heading, that heading
    must print a page number, and the PDF-to-print offset must be verified by
    the dominant footer offset across handbook body pages.
    """

    if not pages:
        return None
    body_start = int(pages[0]["body_start"])
    index_start = int(pages[0]["index_start"])
    footer_offsets: Counter[int] = Counter()
    for page in pages:
        page_number = int(page["page"])
        if not (body_start <= page_number < index_start):
            continue
        lines = [line.strip() for line in str(page["text"]).splitlines() if line.strip()]
        if lines and re.fullmatch(r"\d{1,3}", lines[-1]):
            footer_offsets[page_number - int(lines[-1])] += 1
    if not footer_offsets:
        return None
    offset, support = footer_offsets.most_common(1)[0]
    if support < 20:
        return None

    grammar = compact(point.get("grammar_point", ""))
    taxonomy = {
        compact(point.get(field, ""))
        for field in ("section_item", "subcategory", "category")
        if compact(point.get(field, ""))
    }
    if not grammar or grammar in {"name", "xname"}:
        return None
    page_lookup = {int(page["page"]): page for page in pages}
    for navigation_page in pages:
        if int(navigation_page["page"]) >= body_start:
            continue
        lines = [line.strip() for line in str(navigation_page["text"]).splitlines() if line.strip()]
        for index, line in enumerate(lines):
            line_compact = compact(line)
            if grammar not in line_compact or not re.match(r"^\d{1,3}", line_compact):
                continue
            if taxonomy and not any(term in line_compact for term in taxonomy if len(term) >= 2):
                continue
            printed_page: int | None = None
            for following in lines[index + 1 : index + 5]:
                if re.fullmatch(r"\d{1,3}", following):
                    printed_page = int(following)
                    break
                if re.match(r"^\d{1,3}[.、]", following):
                    break
            if printed_page is None:
                continue
            physical_page = printed_page + offset
            target = page_lookup.get(physical_page)
            if target is None or physical_page < body_start or physical_page >= index_start:
                continue
            matched = [cue for cue in cues if cue.compact in line_compact]
            if not matched:
                continue
            return PageEvidence(
                page=physical_page,
                score=58.0,
                matched=matched,
                official_id_match=False,
                full_grammar_match=True,
                heading_match=True,
                content_page=bool(target["content"]),
            )
    return None


def align_point(point: dict[str, str], pages: list[dict[str, object]]) -> tuple[dict[str, str], PageEvidence]:
    cues = cues_for_point(point)
    ranked = sorted(
        (score_page(point, cues, page) for page in pages),
        key=lambda evidence: (-evidence.score, not evidence.content_page, evidence.page),
    )
    if not ranked:
        raise ValueError(f"No pages available for {point['standard_point_id']}")
    primary = ranked[0]
    second_score = ranked[1].score if len(ranked) > 1 else 0.0
    margin = round(primary.score - second_score, 3)
    method, status, confidence = classify_alignment(primary, margin)
    if confidence == "low":
        toc_evidence = toc_reference_evidence(point, cues, pages)
        if toc_evidence is not None:
            ranked = [toc_evidence] + [item for item in ranked if item.page != toc_evidence.page]
            primary = toc_evidence
            second_score = ranked[1].score if len(ranked) > 1 else 0.0
            margin = round(primary.score - second_score, 3)
            method, status, confidence = "toc_reference_verified_offset", "aligned", "medium"
    source_id = SOURCE_BY_LEVEL[point["level"]]
    page_record = next(page for page in pages if int(page["page"]) == primary.page)
    candidate_pages = candidate_page_string(ranked)
    if not candidate_pages:
        raise AssertionError(f"Empty candidate pages for {point['standard_point_id']}")
    output = {
        "standard_point_id": point["standard_point_id"],
        "handbook_source_id": source_id,
        "primary_page": str(primary.page),
        "candidate_pages": candidate_pages,
        "matched_source_derived_keywords": keyword_release_value(primary),
        "alignment_score": f"{primary.score:.3f}",
        "score_margin": f"{margin:.3f}",
        "alignment_method": method,
        "alignment_status": status,
        "confidence": confidence,
        "page_text_sha256": str(page_record["hash"]),
    }
    return output, primary


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    points = read_points(args.points)
    points, applied_corrections = apply_catalog_corrections(points, args.catalog_corrections)
    pages_by_source = read_handbook_pages(args.pages)

    outputs: list[dict[str, str]] = []
    evidence_by_id: dict[str, PageEvidence] = {}
    for point in points:
        source_id = SOURCE_BY_LEVEL[point["level"]]
        output, evidence = align_point(point, pages_by_source[source_id])
        outputs.append(output)
        evidence_by_id[point["standard_point_id"]] = evidence

    if len(outputs) != EXPECTED_POINT_COUNT:
        raise AssertionError(f"Expected {EXPECTED_POINT_COUNT} outputs, found {len(outputs)}")
    if any(not row["primary_page"] or not row["candidate_pages"] or not row["page_text_sha256"] for row in outputs):
        raise AssertionError("All 572 points must have nonempty page alignment and page hash")
    if len({row["standard_point_id"] for row in outputs}) != EXPECTED_POINT_COUNT:
        raise AssertionError("Alignment output contains duplicate or missing standard IDs")

    write_csv(args.alignment_out, outputs)

    status_counts = Counter(row["alignment_status"] for row in outputs)
    confidence_counts = Counter(row["confidence"] for row in outputs)
    method_counts = Counter(row["alignment_method"] for row in outputs)
    source_counts = Counter(row["handbook_source_id"] for row in outputs)
    low_rows = sorted(
        (row for row in outputs if row["confidence"] == "low"),
        key=lambda row: (float(row["alignment_score"]), float(row["score_margin"]), row["standard_point_id"]),
    )
    medium_rows = sorted(
        (row for row in outputs if row["confidence"] == "medium"),
        key=lambda row: (float(row["alignment_score"]), float(row["score_margin"]), row["standard_point_id"]),
    )
    score_values = [float(row["alignment_score"]) for row in outputs]
    margin_values = [float(row["score_margin"]) for row in outputs]

    coverage = {
        "schema_version": "handbook-alignment-v3",
        "alignment_policy": {
            "ordinal_projection_used": False,
            "page_local_lexical_evidence_preferred": True,
            "toc_reference_allowed_only_with_verified_pdf_offset": True,
            "handbook_source_evidence_only": True,
            "split_merge_aware_candidates": True,
            "released_handbook_text_or_snippets": False,
            "low_confidence_is_separately_flagged": True,
        },
        "input": {
            "standard_inventory": str(args.points),
            "handbook_page_corpus": str(args.pages),
            "catalog_corrections": str(args.catalog_corrections),
            "applied_catalog_correction_count": len(applied_corrections),
            "applied_catalog_corrections": [
                {
                    "standard_point_id": correction["standard_point_id"],
                    "field_name": correction["field_name"],
                    "authority_source_id": correction["authority_source_id"],
                    "authority_pdf_page": int(correction["authority_pdf_page"]),
                    "verification_status": correction["verification_status"],
                }
                for correction in applied_corrections
            ],
            "handbook_page_counts": {source: len(pages) for source, pages in sorted(pages_by_source.items())},
        },
        "coverage": {
            "expected_standard_points": EXPECTED_POINT_COUNT,
            "aligned_with_nonempty_primary_page": len(outputs),
            "aligned_with_nonempty_candidate_pages": sum(bool(row["candidate_pages"]) for row in outputs),
            "aligned_with_page_hash": sum(bool(row["page_text_sha256"]) for row in outputs),
            "source_counts": dict(sorted(source_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "confidence_counts": dict(sorted(confidence_counts.items())),
            "method_counts": dict(sorted(method_counts.items())),
        },
        "score_summary": {
            "minimum": min(score_values),
            "maximum": max(score_values),
            "mean": round(sum(score_values) / len(score_values), 3),
            "minimum_margin": min(margin_values),
            "mean_margin": round(sum(margin_values) / len(margin_values), 3),
        },
        "low_confidence_count": len(low_rows),
        "low_confidence_cases": [
            {
                "standard_point_id": row["standard_point_id"],
                "handbook_source_id": row["handbook_source_id"],
                "primary_page": int(row["primary_page"]),
                "candidate_pages": [int(value) for value in row["candidate_pages"].split(";") if value],
                "alignment_score": float(row["alignment_score"]),
                "score_margin": float(row["score_margin"]),
                "alignment_method": row["alignment_method"],
            }
            for row in low_rows
        ],
        "medium_confidence_count": len(medium_rows),
        "medium_confidence_cases": [
            {
                "standard_point_id": row["standard_point_id"],
                "handbook_source_id": row["handbook_source_id"],
                "primary_page": int(row["primary_page"]),
                "candidate_pages": [int(value) for value in row["candidate_pages"].split(";") if value],
                "alignment_score": float(row["alignment_score"]),
                "score_margin": float(row["score_margin"]),
                "alignment_method": row["alignment_method"],
            }
            for row in medium_rows
        ],
        "integrity": {
            "exactly_572_rows": len(outputs) == EXPECTED_POINT_COUNT,
            "all_standard_ids_unique": len({row["standard_point_id"] for row in outputs}) == EXPECTED_POINT_COUNT,
            "all_primary_pages_nonempty": all(bool(row["primary_page"]) for row in outputs),
            "all_candidate_pages_nonempty": all(bool(row["candidate_pages"]) for row in outputs),
            "all_page_hashes_nonempty": all(bool(row["page_text_sha256"]) for row in outputs),
            "all_catalog_corrections_applied": len(applied_corrections) > 0,
            "no_handbook_text_fields_released": set(OUTPUT_FIELDS).isdisjoint({"text", "snippet", "handbook_text"}),
        },
    }
    args.coverage_out.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_out.write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Aligned {len(outputs)}/{EXPECTED_POINT_COUNT} standard points")
    print(f"Applied verified catalog corrections: {len(applied_corrections)}")
    print(f"Status counts: {dict(sorted(status_counts.items()))}")
    print(f"Confidence counts: {dict(sorted(confidence_counts.items()))}")
    print(f"Method counts: {dict(sorted(method_counts.items()))}")
    print(f"Low-confidence cases: {len(low_rows)}")
    for row in low_rows[: max(0, args.low_confidence_limit)]:
        print(
            "  {standard_point_id}: source={handbook_source_id} page={primary_page} "
            "score={alignment_score} margin={score_margin} method={alignment_method}".format(**row)
        )
    print(f"Medium-confidence review cases: {len(medium_rows)}")
    for row in medium_rows[: max(0, args.low_confidence_limit)]:
        print(
            "  {standard_point_id}: source={handbook_source_id} page={primary_page} "
            "score={alignment_score} margin={score_margin} method={alignment_method}".format(**row)
        )
    print(f"Wrote {args.alignment_out}")
    print(f"Wrote {args.coverage_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
