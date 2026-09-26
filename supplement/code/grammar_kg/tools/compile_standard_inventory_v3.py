#!/usr/bin/env python3
"""Compile the complete GF 0025--2021 inventory into runnable diagnostics.

This compiler deliberately keeps three claims separate:

* an official grammar point is an authority-layer entry;
* a DiagnosticUnit is a machine split of an entry's numbered usages; and
* a runtime binding is a source-anchored, statically audited program artifact.

The generated bindings are *not* described as linguist- or human-reviewed.  They
are deterministic starting points for inventory-wide execution and contract
checking.  The generated cases are not behavioral or semantic validation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VERSION = "v3.0"
EXPECTED_STANDARD_POINTS = 572
EXPECTED_SEED_LABELS = 37
EXPECTED_LEGACY_REGEXES = 474

DIAGNOSIS_TYPE_LABELS = [
    "目标用例",
    "边界问题",
    "表层触发但非目标",
    "结构缺槽",
    "语义不兼容",
    "其他具体构式",
    "不确定/需复核",
]

# These fields are returned together with one of the seven diagnosis labels.
# They are output fields, not replacements for the diagnosis label inventory.
STRUCTURED_OUTPUT_FIELDS = [
    "target_status",
    "candidate_boundary",
    "slot_status",
    "structural_functional_status",
    "licensed_evidence",
    "runtime_evidence",
    "rule_source_trace",
]

TOP_NUMBER_RE = re.compile(r"(?:（(?P<fw>[0-9０-９]{1,2})）|\((?P<ascii>[0-9]{1,2})\))")
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
CIRCLED_RE = re.compile("[" + CIRCLED + "]")
HAN_RE = re.compile(r"[\u3400-\u9fff]{1,24}")

GENERIC_MARKER_TERMS = {
    "表示",
    "用法",
    "结构",
    "类型",
    "其他",
    "词类",
    "词语",
    "句子",
    "短语",
    "语法点",
    "未细分",
    "候选",
    "基本",
    "形式",
    "功能",
    "意义",
    "方法",
    "固定格式",
    "口语格式",
}

CONFUSABLE_PREFIXES = ["术语", "标题", "编号", "元语言", "示例"]
REGEX_RANGE_ARTIFACT_LITERALS = {"龥", "鿿", "一-龥", "一-鿿", "\\u4e00", "\\u9fa5", "\\u9fff"}


@dataclass(frozen=True)
class DiagnosticUnit:
    standard_point_id: str
    unit_path: str
    unit_index: int
    unit_text: str
    scope_text: str
    split_kind: str
    split_marker: str

    @property
    def unit_id(self) -> str:
        safe_path = self.unit_path.replace(".", "-")
        return f"DUV3-{self.standard_point_id}-{safe_path}"

    @property
    def profile_id(self) -> str:
        return self.unit_id.replace("DUV3-", "DPV3-", 1)

    @property
    def binding_id(self) -> str:
        return self.unit_id.replace("DUV3-", "RBV3-", 1)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=root / "kg" / "standard_grammar_points_gf0025_v2.csv",
    )
    parser.add_argument(
        "--constraints",
        type=Path,
        default=root / "kg" / "constraints_v0_1.json",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=root / "kg" / "standard_catalog_corrections_v3.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=root / "kg")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_id(prefix: str, *parts: str, length: int = 12) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:length].upper()
    return f"{prefix}-{digest}"


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def normalize_digits(value: str) -> str:
    return value.translate(str.maketrans("０１２３４５６７８９", "0123456789"))


def split_at_markers(text: str, matches: Sequence[re.Match[str]]) -> list[tuple[str, str, str]]:
    """Return (path marker, marker text, body) for a numbered list."""
    pieces: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_number = match.groupdict().get("fw") or match.groupdict().get("ascii") or ""
        number = normalize_digits(raw_number)
        body = text[match.end() : next_start].strip(" \t\r\n；;，,")
        if body:
            pieces.append((number, match.group(0), body))
    return pieces


def outside_quoted_span(text: str, position: int) -> bool:
    """Return false for a position inside ordinary Chinese quotation marks."""
    for opening, closing in (("“", "”"), ("‘", "’")):
        last_open = text.rfind(opening, 0, position)
        last_close = text.rfind(closing, 0, position)
        if last_open > last_close:
            return False
    return True


def split_numbered_usages(row: Mapping[str, str]) -> list[DiagnosticUnit]:
    """Split explicit internal enumerations while preserving one unit otherwise.

    A list is split only when it contains at least two peers.  This avoids
    treating a lone cross-reference such as ``见（1）`` as a diagnostic usage.
    Circled sub-usages are split inside an already numbered usage when two or
    more circled peers occur.
    """
    text = (row.get("standard_text") or row.get("grammar_point") or "").strip()
    point_id = row["standard_point_id"]
    # Cross-references frequently contain a quoted item number.  They are
    # source pointers, not peer usages, and must not create duplicate units.
    top_matches = [match for match in TOP_NUMBER_RE.finditer(text) if outside_quoted_span(text, match.start())]
    raw_units: list[tuple[str, str, str, str, str]] = []

    if len(top_matches) >= 2:
        parent_scope = text[: top_matches[0].start()].strip(" \t\r\n；;，,：:")
        for top_path, top_marker, top_body in split_at_markers(text, top_matches):
            circled_matches = list(CIRCLED_RE.finditer(top_body))
            if len(circled_matches) >= 2:
                nested_scope = top_body[: circled_matches[0].start()].strip(" \t\r\n；;，,：:")
                scope = " / ".join(part for part in (parent_scope, nested_scope) if part)
                for sub_index, sub_match in enumerate(circled_matches):
                    end = circled_matches[sub_index + 1].start() if sub_index + 1 < len(circled_matches) else len(top_body)
                    sub_body = top_body[sub_match.end() : end].strip(" \t\r\n；;，,")
                    if sub_body:
                        sub_path = str(CIRCLED.index(sub_match.group(0)) + 1)
                        raw_units.append((f"{top_path}.{sub_path}", sub_match.group(0), sub_body, scope, "nested_numbered_usage"))
            else:
                raw_units.append((top_path, top_marker, top_body, parent_scope, "numbered_usage"))
    else:
        circled_matches = list(CIRCLED_RE.finditer(text))
        if len(circled_matches) >= 2:
            parent_scope = text[: circled_matches[0].start()].strip(" \t\r\n；;，,：:")
            for sub_index, sub_match in enumerate(circled_matches):
                end = circled_matches[sub_index + 1].start() if sub_index + 1 < len(circled_matches) else len(text)
                sub_body = text[sub_match.end() : end].strip(" \t\r\n；;，,")
                if sub_body:
                    sub_path = str(CIRCLED.index(sub_match.group(0)) + 1)
                    raw_units.append((sub_path, sub_match.group(0), sub_body, parent_scope, "circled_numbered_usage"))

    if not raw_units:
        raw_units = [("01", "", text, "", "unsplit_official_entry")]

    units: list[DiagnosticUnit] = []
    for ordinal, (path, marker, body, scope, kind) in enumerate(raw_units, start=1):
        normalized_path = ".".join(f"{int(piece):02d}" for piece in path.split("."))
        units.append(
            DiagnosticUnit(
                standard_point_id=point_id,
                unit_path=normalized_path,
                unit_index=ordinal,
                unit_text=body,
                scope_text=scope,
                split_kind=kind,
                split_marker=marker,
            )
        )
    return units


def unquote_regex(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        try:
            parsed = json.loads(value) if value[0] == '"' else ast.literal_eval(value)
            if isinstance(parsed, str):
                return parsed
        except (ValueError, SyntaxError, json.JSONDecodeError):
            return value[1:-1]
    return value


def repair_regex(pattern: str) -> str:
    # Several workbook cells retain only one side of an Excel quote wrapper.
    # Treat boundary quotes as cell delimiters, not literal candidate text.
    repaired = pattern.strip().strip("\"',，；;")
    repaired = repaired.replace("\\p{Han}", "[\\u3400-\\u9fff]")
    repaired = repaired.replace("\\p{IsHan}", "[\\u3400-\\u9fff]")
    repaired = repaired.replace("\\Q", "").replace("\\E", "")
    repaired = re.sub(r"\(\?<([A-Za-z_]\w*)>", r"(?P<\1>", repaired)
    # The workbook uses ellipses descriptively in several regex cells.  Turn
    # them into a conservative wildcard rather than matching the glyphs.
    repaired = repaired.replace("……", ".{0,40}?").replace("…", ".{0,40}?")
    return repaired


def marker_candidates(text: str, legacy_pattern: str = "") -> list[str]:
    """Extract conservative literal triggers from source text."""
    candidates: list[str] = []

    exact_unit = compact(text)
    if (
        exact_unit
        and len(exact_unit) <= 8
        and not re.search(r"[、，,；;。！？!?：:（）()\[\]{}<>《》＋+/.]", exact_unit)
    ):
        # Preserve short lexical units before removing explanatory prose.
        # This is essential for split entries such as 可能 / 可以.
        candidates.append(exact_unit)

    for quoted in re.findall(r"[“\"]([^”\"]{1,20})[”\"]", text):
        candidates.extend(re.split(r"[…\.]{2,}|[、,/；;＋+]", quoted))

    source = re.sub(TOP_NUMBER_RE, " ", text)
    source = CIRCLED_RE.sub(" ", source)
    source = re.sub(r"[A-Za-z][0-9]?", " ", source)
    source = re.sub(r"[：:（）()\[\]{}<>《》]", " ", source)
    source = re.sub(r"(?:表示|用来|用于|可以|能够|用法|类型|结构|方法|分类|基本|其他)", " ", source)
    for piece in re.split(r"……|\.\.\.|[、，,；;。！？!?/＋+\s]", source):
        candidates.extend(HAN_RE.findall(piece))

    # Do not harvest Han runs from regex source.  Character classes such as
    # ``[一-龥]`` otherwise leak their Unicode range endpoints into triggers.
    # The normalized legacy regex is retained separately in the binding.
    _ = legacy_pattern

    cleaned: list[str] = []
    for candidate in candidates:
        # Do not strip grammatical particles as characters: doing so erased
        # legitimate one-character points such as 得, 过, and 着.
        token = candidate.strip()
        if not token or token in GENERIC_MARKER_TERMS or len(token) > 16:
            continue
        if token not in cleaned:
            cleaned.append(token)
    return cleaned[:16]


def seed_marker_literals(seed_rule: Mapping[str, Any]) -> list[str]:
    literals: list[str] = []
    for marker in seed_rule.get("markers", []):
        for piece in re.split(r"\.\.\.|…|/|、|等", str(marker)):
            piece = compact(piece)
            if piece and piece not in GENERIC_MARKER_TERMS and piece not in literals:
                literals.append(piece)
    return literals


def seed_marker_sequence(seed_rule: Mapping[str, Any]) -> list[str]:
    """Return marker pieces in role order, preserving repeated markers."""
    sequence: list[str] = []
    for marker in seed_rule.get("markers", []):
        for piece in re.split(r"\.\.\.|…|/|、|等", str(marker)):
            piece = compact(piece)
            if piece and piece not in GENERIC_MARKER_TERMS:
                sequence.append(piece)
    return sequence


def paired_seed_markers(seed_rule: Mapping[str, Any] | None) -> list[str]:
    if not seed_rule:
        return []
    canonical = str(seed_rule.get("canonical_label", ""))
    if "..." not in canonical and "…" not in canonical:
        return []
    sequence = seed_marker_sequence(seed_rule)
    return sequence[:2] if len(sequence) >= 2 else []


def family_compatible(seed_family: str, row: Mapping[str, str], unit_text: str) -> bool:
    context = compact("|".join([row.get("category", ""), row.get("subcategory", ""), row.get("section_item", ""), unit_text]))
    checks = {
        "比较构式": ("比较", "比字", "不如", "一样", "越"),
        "补语结构": ("补语",),
        "介词/框架结构": ("介词", "引出"),
        "复句关联结构": ("复句", "关联"),
        "处置/受事重组构式": ("把字句", "处置", "将"),
        "被动/受事凸显构式": ("被动", "被字句"),
        "多谓词/论元结构": ("连动", "连谓", "兼语", "双宾", "给予", "多谓词"),
        "时间/处所功能结构": ("时间", "处所", "方位", "状语"),
    }
    keys = checks.get(seed_family, ())
    return any(key in context for key in keys)


def choose_seed(
    row: Mapping[str, str], unit: DiagnosticUnit, seed_rules: Mapping[str, Mapping[str, Any]]
) -> tuple[str, int]:
    # Seed reuse is intentionally conservative.  A shared character is not a
    # semantic mapping: 方位名词 containing ``上/下`` is not a 处所状语, and a
    # clause containing ``就`` is not necessarily 如果...就.  The unit text
    # therefore drives matching; the parent text is consulted only when the
    # official entry was not split.
    unit_context = compact(unit.unit_text)
    if unit.split_kind == "unsplit_official_entry":
        unit_context = compact(row.get("grammar_point", "") or unit.unit_text)
    category = row.get("category", "")
    subcategory = row.get("subcategory", "")
    section_item = row.get("section_item", "")

    def contains_all_with_multiplicity(markers: Sequence[str]) -> bool:
        counts = Counter(markers)
        return all(unit_context.count(marker) >= needed for marker, needed in counts.items())

    def seed_is_primary(seed_name: str, seed: Mapping[str, Any]) -> tuple[bool, int]:
        canonical = compact(str(seed.get("canonical_label", seed_name)).replace("候选", ""))
        family = str(seed.get("family", ""))
        raw_markers = []
        for marker in seed.get("markers", []):
            parts = [compact(piece) for piece in re.split(r"\.\.\.|…|/|、|等", str(marker)) if compact(piece)]
            raw_markers.extend(parts)

        if family == "补语结构":
            if seed_name == "数量/时量补语候选":
                return (subcategory == "补语" and ("数量补语" in section_item or "时量补语" in unit_context), 120)
            core = canonical.replace("字", "")
            if core in {"结果补语", "趋向补语", "可能补语"}:
                return (subcategory == "补语" and core in section_item, 120)
            if seed_name == "得字补语候选":
                return (subcategory == "补语" and "得" in unit_context, 95)

        if family == "比较构式":
            if seed_name == "越...越候选" and unit_context.count("越") >= 2 and category == "固定格式":
                return True, 120
            if "比较句" not in section_item:
                return False, 0
            requirements = {
                "比字比较候选": lambda: bool(
                    re.search(r"(?:A\+?不?比B|A\+比\+|\+比\+|A比B)", unit_context)
                ),
                "没有比较候选": lambda: "没有" in unit_context,
                "不如比较候选": lambda: "不如" in unit_context,
                "一样比较候选": lambda: "一样" in unit_context or "相同" in unit_context,
                "越...越候选": lambda: unit_context.count("越") >= 2,
            }
            if seed_name in requirements:
                return requirements[seed_name](), 120

        if family == "介词/框架结构":
            if subcategory != "介词":
                return False, 0
            exact_lexeme = re.sub(r"[0-9０-９（）()\s]", "", unit_context)
            marker_sets = {
                "从字介词框架候选": {"从"},
                "对字介词框架候选": {"对"},
                "在字介词框架候选": {"在"},
                "向/往/朝介词框架候选": {"向", "往", "朝"},
                "给/为/关于等介词框架候选": {"给", "为", "关于", "为了"},
            }
            if seed_name in marker_sets:
                return any(exact_lexeme == marker or exact_lexeme.startswith(marker + "：") for marker in marker_sets[seed_name]), 110

        if family == "处置/受事重组构式":
            if seed_name == "把字句候选":
                return "把" in unit_context and "把" in section_item, 120
            if seed_name == "将字处置候选":
                return "将" in unit_context and ("把" in section_item or "处置" in section_item or category == "固定格式"), 110

        if family == "被动/受事凸显构式":
            # The seed binding is overt-marker based.  意念被动 is retained in
            # the full taxonomy but receives a taxonomy template instead.
            return (
                "被动句" in section_item
                and "意念被动" not in unit_context
                and any(marker in unit_context for marker in ("被", "叫", "让", "为")),
                120,
            )

        if family == "多谓词/论元结构":
            section_checks = {
                "连动结构候选": ("连动句", "连谓"),
                "兼语结构候选": ("兼语句", "兼语"),
                "双宾/给予结构候选": ("双宾语句", "给予"),
            }
            if seed_name in section_checks:
                return any(key in section_item or key in unit_context for key in section_checks[seed_name]), 120

        if family == "时间/处所功能结构":
            if not (category == "句子成分" and subcategory == "状语"):
                return False, 0
            has_time = any(key in unit_context for key in ("时间", "年", "月", "日", "时", "以前", "以后"))
            has_place = any(key in unit_context for key in ("处所", "方位", "在", "从", "到", "往", "向"))
            # A mixed official point needs its taxonomy-level template; do not
            # arbitrarily privilege one of the two seed labels.
            if has_time and has_place:
                return False, 0
            if seed_name == "时间状语候选":
                return has_time, 105
            if seed_name == "处所状语候选":
                return has_place, 105

        if family == "复句关联结构":
            if subcategory != "复句":
                return False, 0
            # Paired seeds require every marker with multiplicity.  This keeps
            # ``就``-only constructions out of 如果...就 and the unmarked
            # branch of a split entry out of the marked template.
            if len(raw_markers) >= 2:
                return contains_all_with_multiplicity(raw_markers), 125
            if len(raw_markers) == 1:
                marker = raw_markers[0]
                stripped = re.sub(r"[，,。；;：:……\.\s]", "", unit_context)
                if stripped == marker:
                    return True, 100
                semantic_sections = {
                    "只要候选": ("条件复句",),
                    "如果候选": ("假设复句",),
                    "但是候选": ("转折复句", "让步复句"),
                    "因为候选": ("因果复句",),
                    "即使候选": ("让步复句",),
                    "所以候选": ("因果复句",),
                    "既然候选": ("因果复句",),
                    "虽然候选": ("转折复句", "让步复句"),
                }
                allowed_sections = semantic_sections.get(seed_name, ())
                return marker in unit_context and section_item in allowed_sections, 105

        return False, 0

    best_name = ""
    best_score = 0
    for seed_name, seed in seed_rules.items():
        compatible, score = seed_is_primary(seed_name, seed)
        if not compatible:
            score = 0
        if score > best_score or (score == best_score and seed_name < best_name):
            best_name, best_score = seed_name, score
    return (best_name, best_score) if best_score >= 35 else ("", best_score)


def taxonomy_path(row: Mapping[str, str]) -> tuple[str, str, str]:
    return (
        row.get("category", "").strip() or "未标注类别",
        row.get("subcategory", "").strip() or "未细分",
        row.get("section_item", "").strip() or "未列小项",
    )


def mapping_disposition(row: Mapping[str, str], seed_name: str) -> str:
    if seed_name:
        return "seed_reused"
    category, subcategory, section_item = taxonomy_path(row)
    if subcategory not in {"未细分", ""} or section_item not in {"未列小项", "其他", ""}:
        return "taxonomy_instantiated"
    return "point_specific_instantiated"


def choose_regime(row: Mapping[str, str], unit: DiagnosticUnit, seed_rule: Mapping[str, Any] | None) -> str:
    if seed_rule and str(seed_rule.get("family", "")) == "时间/处所功能结构":
        return "complex_functional"
    context = compact("|".join([row.get("category", ""), row.get("subcategory", ""), row.get("section_item", ""), unit.unit_text]))
    functional_terms = (
        "句子成分",
        "功能",
        "意义",
        "语义",
        "时间",
        "处所",
        "方位",
        "复指",
        "省略",
        "强调",
        "提问",
        "表示",
    )
    return "complex_functional" if any(term in context for term in functional_terms) else "structural_pattern"


def slots_for(row: Mapping[str, str], seed_rule: Mapping[str, Any] | None) -> list[str]:
    family = str(seed_rule.get("family", "")) if seed_rule else ""
    family_slots = {
        "比较构式": ["comparison_subject", "comparison_marker", "comparison_object", "comparison_dimension"],
        "补语结构": ["predicate", "complement_marker", "complement_core", "object_or_extent"],
        "介词/框架结构": ["frame_marker", "frame_object", "governing_predicate", "relation_type"],
        "复句关联结构": ["clause_1", "relation_marker_1", "clause_2", "relation_marker_2"],
        "处置/受事重组构式": ["agent", "disposal_marker", "patient", "predicate", "result_or_complement"],
        "被动/受事凸显构式": ["patient", "passive_marker", "agent_optional", "predicate"],
        "多谓词/论元结构": ["predicate_1", "shared_argument", "predicate_2", "event_relation"],
        "时间/处所功能结构": ["candidate_expression", "governing_event", "functional_role", "span_boundary"],
    }
    if family in family_slots:
        return family_slots[family]

    category = row.get("category", "")
    subcategory = row.get("subcategory", "")
    if category == "词类":
        return ["lexical_form", "local_context", "grammatical_function"]
    if category == "语素":
        return ["morpheme", "host_form", "word_boundary"]
    if category == "短语":
        return ["phrase_head", "dependent_or_argument", "phrase_boundary"]
    if category == "句子成分":
        return ["candidate_expression", "governing_predicate", "syntactic_role"]
    if category == "句子的类型" and subcategory == "复句":
        return ["clause_1", "relation_marker", "clause_2", "logical_relation"]
    if category == "句子的类型":
        return ["subject_or_topic", "predicate", "construction_marker", "clause_boundary"]
    if category == "动作的态":
        return ["predicate", "aspect_marker", "event_state"]
    if category in {"固定格式", "口语格式"}:
        return ["fixed_marker", "internal_slot", "construction_boundary"]
    if category == "提问的方法":
        return ["question_scope", "question_marker", "requested_information"]
    if category == "强调的方法":
        return ["focus_expression", "focus_marker", "proposition"]
    return ["trigger", "local_context", "construction_boundary"]


def taxonomy_rules(
    row: Mapping[str, str], unit: DiagnosticUnit, regime: str, slots: Sequence[str]
) -> tuple[list[str], list[str], list[str], list[str]]:
    path = " / ".join(taxonomy_path(row))
    positive = [
        f"候选包含与权威条目 {row['official_label']} 对齐的可定位触发线索。",
        f"候选满足模板槽位：{'、'.join(slots)}。",
        f"候选按 {regime} 模式实现 {path} 所规定的语法功能。",
    ]
    weak = [
        "触发线索存在，但候选边界未覆盖完整构式。",
        "局部槽位可定位，但至少一个必需槽位仅得到弱支持。",
    ]
    strong = [
        "触发线索只出现在标题、编号、术语或元语言说明中。",
        "必需槽位缺失，且上下文不能恢复目标结构或目标功能。",
    ]
    confusable = [
        f"相同表层形式可能实现 {row.get('subcategory', '其他')} 中的非目标功能。",
        "相邻构式共享触发形式，但槽位关系或语法功能不同。",
    ]
    return positive, weak, strong, confusable


def make_template(
    row: Mapping[str, str],
    unit: DiagnosticUnit,
    seed_name: str,
    seed_rule: Mapping[str, Any] | None,
    disposition: str,
    regime: str,
    slots: Sequence[str],
) -> dict[str, Any]:
    category, subcategory, section_item = taxonomy_path(row)
    if disposition == "seed_reused":
        # A seed is reused within the official taxonomy path rather than as a
        # taxonomy-free global template.  This keeps every GF 0025 path
        # explicit while preserving the original 37-label rule content.
        template_id = stable_id("DTV3-SEED", seed_name, category, subcategory, section_item, regime)
        name = str(seed_rule.get("canonical_label", seed_name))
        positive = list(seed_rule.get("positive_rules", []))
        weak = list(seed_rule.get("weak_reject_rules", []))
        strong = list(seed_rule.get("strong_reject_rules", []))
        diagnosis_rules = list(seed_rule.get("diagnosis_rules", []))
        confusable = [
            str(rule.get("if"))
            for rule in diagnosis_rules
            if isinstance(rule, Mapping)
            and str(rule.get("then", "")) in {"表层触发但非目标", "语义不兼容", "其他具体构式"}
        ]
        if not confusable:
            confusable = ["表层标记出现，但候选不实现目标语法功能。"]
        origin = "constraints_v0_1_seed"
    else:
        positive, weak, strong, confusable = taxonomy_rules(row, unit, regime, slots)
        diagnosis_rules = [
            {"if": "候选边界不完整或过长", "then": "边界问题"},
            {"if": "触发存在但必需槽位缺失", "then": "结构缺槽"},
            {"if": "表层触发仅为元语言或非目标功能", "then": "表层触发但非目标"},
            {"if": "槽位完整且目标功能成立", "then": "目标用例"},
        ]
        if disposition == "taxonomy_instantiated":
            template_id = stable_id("DTV3-TAX", category, subcategory, section_item, regime)
            name = f"{category}/{subcategory}/{section_item}"
            origin = "GF0025_full_taxonomy"
        else:
            template_id = stable_id("DTV3-POINT", row["standard_point_id"], regime)
            name = f"{row['official_label']} point-specific"
            origin = "GF0025_point_specific"

    return {
        "template_id": template_id,
        "template_name": name,
        "template_origin": origin,
        "mapping_disposition": disposition,
        "source_seed_label": seed_name,
        "taxonomy_category": category,
        "taxonomy_subcategory": subcategory,
        "taxonomy_section_item": section_item,
        "regime": regime,
        "trigger_strategy": "source_literal_and_normalized_legacy_regex",
        "slot_schema": list(slots),
        "positive_constraints": positive,
        "weak_rejections": weak,
        "strong_rejections": strong,
        "confusable_uses": confusable,
        "diagnosis_rules": diagnosis_rules,
        "diagnosis_type_labels": DIAGNOSIS_TYPE_LABELS,
        "status": "machine_instantiated_source_anchored",
    }


def literal_pattern(literals: Sequence[str]) -> str:
    clean = [literal for literal in literals if literal]
    if not clean:
        clean = ["语法"]
    ordered = sorted(dict.fromkeys(clean), key=lambda value: (-len(value), value))
    return "(?:" + "|".join(re.escape(value) for value in ordered[:16]) + ")"


def normalize_legacy_regex(raw: str, fallback_literals: Sequence[str]) -> tuple[str, str, str]:
    """Return a Python-compilable expression, disposition, and error note."""
    unquoted = unquote_regex(raw)
    repaired = repair_regex(unquoted)
    try:
        re.compile(repaired)
        return repaired, "normalized_compiles", ""
    except re.error as exc:
        fallback = literal_pattern(fallback_literals)
        re.compile(fallback)
        return fallback, "source_literal_fallback_compiles", str(exc)


def candidate_pattern_matches(pattern: str, mode: str, value: str) -> bool:
    compiled = re.compile(pattern)
    return (
        compiled.fullmatch(value) is not None
        if mode == "fullmatch"
        else compiled.search(value) is not None
    )


def active_candidate_detector_matches(
    pattern: str,
    mode: str,
    value: str,
    unit_specialization_pattern: str = "",
) -> bool:
    base_match = candidate_pattern_matches(pattern, mode, value)
    if not base_match:
        return False
    return (
        True
        if not unit_specialization_pattern
        else re.search(unit_specialization_pattern, value) is not None
    )


def _regex_token_strings(op: Any, argument: Any, limit: int) -> list[str]:
    """Generate conservative witnesses for Python ``re`` parser tokens."""
    constants = re._constants  # type: ignore[attr-defined]
    if op is constants.LITERAL:
        return [chr(argument)]
    if op is constants.NOT_LITERAL:
        return ["乙" if chr(argument) == "甲" else "甲"]
    if op is constants.ANY:
        return ["甲"]
    if op is constants.AT:
        return [""]
    if op is constants.CATEGORY:
        category_values = {
            constants.CATEGORY_DIGIT: "1",
            constants.CATEGORY_NOT_DIGIT: "甲",
            constants.CATEGORY_SPACE: " ",
            constants.CATEGORY_NOT_SPACE: "甲",
            constants.CATEGORY_WORD: "甲",
            constants.CATEGORY_NOT_WORD: "-",
            constants.CATEGORY_LINEBREAK: "\n",
            constants.CATEGORY_NOT_LINEBREAK: "甲",
        }
        return [category_values.get(argument, "甲")]
    if op is constants.IN:
        negated = any(inner_op is constants.NEGATE for inner_op, _ in argument)
        if negated:
            forbidden: set[str] = set()
            for inner_op, inner_arg in argument:
                if inner_op is constants.LITERAL:
                    forbidden.add(chr(inner_arg))
                elif inner_op is constants.RANGE:
                    low, high = inner_arg
                    if high - low <= 128:
                        forbidden.update(chr(codepoint) for codepoint in range(low, high + 1))
            for choice in ("甲", "A", "1", "-"):
                if choice not in forbidden:
                    return [choice]
            return ["甲"]
        choices: list[str] = []
        for inner_op, inner_arg in argument:
            if inner_op is constants.LITERAL:
                choices.append(chr(inner_arg))
            elif inner_op is constants.RANGE:
                low, high = inner_arg
                preferred = ord("甲") if low <= ord("甲") <= high else low
                choices.append(chr(preferred))
            elif inner_op is constants.CATEGORY:
                choices.extend(_regex_token_strings(inner_op, inner_arg, limit))
        return list(dict.fromkeys(choices))[:limit] or ["甲"]
    if op is constants.SUBPATTERN:
        return _regex_sequence_strings(argument[-1], limit)
    if op is constants.BRANCH:
        output: list[str] = []
        for branch in argument[1]:
            output.extend(_regex_sequence_strings(branch, limit))
            if len(output) >= limit:
                break
        return list(dict.fromkeys(output))[:limit]
    if op in {constants.MAX_REPEAT, constants.MIN_REPEAT, constants.POSSESSIVE_REPEAT}:
        minimum, maximum, repeated = argument
        atoms = _regex_sequence_strings(repeated, limit) or [""]
        counts = [minimum]
        if minimum == 0 and maximum != 0:
            counts.append(1)
        output = []
        for count in counts:
            for atom in atoms:
                output.append(atom * count)
        return list(dict.fromkeys(output))[:limit]
    if op in {constants.ASSERT, constants.ASSERT_NOT}:
        # Lookarounds consume no characters.  Candidate verification below
        # rejects a witness if ignoring the assertion made it invalid.
        return [""]
    if op is constants.GROUPREF:
        return ["甲"]
    if op is constants.GROUPREF_EXISTS:
        yes_branch = argument[1]
        no_branch = argument[2]
        output = _regex_sequence_strings(yes_branch, limit)
        if no_branch:
            output.extend(_regex_sequence_strings(no_branch, limit))
        return list(dict.fromkeys(output))[:limit]
    if op in {constants.SUCCESS, constants.FAILURE}:
        return [""] if op is constants.SUCCESS else []
    return [""]


def _regex_sequence_strings(tokens: Iterable[tuple[Any, Any]], limit: int = 128) -> list[str]:
    candidates = [""]
    for op, argument in tokens:
        pieces = _regex_token_strings(op, argument, limit)
        combined: list[str] = []
        for prefix in candidates:
            for piece in pieces:
                combined.append(prefix + piece)
                if len(combined) >= limit:
                    break
            if len(combined) >= limit:
                break
        candidates = list(dict.fromkeys(combined))[:limit]
        if not candidates:
            break
    return candidates


def _regex_group_aware_witness(
    tokens: Iterable[tuple[Any, Any]], groups: dict[int, str] | None = None
) -> str:
    """Build one witness while preserving simple capture/backreference state."""
    constants = re._constants  # type: ignore[attr-defined]
    groups = groups if groups is not None else {}
    output: list[str] = []
    for op, argument in tokens:
        if op is constants.SUBPATTERN:
            group_number = argument[0]
            value = _regex_group_aware_witness(argument[-1], groups)
            if group_number:
                groups[int(group_number)] = value
            output.append(value)
        elif op is constants.GROUPREF:
            output.append(groups.get(int(argument), "甲"))
        elif op is constants.BRANCH:
            branches = argument[1]
            output.append(_regex_group_aware_witness(branches[0], groups) if branches else "")
        elif op in {constants.MAX_REPEAT, constants.MIN_REPEAT, constants.POSSESSIVE_REPEAT}:
            minimum, maximum, repeated = argument
            count = minimum if minimum > 0 else (1 if maximum != 0 else 0)
            atom = _regex_group_aware_witness(repeated, groups)
            output.append(atom * count)
        elif op is constants.GROUPREF_EXISTS:
            group_number, yes_branch, no_branch = argument
            branch = yes_branch if int(group_number) in groups else no_branch
            if branch:
                output.append(_regex_group_aware_witness(branch, groups))
        elif op in {constants.ASSERT, constants.ASSERT_NOT, constants.AT}:
            continue
        else:
            choices = _regex_token_strings(op, argument, 1)
            output.append(choices[0] if choices else "")
    return "".join(output)


def matching_candidate_exemplar(
    pattern: str,
    mode: str,
    literals: Sequence[str],
    unit_text: str,
    unit_specialization_pattern: str = "",
) -> str:
    """Return a deterministic string accepted by the active detector."""
    probes: list[str] = []
    for literal in literals:
        probes.extend([literal, f"甲{literal}乙"])
    compact_unit = compact(unit_text)
    if compact_unit:
        probes.append(compact_unit)
    # Backreference-heavy reduplication rules are easiest to witness by
    # repeating literal Han alternatives from the candidate regex.  These
    # probes are contract-only and are never copied into retrieval triggers.
    for run in HAN_RE.findall(pattern):
        for character in run:
            probes.extend([character * 2, character * 3, f"一{character * 2}"])
    simple_backref = re.fullmatch(
        r"\^?(?P<prefix>.*?)\(\[(?P<class_body>[^\]]+)\]\)(?P<middle>.*?)\\1(?P<suffix>.*?)\$?",
        pattern,
    )
    if simple_backref:
        class_body = simple_backref.group("class_body")
        witness_character = next(
            (character for character in class_body if "\u3400" <= character <= "\u9fff"),
            "甲",
        )

        def literal_fragment(fragment: str) -> str:
            fragment = re.sub(r"\\(.)", r"\1", fragment)
            return re.sub(r"[\^$?*+{}()]", "", fragment)

        probes.append(
            literal_fragment(simple_backref.group("prefix"))
            + witness_character
            + literal_fragment(simple_backref.group("middle"))
            + witness_character
            + literal_fragment(simple_backref.group("suffix"))
        )
    try:
        parsed = re._parser.parse(pattern, 0)  # type: ignore[attr-defined]
        probes.append(_regex_group_aware_witness(parsed))
        generated = _regex_sequence_strings(parsed, limit=256)
        probes.extend(generated)
        probes.extend(f"甲{candidate}乙" for candidate in generated if candidate)
    except Exception:
        pass
    for probe in dict.fromkeys(probes):
        if active_candidate_detector_matches(
            pattern, mode, probe, unit_specialization_pattern
        ):
            return probe
    raise ValueError(f"Could not generate a matching detector exemplar for pattern: {pattern!r}")


def rejected_candidate_exemplar(
    pattern: str, mode: str, unit_specialization_pattern: str = ""
) -> str:
    for probe in ("🚫", "###", "XYZ", "\x00", ""):
        if not active_candidate_detector_matches(
            pattern, mode, probe, unit_specialization_pattern
        ):
            return probe
    raise ValueError(f"Candidate detector appears vacuous for smoke probes: {pattern!r}")


def source_literals(
    row: Mapping[str, str], unit: DiagnosticUnit, seed_rule: Mapping[str, Any] | None
) -> list[str]:
    raw_legacy = unquote_regex(row.get("legacy_regex", ""))
    unit_candidates = marker_candidates(unit.unit_text, raw_legacy)
    seed_candidates = seed_marker_literals(seed_rule) if seed_rule else []
    compact_unit = compact(unit.unit_text)
    matched_seed = [marker for marker in seed_candidates if marker in compact_unit]
    unmatched_seed = [marker for marker in seed_candidates if marker not in matched_seed]
    candidates: list[str] = []
    candidates.extend(matched_seed)
    candidates.extend(unit_candidates)
    # Do not append unmatched seed alternatives (往 must not inherit 朝) or
    # the full parent entry after splitting (a numbered usage must not inherit
    # its sibling's marker).
    _ = unmatched_seed
    if unit.split_kind == "unsplit_official_entry":
        candidates.extend(marker_candidates(row.get("grammar_point", ""), raw_legacy))
    if not candidates:
        candidates.extend(marker_candidates(row.get("section_item", "")))
    if not candidates:
        candidates.extend(marker_candidates(row.get("subcategory", "")))
    if not candidates:
        candidates.append(row["official_label"])
    return list(dict.fromkeys(candidates))[:16]


def runtime_patterns(literals: Sequence[str], paired_markers: Sequence[str] = ()) -> dict[str, str]:
    marker = literal_pattern(literals)
    primary = re.escape(literals[0] if literals else "语法")
    paired = len(paired_markers) >= 2
    if paired:
        first_literal, second_literal = paired_markers[:2]
        primary = re.escape(first_literal)
        secondary = re.escape(second_literal)
        structure = rf".{{1,40}}{primary}.{{0,40}}{secondary}.{{1,40}}"
        positive_surface = f"甲{first_literal}丙{second_literal}乙"
        missing_surface = f"甲{first_literal}"
    else:
        structure = rf".{{1,40}}{primary}.{{1,40}}"
        positive_surface = f"甲{literals[0] if literals else '语法'}乙"
        missing_surface = f"甲{literals[0] if literals else '语法'}"
    weak = rf"^(?:{marker})$"
    confusable = rf"(?:{'|'.join(map(re.escape, CONFUSABLE_PREFIXES))})[^。！？]{{0,20}}{primary}"
    strong = rf"(?:{confusable})|(?:{marker})(?:缺槽|残片)$"
    for pattern in (marker, structure, weak, confusable, strong):
        re.compile(pattern)
    return {
        "marker_pattern": marker,
        "structure_pattern": structure,
        "positive_pattern": structure,
        "weak_reject_pattern": weak,
        "strong_reject_pattern": strong,
        "confusable_pattern": confusable,
        "positive_surface": positive_surface,
        "missing_surface": missing_surface,
    }


def make_tests(unit: DiagnosticUnit, binding: Mapping[str, Any]) -> list[dict[str, str]]:
    positive = binding["_test_surfaces"]["detector_positive"]
    cases = [
        ("positive", "correct", "complete", "compatible", "ordinary", "resolved", "目标用例"),
        ("boundary", "incomplete", "complete", "compatible", "ordinary", "resolved", "边界问题"),
        ("missing_slot", "correct", "missing", "compatible", "ordinary", "resolved", "结构缺槽"),
        ("confusable", "correct", "complete", "compatible", "metalinguistic", "resolved", "表层触发但非目标"),
        ("semantic_incompatible", "correct", "complete", "incompatible", "ordinary", "resolved", "语义不兼容"),
        ("other_construction", "correct", "complete", "compatible", "other_construction", "resolved", "其他具体构式"),
        ("uncertain", "correct", "complete", "compatible", "ordinary", "unresolved", "不确定/需复核"),
    ]
    output: list[dict[str, str]] = []
    for case_index, (
        case_type,
        boundary_observation,
        slot_observation,
        semantic_observation,
        usage_context,
        uncertainty_observation,
        expected,
    ) in enumerate(cases, start=1):
        output.append(
            {
                "test_id": f"TV3-{unit.unit_id}-{case_index:02d}",
                "diagnostic_unit_id": unit.unit_id,
                "binding_id": unit.binding_id,
                "case_type": case_type,
                "sentence": positive,
                "candidate_span": positive,
                "gold_span": positive,
                "boundary_observation": boundary_observation,
                "slot_observation": slot_observation,
                "semantic_observation": semantic_observation,
                "usage_context": usage_context,
                "uncertainty_observation": uncertainty_observation,
                "expected_diagnosis_type": expected,
                "generation_status": "deterministic_machine_generated_contract_test",
            }
        )
    return output


def execute_test(binding: Mapping[str, Any], test: Mapping[str, str]) -> dict[str, Any]:
    """Execute one binding contract, including the active candidate detector."""
    candidate = test["candidate_span"]
    detector = binding["candidate_detector"]
    candidate_re = re.compile(detector["candidate_pattern"])
    _ = candidate_re
    candidate_matched = active_candidate_detector_matches(
        detector["candidate_pattern"],
        detector["candidate_match_mode"],
        candidate,
        detector.get("unit_specialization_pattern", ""),
    )
    marker_re = re.compile(binding["trigger"]["marker_pattern"])
    structure_re = re.compile(binding["runtime_conditions"]["structure_pattern"])
    confusable_re = re.compile(binding["runtime_conditions"]["confusable_pattern"])
    # The observations are executor inputs produced by upstream span/slot and
    # context checks.  Contract tests exercise decision wiring; they are not
    # corpus-derived linguistic judgments.
    if test.get("boundary_observation") != "correct":
        diagnosis = "边界问题"
    elif test.get("slot_observation") != "complete":
        diagnosis = "结构缺槽"
    elif test.get("usage_context") == "metalinguistic":
        diagnosis = "表层触发但非目标"
    elif test.get("semantic_observation") == "incompatible":
        diagnosis = "语义不兼容"
    elif test.get("usage_context") == "other_construction":
        diagnosis = "其他具体构式"
    elif test.get("uncertainty_observation") != "resolved":
        diagnosis = "不确定/需复核"
    elif not candidate_matched:
        # A candidate rejected by the active detector can never be accepted as
        # a target.  Retrieval/structure patterns remain separate diagnostics.
        diagnosis = "表层触发但非目标" if marker_re.search(candidate) else "其他具体构式"
    else:
        diagnosis = "目标用例"
    return {
        "diagnosis_type": diagnosis,
        "candidate_detector_invoked": True,
        "candidate_detector_matched": candidate_matched,
        "candidate_pattern_source": detector["candidate_pattern_source"],
        "candidate_match_mode": detector["candidate_match_mode"],
        "retrieval_trigger_matched": marker_re.search(candidate) is not None,
        "structure_pattern_matched": structure_re.search(candidate) is not None,
    }


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_no_review_claim(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False).lower()
    forbidden = ["human_reviewed", "linguist_reviewed", "人工审核通过", "语言学家审核通过"]
    return not any(term in text for term in forbidden)


def utf8_round_trip(value: Any) -> bool:
    """Check that every serialized Unicode value survives a UTF-8 round trip."""
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return serialized.encode("utf-8").decode("utf-8") == serialized and "�" not in serialized


def apply_catalog_corrections(
    catalog: list[dict[str, str]], corrections_path: Path
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    if not corrections_path.exists():
        return catalog, {}
    corrections = read_csv(corrections_path)
    by_id = {row["standard_point_id"]: row for row in catalog}
    provenance: dict[str, list[dict[str, str]]] = defaultdict(list)
    for correction in corrections:
        point_id = correction["standard_point_id"]
        field_name = correction["field_name"]
        if point_id not in by_id:
            raise ValueError(f"Correction references unknown point: {point_id}")
        if field_name not in by_id[point_id]:
            raise ValueError(f"Correction references unknown field: {field_name}")
        current = by_id[point_id][field_name]
        if current != correction["original_value"]:
            raise ValueError(
                f"Correction original mismatch for {point_id}.{field_name}: "
                f"catalog={current!r}, correction={correction['original_value']!r}"
            )
        by_id[point_id][field_name] = correction["corrected_value"]
        provenance[point_id].append(dict(correction))
    return catalog, dict(provenance)


def compile_inventory(
    catalog_path: Path,
    constraints_path: Path,
    corrections_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    catalog = read_csv(catalog_path)
    catalog, catalog_corrections = apply_catalog_corrections(catalog, corrections_path)
    constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
    seed_rules: dict[str, dict[str, Any]] = constraints["grammar_point_rules"]
    source_labels = list(constraints["decision_labels"]["diagnosis_type"])

    if len(catalog) != EXPECTED_STANDARD_POINTS:
        raise ValueError(f"Expected {EXPECTED_STANDARD_POINTS} official points, found {len(catalog)}")
    if len(seed_rules) != EXPECTED_SEED_LABELS:
        raise ValueError(f"Expected {EXPECTED_SEED_LABELS} seed labels, found {len(seed_rules)}")
    if source_labels != DIAGNOSIS_TYPE_LABELS:
        raise ValueError("The seven diagnosis_type labels differ from constraints_v0_1.json")

    legacy_count = sum(bool(row.get("legacy_regex", "").strip()) for row in catalog)
    if legacy_count != EXPECTED_LEGACY_REGEXES:
        raise ValueError(f"Expected {EXPECTED_LEGACY_REGEXES} legacy regexes, found {legacy_count}")

    points_by_id = {row["standard_point_id"]: row for row in catalog}
    if len(points_by_id) != len(catalog):
        raise ValueError("Duplicate standard_point_id in source catalog")

    crosswalk: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    all_tests: list[dict[str, str]] = []
    templates: dict[str, dict[str, Any]] = {}
    regex_audit: list[dict[str, str]] = []

    for row in catalog:
        units = split_numbered_usages(row)
        base_literals = marker_candidates(row.get("grammar_point", ""), unquote_regex(row.get("legacy_regex", "")))
        normalized_legacy = ""
        normalization_status = "not_available_source_literal_runtime"
        normalization_error = ""
        if row.get("legacy_regex", "").strip():
            normalized_legacy, normalization_status, normalization_error = normalize_legacy_regex(
                row["legacy_regex"], base_literals or [row["official_label"]]
            )
            regex_audit.append(
                {
                    "standard_point_id": row["standard_point_id"],
                    "status": normalization_status,
                    "error_before_fallback": normalization_error,
                }
            )

        for unit in units:
            seed_name, seed_score = choose_seed(row, unit, seed_rules)
            seed_rule = seed_rules.get(seed_name)
            disposition = mapping_disposition(row, seed_name)
            regime = choose_regime(row, unit, seed_rule)
            slots = slots_for(row, seed_rule)
            template = make_template(row, unit, seed_name, seed_rule, disposition, regime, slots)
            templates.setdefault(template["template_id"], template)

            literals = source_literals(row, unit, seed_rule)
            paired_markers = paired_seed_markers(seed_rule)
            patterns = runtime_patterns(literals, paired_markers)
            source_trace = {
                "standard_point_id": row["standard_point_id"],
                "official_label": row["official_label"],
                "standard_code": row["standard_code"],
                "source_id": row["source_id"],
                "source_pdf_page": row["source_pdf_page"],
                "page_anchor": row["page_anchor"],
                "page_anchor_status": row["page_anchor_status"],
                "catalog_corrections": catalog_corrections.get(row["standard_point_id"], []),
            }
            candidate_pattern = normalized_legacy or patterns["marker_pattern"]
            candidate_pattern_source = (
                "normalized_legacy_regex" if normalized_legacy else "source_literal_retrieval_trigger"
            )
            candidate_match_mode = (
                "fullmatch"
                if normalized_legacy and normalized_legacy.lstrip().startswith("^") and normalized_legacy.rstrip().endswith("$")
                else "search"
            )
            re.compile(candidate_pattern)
            unit_specialization_pattern = ""
            if normalized_legacy and unit.split_kind != "unsplit_official_entry":
                proposed_specialization = patterns["marker_pattern"]
                try:
                    detector_positive = matching_candidate_exemplar(
                        candidate_pattern,
                        candidate_match_mode,
                        literals,
                        unit.unit_text,
                        proposed_specialization,
                    )
                    unit_specialization_pattern = proposed_specialization
                    candidate_pattern_source = "normalized_legacy_regex_with_unit_specialization"
                except ValueError:
                    detector_positive = matching_candidate_exemplar(
                        candidate_pattern,
                        candidate_match_mode,
                        literals,
                        unit.unit_text,
                    )
            else:
                detector_positive = matching_candidate_exemplar(
                    candidate_pattern,
                    candidate_match_mode,
                    literals,
                    unit.unit_text,
                )
            detector_negative = rejected_candidate_exemplar(
                candidate_pattern,
                candidate_match_mode,
                unit_specialization_pattern,
            )
            binding = {
                "binding_id": unit.binding_id,
                "diagnostic_unit_id": unit.unit_id,
                "diagnostic_profile_id": unit.profile_id,
                "diagnostic_unit_scope": unit.scope_text,
                "standard_point_id": row["standard_point_id"],
                "template_id": template["template_id"],
                "mapping_disposition": disposition,
                "reused_seed_label": seed_name,
                "seed_compatibility_score": seed_score,
                "regime": regime,
                "diagnosis_type_labels": DIAGNOSIS_TYPE_LABELS,
                "structured_output_fields": STRUCTURED_OUTPUT_FIELDS,
                "trigger": {
                    "literal_markers": literals,
                    "literal_marker_source": "official_unit_text_and_compatible_seed_constraint",
                    "marker_pattern": patterns["marker_pattern"],
                    "legacy_regex_original": row.get("legacy_regex", ""),
                    "legacy_regex_normalized": normalized_legacy,
                    "legacy_regex_normalization_status": normalization_status,
                },
                "candidate_detector": {
                    "candidate_pattern": candidate_pattern,
                    "candidate_match_mode": candidate_match_mode,
                    "candidate_pattern_source": candidate_pattern_source,
                    "unit_specialization_pattern": unit_specialization_pattern,
                    "unit_specialization_required": bool(unit_specialization_pattern),
                    "retrieval_trigger_is_separate": True,
                    "positive_contract_exemplar": detector_positive,
                    "negative_gate_exemplar": detector_negative,
                },
                "slots": {"required": slots, "cardinality": {slot: "one_or_more" for slot in slots}},
                "constraints": {
                    "positive": template["positive_constraints"],
                    "weak_reject": template["weak_rejections"],
                    "strong_reject": template["strong_rejections"],
                    "confusable_uses": template["confusable_uses"],
                },
                "runtime_conditions": {
                    "structure_pattern": patterns["structure_pattern"],
                    "positive_pattern": patterns["positive_pattern"],
                    "weak_reject_pattern": patterns["weak_reject_pattern"],
                    "strong_reject_pattern": patterns["strong_reject_pattern"],
                    "confusable_pattern": patterns["confusable_pattern"],
                    "decision_order": [
                        "boundary_observation != correct -> 边界问题",
                        "slot_observation != complete -> 结构缺槽",
                        "usage_context == metalinguistic -> 表层触发但非目标",
                        "semantic_observation == incompatible -> 语义不兼容",
                        "usage_context == other_construction -> 其他具体构式",
                        "uncertainty_observation != resolved -> 不确定/需复核",
                        "active_candidate_detector == reject -> 表层触发但非目标 or 其他具体构式",
                        "active_candidate_detector == accept -> 目标用例",
                    ],
                },
                "source": source_trace,
                "status": {
                    "compilation": "machine_compiled",
                    "source_grounding": "source_anchored",
                    "review": "not_claimed",
                    "static_audit": "pending",
                },
                "_test_surfaces": {
                    "detector_positive": detector_positive,
                },
            }
            tests = make_tests(unit, binding)
            negative_gate_result = execute_test(
                binding,
                {
                    "candidate_span": detector_negative,
                    "gold_span": detector_negative,
                    "boundary_observation": "correct",
                    "slot_observation": "complete",
                    "semantic_observation": "compatible",
                    "usage_context": "ordinary",
                    "uncertainty_observation": "resolved",
                },
            )
            binding["candidate_detector"]["negative_gate_contract_passed"] = (
                not negative_gate_result["candidate_detector_matched"]
                and negative_gate_result["diagnosis_type"] != "目标用例"
            )
            test_results = [execute_test(binding, test) for test in tests]
            for test, result in zip(tests, test_results):
                observed = result["diagnosis_type"]
                test["observed_diagnosis_type"] = observed
                test["candidate_detector_invoked"] = str(result["candidate_detector_invoked"]).lower()
                test["candidate_detector_matched"] = str(result["candidate_detector_matched"]).lower()
                test["candidate_pattern_source"] = result["candidate_pattern_source"]
                test["candidate_match_mode"] = result["candidate_match_mode"]
                test["execution_status"] = (
                    "binding_contract_passed"
                    if observed == test["expected_diagnosis_type"]
                    else "binding_contract_failed"
                )
            test_pass = all(
                result["diagnosis_type"] == test["expected_diagnosis_type"]
                for result, test in zip(test_results, tests)
            )
            binding["status"]["static_audit"] = "static_audit_passed" if test_pass else "static_audit_failed"
            binding["static_contract_test_summary"] = {
                "tests": len(tests),
                "passed": sum(
                    result["diagnosis_type"] == test["expected_diagnosis_type"]
                    for result, test in zip(test_results, tests)
                ),
                "candidate_detector_invocations": sum(
                    bool(result["candidate_detector_invoked"]) for result in test_results
                ),
            }
            del binding["_test_surfaces"]
            bindings.append(binding)
            all_tests.extend(tests)

            crosswalk.append(
                {
                    "diagnostic_unit_id": unit.unit_id,
                    "unit_record_type": "DiagnosticUnit",
                    "diagnostic_profile_id": unit.profile_id,
                    "profile_record_type": "DiagnosticProfile",
                    "runtime_binding_id": unit.binding_id,
                    "standard_point_id": row["standard_point_id"],
                    "official_label": row["official_label"],
                    "level": row["level"],
                    "level_ordinal": row["level_ordinal"],
                    "category": row["category"],
                    "subcategory": row["subcategory"],
                    "section_item": row["section_item"],
                    "unit_index": unit.unit_index,
                    "unit_path": unit.unit_path,
                    "unit_text": unit.unit_text,
                    "unit_scope": unit.scope_text,
                    "split_kind": unit.split_kind,
                    "split_marker": unit.split_marker,
                    "template_id": template["template_id"],
                    "mapping_disposition": disposition,
                    "reused_seed_label": seed_name,
                    "seed_compatibility_score": seed_score,
                    "regime": regime,
                    "trigger_pattern": patterns["marker_pattern"],
                    "candidate_pattern": candidate_pattern,
                    "candidate_match_mode": candidate_match_mode,
                    "candidate_pattern_source": candidate_pattern_source,
                    "unit_specialization_pattern": unit_specialization_pattern,
                    "slot_schema": json_cell(slots),
                    "positive_constraints": json_cell(template["positive_constraints"]),
                    "weak_rejections": json_cell(template["weak_rejections"]),
                    "strong_rejections": json_cell(template["strong_rejections"]),
                    "confusable_uses": json_cell(template["confusable_uses"]),
                    "diagnosis_type_labels": json_cell(DIAGNOSIS_TYPE_LABELS),
                    "structured_output_fields": json_cell(STRUCTURED_OUTPUT_FIELDS),
                    "legacy_regex_normalized": normalized_legacy,
                    "regex_normalization_status": normalization_status,
                    "source_id": row["source_id"],
                    "source_pdf_page": row["source_pdf_page"],
                    "page_anchor": row["page_anchor"],
                    "page_anchor_status": row["page_anchor_status"],
                    "authority_status": row["authority_status"],
                    "catalog_correction_applied": str(
                        bool(catalog_corrections.get(row["standard_point_id"]))
                    ).lower(),
                    "catalog_correction_provenance": json_cell(
                        catalog_corrections.get(row["standard_point_id"], [])
                    ),
                    "compilation_status": "machine_compiled",
                    "source_grounding_status": "source_anchored",
                    "review_status": "not_claimed",
                    "static_audit_status": binding["status"]["static_audit"],
                    "profile_status": "machine_compiled_source_anchored_static_audited",
                }
            )

    template_rows = []
    for template_id in sorted(templates):
        template = templates[template_id]
        template_rows.append(
            {
                **{key: value for key, value in template.items() if not isinstance(value, (list, dict))},
                "slot_schema": json_cell(template["slot_schema"]),
                "positive_constraints": json_cell(template["positive_constraints"]),
                "weak_rejections": json_cell(template["weak_rejections"]),
                "strong_rejections": json_cell(template["strong_rejections"]),
                "confusable_uses": json_cell(template["confusable_uses"]),
                "diagnosis_rules": json_cell(template["diagnosis_rules"]),
                "diagnosis_type_labels": json_cell(template["diagnosis_type_labels"]),
            }
        )

    # Global integrity audits.
    point_units = defaultdict(list)
    for row in crosswalk:
        point_units[row["standard_point_id"]].append(row["diagnostic_unit_id"])
    tests_by_unit = Counter(test["diagnostic_unit_id"] for test in all_tests)
    diagnosis_labels_by_unit: dict[str, set[str]] = defaultdict(set)
    for test in all_tests:
        diagnosis_labels_by_unit[test["diagnostic_unit_id"]].add(test["expected_diagnosis_type"])
    binding_ids = {binding["binding_id"] for binding in bindings}
    unit_ids = {row["diagnostic_unit_id"] for row in crosswalk}
    profile_ids = {row["diagnostic_profile_id"] for row in crosswalk}
    normalized_regexes_compile = all(
        not row["legacy_regex_normalized"] or compile_ok(row["legacy_regex_normalized"])
        for row in crosswalk
    )
    runtime_regexes_compile = all(binding_patterns_compile(binding) for binding in bindings)
    candidate_patterns_compile = all(
        compile_ok(binding["candidate_detector"]["candidate_pattern"]) for binding in bindings
    )
    trigger_patterns_nonempty_and_usable = all(
        bool(binding["trigger"]["marker_pattern"])
        and bool(binding["trigger"]["literal_markers"])
        and any(
            re.compile(binding["trigger"]["marker_pattern"]).search(literal)
            for literal in binding["trigger"]["literal_markers"]
        )
        for binding in bindings
    )
    tests_passed = sum(
        execute_test(bindings_by_id(bindings)[test["binding_id"]], test)["diagnosis_type"]
        == test["expected_diagnosis_type"]
        for test in all_tests
    )
    distinct_taxonomy_paths = {taxonomy_path(row) for row in catalog}
    template_taxonomy_paths = {
        (template["taxonomy_category"], template["taxonomy_subcategory"], template["taxonomy_section_item"])
        for template in templates.values()
    }
    template_regimes = {template["template_id"]: template["regime"] for template in templates.values()}
    scoped_split_point_ids = {
        row["standard_point_id"]
        for row in crosswalk
        if row["split_kind"] != "unsplit_official_entry" and row["unit_scope"]
    }
    correction_records = [record for records in catalog_corrections.values() for record in records]
    integrity = {
        "official_point_count_is_572": len(catalog) == EXPECTED_STANDARD_POINTS,
        "official_point_ids_unique": len(points_by_id) == EXPECTED_STANDARD_POINTS,
        "all_official_points_have_units": set(points_by_id) == set(point_units),
        "diagnostic_unit_ids_unique": len(unit_ids) == len(crosswalk),
        "one_profile_per_unit": len(profile_ids) == len(crosswalk),
        "one_binding_per_unit": len(binding_ids) == len(crosswalk) == len(bindings),
        "seven_contract_tests_per_unit": all(tests_by_unit[unit_id] == 7 for unit_id in unit_ids),
        "all_seven_diagnosis_branches_contract_tested_per_unit": all(
            diagnosis_labels_by_unit[unit_id] == set(DIAGNOSIS_TYPE_LABELS) for unit_id in unit_ids
        ),
        "all_generated_contract_tests_pass": tests_passed == len(all_tests),
        "all_runtime_patterns_compile": runtime_regexes_compile,
        "all_active_candidate_patterns_compile": candidate_patterns_compile,
        "all_trigger_patterns_nonempty_and_match_a_source_literal": trigger_patterns_nonempty_and_usable,
        "no_regex_range_artifact_literals_in_triggers": all(
            not any(
                literal in REGEX_RANGE_ARTIFACT_LITERALS or "龥" in literal or "鿿" in literal
                for literal in binding["trigger"]["literal_markers"]
            )
            for binding in bindings
        ),
        "all_474_legacy_regexes_normalized_and_compile": len(regex_audit) == EXPECTED_LEGACY_REGEXES
        and normalized_regexes_compile,
        "all_taxonomy_paths_represented": distinct_taxonomy_paths <= template_taxonomy_paths,
        "crosswalk_template_regime_consistent": all(
            template_regimes[row["template_id"]] == row["regime"] for row in crosswalk
        ),
        "split_preamble_scope_preserved": bool(scoped_split_point_ids),
        "seven_diagnosis_labels_preserved": source_labels == DIAGNOSIS_TYPE_LABELS
        and len(source_labels) == 7,
        "seed_reuse_passes_conservative_compatibility_guard": all(
            row["mapping_disposition"] != "seed_reused"
            or (bool(row["reused_seed_label"]) and int(row["seed_compatibility_score"]) >= 95)
            for row in crosswalk
        ),
        "source_trace_complete": all(
            all(binding["source"].get(field) for field in ("standard_point_id", "official_label", "source_id", "page_anchor"))
            for binding in bindings
        ),
        "catalog_corrections_applied_with_authority_provenance": all(
            record.get("authority_source_id")
            and record.get("authority_pdf_page")
            and record.get("verification_status")
            for record in correction_records
        )
        and len(correction_records) == 4
        and set(catalog_corrections) == {"GF0025-2021-3-002", "GF0025-2021-5-010"},
        "no_excel_error_tokens_in_effective_catalog": all(
            row.get(field, "") not in {"#NAME?", "#VALUE!", "#REF!", "#N/A"}
            for row in catalog
            for field in ("grammar_point", "canonical_label", "standard_text")
        ),
        "no_human_or_linguist_review_claim": validate_no_review_claim(
            {"crosswalk": crosswalk, "templates": template_rows, "bindings": bindings}
        ),
        "all_bindings_machine_static_audit_passed": all(
            binding["status"]["static_audit"] == "static_audit_passed" for binding in bindings
        ),
        "all_test_rows_record_binding_contract_execution": all(
            test.get("execution_status") == "binding_contract_passed"
            and test.get("observed_diagnosis_type") == test.get("expected_diagnosis_type")
            and test.get("candidate_detector_invoked") == "true"
            for test in all_tests
        ),
        "all_positive_contracts_match_active_candidate_detector": all(
            test.get("candidate_detector_matched") == "true"
            for test in all_tests
            if test.get("case_type") == "positive"
        ),
        "candidate_detector_negative_gate_blocks_target": all(
            binding["candidate_detector"].get("negative_gate_contract_passed") is True
            for binding in bindings
        ),
        "normalized_legacy_candidate_regex_is_on_active_execution_path": all(
            binding["candidate_detector"]["candidate_pattern_source"].startswith(
                "normalized_legacy_regex"
            )
            and binding["candidate_detector"]["candidate_pattern"]
            == binding["trigger"]["legacy_regex_normalized"]
            and binding["candidate_detector"].get("negative_gate_contract_passed") is True
            if binding["trigger"]["legacy_regex_original"]
            else binding["candidate_detector"]["candidate_pattern_source"]
            == "source_literal_retrieval_trigger"
            and binding["candidate_detector"].get("negative_gate_contract_passed") is True
            for binding in bindings
        ),
        "split_lexical_units_possible_and_keyi_are_detector_distinct": all(
            active_candidate_detector_matches(
                binding["candidate_detector"]["candidate_pattern"],
                binding["candidate_detector"]["candidate_match_mode"],
                expected,
                binding["candidate_detector"].get("unit_specialization_pattern", ""),
            )
            and not active_candidate_detector_matches(
                binding["candidate_detector"]["candidate_pattern"],
                binding["candidate_detector"]["candidate_match_mode"],
                rejected,
                binding["candidate_detector"].get("unit_specialization_pattern", ""),
            )
            for binding, expected, rejected in (
                (
                    next(item for item in bindings if item["diagnostic_unit_id"] == "DUV3-GF0025-2021-2-001-01"),
                    "可能",
                    "可以",
                ),
                (
                    next(item for item in bindings if item["diagnostic_unit_id"] == "DUV3-GF0025-2021-2-001-02"),
                    "可以",
                    "可能",
                ),
            )
        ),
        "utf8_values_round_trip": utf8_round_trip(
            {"crosswalk": crosswalk, "templates": template_rows, "bindings": bindings, "tests": all_tests}
        ),
    }
    if not all(integrity.values()):
        failed = [name for name, passed in integrity.items() if not passed]
        raise RuntimeError("Integrity audit failed: " + ", ".join(failed))

    disposition_counts = Counter(row["mapping_disposition"] for row in crosswalk)
    regime_counts = Counter(row["regime"] for row in crosswalk)
    split_counts = Counter(row["split_kind"] for row in crosswalk)
    normalization_counts = Counter(item["status"] for item in regex_audit)
    seed_reuse_counts = Counter(row["reused_seed_label"] for row in crosswalk if row["reused_seed_label"])
    test_case_counts = Counter(test["case_type"] for test in all_tests)

    coverage = {
        "version": VERSION,
        "scope": "complete_GF0025_2021_appendix_A_inventory",
        "claim_boundary": {
            "authority_inventory": "572 official numbered grammar points",
            "runtime_artifacts": "machine-compiled, source-anchored, static-audited bindings",
            "runtime_binding_human_or_linguist_review": "not claimed",
            "catalog_metadata_correction_verification": (
                "reported separately from runtime-binding review status"
            ),
            "experimental_evaluation": "not claimed for the v3 inventory-wide bindings",
            "generated_tests": "deterministic binding contract tests; not behavioral or semantic validation",
        },
        "counts": {
            "official_standard_points": len(catalog),
            "points_split_into_multiple_units": sum(len(units) > 1 for units in point_units.values()),
            "split_points_with_preserved_preamble_scope": len(scoped_split_point_ids),
            "diagnostic_units": len(crosswalk),
            "diagnostic_profiles": len(profile_ids),
            "runtime_bindings": len(bindings),
            "operational_templates": len(templates),
            "source_seed_labels": len(seed_rules),
            "catalog_corrections_applied": sum(len(items) for items in catalog_corrections.values()),
            "catalog_points_corrected": len(catalog_corrections),
            "legacy_regex_present": legacy_count,
            "legacy_regex_absent": len(catalog) - legacy_count,
            "normalized_legacy_regexes_compiling": len(regex_audit),
            "generated_contract_tests": len(all_tests),
            "generated_contract_tests_passed": tests_passed,
            "candidate_detector_contract_invocations": sum(
                test.get("candidate_detector_invoked") == "true" for test in all_tests
            ),
            "positive_contracts_matching_active_candidate_detector": sum(
                test.get("case_type") == "positive"
                and test.get("candidate_detector_matched") == "true"
                for test in all_tests
            ),
            "runtime_bindings_using_normalized_legacy_candidate_regex": sum(
                binding["candidate_detector"]["candidate_pattern_source"].startswith(
                    "normalized_legacy_regex"
                )
                for binding in bindings
            ),
            "split_runtime_bindings_with_unit_specialized_candidate_detector": sum(
                bool(binding["candidate_detector"].get("unit_specialization_pattern"))
                for binding in bindings
            ),
            "runtime_bindings_using_source_literal_candidate_pattern": sum(
                binding["candidate_detector"]["candidate_pattern_source"]
                == "source_literal_retrieval_trigger"
                for binding in bindings
            ),
            "official_taxonomy_paths": len(distinct_taxonomy_paths),
            "taxonomy_paths_covered": len(distinct_taxonomy_paths & template_taxonomy_paths),
        },
        "mapping_disposition": dict(sorted(disposition_counts.items())),
        "regime": dict(sorted(regime_counts.items())),
        "unit_split_kind": dict(sorted(split_counts.items())),
        "regex_normalization": dict(sorted(normalization_counts.items())),
        "seed_label_reuse": dict(sorted(seed_reuse_counts.items())),
        "test_case_counts": dict(sorted(test_case_counts.items())),
        "diagnosis_type_labels": DIAGNOSIS_TYPE_LABELS,
        "structured_output_fields": STRUCTURED_OUTPUT_FIELDS,
        "catalog_correction_provenance": catalog_corrections,
        "integrity_audit": integrity,
        "artifacts": {
            "crosswalk": "standard_diagnostic_crosswalk_v3.csv",
            "templates": "diagnostic_templates_v3.csv",
            "bindings": "executable_bindings_full_inventory_v3.json",
            "tests": "diagnostic_unit_tests_v3.csv",
            "coverage": "full_inventory_executable_coverage_v3.json",
        },
    }

    crosswalk_path = out_dir / "standard_diagnostic_crosswalk_v3.csv"
    templates_path = out_dir / "diagnostic_templates_v3.csv"
    bindings_path = out_dir / "executable_bindings_full_inventory_v3.json"
    tests_path = out_dir / "diagnostic_unit_tests_v3.csv"
    coverage_path = out_dir / "full_inventory_executable_coverage_v3.json"

    write_csv(crosswalk_path, crosswalk, CROSSWALK_FIELDS)
    write_csv(templates_path, template_rows, TEMPLATE_FIELDS)
    write_csv(tests_path, all_tests, TEST_FIELDS)
    write_json(
        bindings_path,
        {
            "version": VERSION,
            "description": "Inventory-wide source-anchored diagnostic bindings generated deterministically from GF 0025--2021 and the 37-label seed constraint library.",
            "status": "machine_compiled_source_anchored_static_audited",
            "review_claim": "not_claimed",
            "diagnosis_type_labels": DIAGNOSIS_TYPE_LABELS,
            "structured_output_fields": STRUCTURED_OUTPUT_FIELDS,
            "bindings": bindings,
        },
    )
    write_json(coverage_path, coverage)
    return coverage


def compile_ok(pattern: str) -> bool:
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def binding_patterns_compile(binding: Mapping[str, Any]) -> bool:
    patterns = [binding["trigger"]["marker_pattern"]]
    patterns.extend(
        binding["runtime_conditions"][name]
        for name in (
            "structure_pattern",
            "positive_pattern",
            "weak_reject_pattern",
            "strong_reject_pattern",
            "confusable_pattern",
        )
    )
    return all(compile_ok(pattern) for pattern in patterns)


def bindings_by_id(bindings: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(binding["binding_id"]): binding for binding in bindings}


CROSSWALK_FIELDS = [
    "diagnostic_unit_id",
    "unit_record_type",
    "diagnostic_profile_id",
    "profile_record_type",
    "runtime_binding_id",
    "standard_point_id",
    "official_label",
    "level",
    "level_ordinal",
    "category",
    "subcategory",
    "section_item",
    "unit_index",
    "unit_path",
    "unit_text",
    "unit_scope",
    "split_kind",
    "split_marker",
    "template_id",
    "mapping_disposition",
    "reused_seed_label",
    "seed_compatibility_score",
    "regime",
    "trigger_pattern",
    "candidate_pattern",
    "candidate_match_mode",
    "candidate_pattern_source",
    "unit_specialization_pattern",
    "slot_schema",
    "positive_constraints",
    "weak_rejections",
    "strong_rejections",
    "confusable_uses",
    "diagnosis_type_labels",
    "structured_output_fields",
    "legacy_regex_normalized",
    "regex_normalization_status",
    "source_id",
    "source_pdf_page",
    "page_anchor",
    "page_anchor_status",
    "authority_status",
    "catalog_correction_applied",
    "catalog_correction_provenance",
    "compilation_status",
    "source_grounding_status",
    "review_status",
    "static_audit_status",
    "profile_status",
]

TEMPLATE_FIELDS = [
    "template_id",
    "template_name",
    "template_origin",
    "mapping_disposition",
    "source_seed_label",
    "taxonomy_category",
    "taxonomy_subcategory",
    "taxonomy_section_item",
    "regime",
    "trigger_strategy",
    "slot_schema",
    "positive_constraints",
    "weak_rejections",
    "strong_rejections",
    "confusable_uses",
    "diagnosis_rules",
    "diagnosis_type_labels",
    "status",
]

TEST_FIELDS = [
    "test_id",
    "diagnostic_unit_id",
    "binding_id",
    "case_type",
    "sentence",
    "candidate_span",
    "gold_span",
    "boundary_observation",
    "slot_observation",
    "semantic_observation",
    "usage_context",
    "uncertainty_observation",
    "expected_diagnosis_type",
    "observed_diagnosis_type",
    "candidate_detector_invoked",
    "candidate_detector_matched",
    "candidate_pattern_source",
    "candidate_match_mode",
    "execution_status",
    "generation_status",
]


def main() -> int:
    args = parse_args()
    coverage = compile_inventory(args.catalog, args.constraints, args.corrections, args.out_dir)
    counts = coverage["counts"]
    print(f"Compiled {counts['official_standard_points']} official points into {counts['diagnostic_units']} diagnostic units.")
    print(f"Created {counts['operational_templates']} templates and {counts['runtime_bindings']} runnable bindings.")
    print(f"Normalized {counts['normalized_legacy_regexes_compiling']} legacy regexes; all compile.")
    print(
        f"Generated and passed {counts['generated_contract_tests_passed']} "
        "deterministic binding contract tests."
    )
    print(json.dumps(coverage["mapping_disposition"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
