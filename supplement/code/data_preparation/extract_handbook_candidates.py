import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import fitz


VOLUME_METADATA = [
    {
        "argument": "elementary",
        "level": "初等",
        "title": "国际中文教育中文水平等级标准·语法学习手册（初等）",
        "source_id": "SRC-GRAMMAR-HANDBOOK-ELEM-2022",
    },
    {
        "argument": "intermediate",
        "level": "中等",
        "title": "国际中文教育中文水平等级标准·语法学习手册（中等）",
        "source_id": "SRC-GRAMMAR-HANDBOOK-INTER-2022",
    },
    {
        "argument": "advanced",
        "level": "高等",
        "title": "国际中文教育中文水平等级标准·语法学习手册（高等）",
        "source_id": "SRC-GRAMMAR-HANDBOOK-ADV-2022",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elementary", type=Path, required=True)
    parser.add_argument("--intermediate", type=Path, required=True)
    parser.add_argument("--advanced", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260629)
    return parser.parse_args()


def build_pdf_records(args):
    records = []
    for metadata in VOLUME_METADATA:
        path = getattr(args, metadata["argument"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({
            **metadata,
            "path": path,
            "authors": "Chinese Testing International and volume editors",
            "publisher": "Beijing Language and Culture University Press",
            "year": "2022",
        })
    return records

FAMILY_PATTERNS = {
    "C1_DISPOSAL_PATIENT_REORDERING": [
        ("BA", r"把[^，。！？；;]{1,20}(?:了|完|好|到|成|在|给|走|来|去|出来|起来|下来|下去|上来|进去)?"),
        ("JIANG_DISPOSAL", r"将[^，。！？；;]{1,24}(?:交给|放在|变成|作为|用于|保存|送到|改成)"),
    ],
    "C2_PASSIVE_PATIENT_PROMINENCE": [
        ("BEI", r"被[^，。！？；;]{1,32}"),
    ],
    "C3_COMPARATIVE_CONSTRUCTION": [
        ("BI", r"比[^，。！？；;]{1,20}"),
        ("MEIYOU_COMPARE", r"没有[^，。！？；;]{1,18}(?:那么|这么|更|高|大|好|快|多|少|容易|方便)?"),
        ("BURU", r"不如[^，。！？；;]{1,18}"),
        ("YIYANG", r"(?:跟|和|与)[^，。！？；;]{1,16}一样"),
        ("YUEYUE", r"越[^，。！？；;]{1,10}越[^，。！？；;]{1,16}"),
    ],
    "C4_COMPLEMENT_STRUCTURE": [
        ("DE_COMPLEMENT", r"得[^，。！？；;]{1,18}"),
        ("RESULT_COMPLEMENT", r"(?:写完|做好|看见|听懂|学会|吃饱|弄清楚|准备好|收好|办好|打开|关上|提高|降低|变成|成为)"),
        ("DIRECTION_COMPLEMENT", r"(?:跑进|走进|拿出|说出|看出来|站起来|坐下|走过去|赶到|赶去|回到|送到|搬到|放到|来到|起来|下去|出来|进去)"),
        ("POTENTIAL_COMPLEMENT", r"(?:看得懂|听得懂|吃得下|买不起|做不了|去不了|来得及|来不及|说得出来|说不出来)"),
        ("QUANTITY_COMPLEMENT", r"(?:[一二两三四五六七八九十0-9]+(?:次|遍|趟|回|下|年|个月|天|小时|分钟))"),
    ],
    "C5_MULTI_VERB_ARGUMENT_STRUCTURE": [
        ("LIANDONG", r"(?:去|来|上|到|回)[^，。！？；;]{0,8}(?:买|看|找|参加|学习|上课|旅游|访问)"),
        ("JIANYU", r"(?:请|让|叫|派|使|劝|通知|邀请)[^，。！？；;]{1,12}(?:去|来|做|参加|学习|办理|帮助|完成)"),
        ("DOUBLE_OBJECT", r"(?:给|送|教|告诉|问|借|还)[^，。！？；;]{1,10}[^，。！？；;]{1,10}"),
    ],
    "C6_PREPOSITIONAL_FRAME": [
        ("PREP_ZAI", r"在[^，。！？；;]{1,18}"),
        ("PREP_CONG", r"从[^，。！？；;]{1,18}"),
        ("PREP_XIANG", r"(?:向|往|朝)[^，。！？；;]{1,18}"),
        ("PREP_DUI", r"对[^，。！？；;]{1,18}"),
        ("PREP_GEI_WEI", r"(?:给|为|关于|按照|由于)[^，。！？；;]{1,18}"),
    ],
    "C8_COMPLEX_SENTENCE_CONNECTIVE": [
        ("YINWEI_SUOYI", r"因为[^。！？；;]{1,40}所以"),
        ("SUIRAN_DANSHI", r"虽然[^。！？；;]{1,40}(?:但是|可是|但)"),
        ("RUGUO_JIU", r"如果[^。！？；;]{1,40}就"),
        ("YIBIAN_YIBIAN", r"一边[^。！？；;]{1,30}一边"),
        ("BUDAN_ERQIE", r"不但[^。！？；;]{1,40}而且"),
        ("ZHISUOYI_SHIYINWEI", r"之所以[^。！？；;]{1,40}是因为"),
        ("YINWEI", r"因为[^。！？；;]{1,36}"),
        ("SUOYI", r"所以[^。！？；;]{1,30}"),
        ("SUIRAN", r"虽然[^。！？；;]{1,36}"),
        ("DANSHI", r"但是[^。！？；;]{1,32}"),
        ("RUGUO", r"如果[^。！？；;]{1,36}"),
        ("ZHIYAO", r"只要[^。！？；;]{1,36}"),
        ("JIRAN", r"既然[^。！？；;]{1,36}"),
        ("JISHI", r"即使[^。！？；;]{1,36}"),
    ],
}

PAGE_TOPIC_HINTS = {
    "C1_DISPOSAL_PATIENT_REORDERING": ["把字句", "处置", "把"],
    "C2_PASSIVE_PATIENT_PROMINENCE": ["被字句", "被动", "被"],
    "C3_COMPARATIVE_CONSTRUCTION": ["比较句", "A比B", "没有B", "不如", "一样", "越"],
    "C4_COMPLEMENT_STRUCTURE": ["结果补语", "趋向补语", "可能补语", "时量补语", "动量补语", "程度补语"],
    "C5_MULTI_VERB_ARGUMENT_STRUCTURE": ["连动", "兼语", "双宾"],
    "C6_PREPOSITIONAL_FRAME": ["介词", "引出时间", "引出处所", "引出对象", "引出方向", "引出原因"],
    "C8_COMPLEX_SENTENCE_CONNECTIVE": ["复句", "关联词", "因为", "虽然", "如果", "一边"],
}


def normalize(text):
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def clean_sentence(s):
    s = normalize(s)
    s = re.sub(r"^(例句|交际实践|补充例句|小提示|妈妈|爸爸|老师[A-Z]?|学生|A|B|记者|专家|儿子|孩子|司机|姐姐|弟弟)[：:]\s*", "", s)
    s = s.strip(" ①②③④⑤⑥⑦⑧⑨⑩◎●•")
    s = re.sub(r"\s+", "", s)
    return s


def split_sentences(text):
    chunks = re.split(r"(?<=[。！？；;])|\n", text)
    out = []
    for c in chunks:
        c = clean_sentence(c)
        if not c:
            continue
        if re.search(r"[A-Za-z]\+|[A-Za-z]、|VP|Adj|Prep|N方位|S\+", c):
            continue
        if any(mark in c for mark in ["◎", "【", "】", "=", "⋯⋯", "...", "基本语义", "结构特点", "小提示"]):
            continue
        if re.match(r"^\d+[\.\、]?", c):
            continue
        if not re.search(r"[\u4e00-\u9fff]", c):
            continue
        if len(c) < 6 or len(c) > 85:
            continue
        if re.search(r"[A-Za-z]{8,}", c):
            continue
        if c.count("*") > 0:
            # Keep erroneous forms only if they still contain a target pattern.
            c = c.replace("*", "")
        out.append(c)
    return out


def page_hint(text, family):
    return "；".join([h for h in PAGE_TOPIC_HINTS[family] if h in text][:4])


def find_candidate(sentence, family):
    for grammar_point, pat in FAMILY_PATTERNS[family]:
        m = re.search(pat, sentence)
        if m:
            span = m.group(0)
            span = span[:28]
            return grammar_point, span
    return None, None


def has_any_pattern(sentence, family):
    gp, span = find_candidate(sentence, family)
    return bool(span)


def extract(pdfs):
    candidates = []
    seen = set()
    for meta in pdfs:
        doc = fitz.open(str(meta["path"]))
        for page_idx, page in enumerate(doc, start=1):
            if page_idx <= 12:
                continue
            text = normalize(page.get_text("text"))
            if not text:
                continue
            sentences = split_sentences(text)
            for family in FAMILY_PATTERNS:
                hint = page_hint(text, family)
                page_is_relevant = bool(hint)
                for s in sentences:
                    gp, span = find_candidate(s, family)
                    if not span:
                        continue
                    # Avoid C6 swallowing every locative if the page is clearly a different structural topic.
                    if family == "C6_PREPOSITIONAL_FRAME" and not page_is_relevant and len(candidates) > 0:
                        if not re.search(r"^(在|从|向|往|朝|对|给|为|关于|按照|由于)", span):
                            continue
                    key = (family, meta["source_id"], page_idx, s, span)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append({
                        "construction_family": family,
                        "grammar_point": gp,
                        "sentence": s,
                        "candidate_span": span,
                        "page": page_idx,
                        "grammar_page_clue": hint,
                        "source_id": meta["source_id"],
                        "source_type": "权威语法学习手册",
                        "level_book": meta["level"],
                        "book_title": meta["title"],
                        "authors_editors": meta["authors"],
                        "publisher": meta["publisher"],
                        "year": meta["year"],
                        "source_file_or_pool": meta["path"].name,
                        "candidate_generation_method": f"pdf_page_regex:{gp}",
                        "source_status": "AUTH_READY",
                        "annotation_ready": "是",
                        "prefill_note": "自动从本地权威语法学习手册 PDF 抽取；需人工正式判断。",
                    })
    return candidates


def main():
    args = parse_args()
    candidates = extract(build_pdf_records(args))
    # Deduplicate by sentence + family + span across repeated handbook contexts.
    uniq = {}
    for c in candidates:
        key = (c["construction_family"], c["sentence"], c["candidate_span"])
        if key not in uniq:
            uniq[key] = c
        else:
            # Prefer entries with a page topic hint.
            if c["grammar_page_clue"] and not uniq[key]["grammar_page_clue"]:
                uniq[key] = c
    candidates = list(uniq.values())

    random.Random(args.seed).shuffle(candidates)
    counts = Counter(c["construction_family"] for c in candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("total", len(candidates))
    print(json.dumps(dict(sorted(counts.items())), ensure_ascii=False, indent=2))
    for family in sorted(counts):
        sample = [c for c in candidates if c["construction_family"] == family][:3]
        print("\n", family)
        for c in sample:
            print(c["page"], c["grammar_point"], c["candidate_span"], c["sentence"][:80])


if __name__ == "__main__":
    main()
