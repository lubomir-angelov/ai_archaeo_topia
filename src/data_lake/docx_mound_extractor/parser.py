"""Parse raw text blocks into structured DocxSample metadata.

Transform stage: applies regex patterns, keyword detection, and normalization
to convert free-form Bulgarian text into protocol-compliant fields.
"""

from __future__ import annotations

import logging
import re

from .models import DocxSample, RawTextBlock

logger = logging.getLogger(__name__)

# Relief normalization map: Bulgarian original -> protocol normalized value
RELIEF_MAP: dict[str, str] = {
    "равнинен": "plain",
    "равниен": "plain",
    "равен": "plain",
    "хълмист": "hilly",
    "слабо хълмист": "hilly",
    "полупланински": "mixed",
    "полупланински/равнинен": "mixed",
    "планински": "mountain",
    "склон": "slope",
    "склон на хълм": "slope",
    "склон на рид": "slope",
    "склон на планина": "slope",
    "склон на възвишение": "slope",
    "било на възвишение": "ridge",
    "било": "ridge",
    "било на рид": "ridge",
    "ниска част на възвишение": "ridge",
    "средна част на ниско възвишение": "ridge",
    "било на възвишение и околностите му": "ridge",
    "долна част на възвишение": "valley",
    "долен част на възвишение": "valley",
    "речна долина": "valley",
    "дере": "valley",
    "подножие на възвишение": "hilly",
    "градска среда": "urban",
    "урбанизиран терен": "urban",
    "урбанизиран": "urban",
}

# Bulgarian number words -> int
BG_NUMBERS: dict[str, int] = {
    "една": 1,
    "две": 2,
    "три": 3,
    "четири": 4,
    "пет": 5,
    "шест": 6,
    "седем": 7,
    "осем": 8,
    "девет": 9,
    "десет": 10,
    "единадесет": 11,
    "дванадесет": 12,
    "тринадесет": 13,
    "четиринадесет": 14,
    "петнадесет": 15,
    "шестнадесет": 16,
    "седемнадесет": 17,
    "деветнадесет": 18,
    "двадесет": 20,
    "петнайсет": 15,
    "шестнайсет": 16,
    "седемнайсет": 17,
    "деветнайсет": 18,
}

# Keywords that indicate hard negatives
HARD_NEGATIVE_KW = [
    "воденица",
    "мелниц",
    "резервоар",
    "полезащитен",
    "изкоп",
    "каптаж",
    "кладенец",
    "рибарник",
    "кошара",
    "табия",
    "наподобява",
    "насип",
    "рибарник",
    "свин",
]

# Keywords that indicate uncertainty
UNCERTAINTY_KW = [
    "не е могила",
    "няма могили",
    "няма знак за могила",
    "не знам",
    "наподобява",
    "може и да е",
    "вероятно",
    "не е ясен",
    "не е ясно",
]

# Keywords that indicate target is NOT present
NEGATIVE_TARGET_KW = [
    "не е могила",
    "няма могили",
    "няма знаци за могили",
    "не е могила",
    "не трябва да е могила",
]


def normalize_relief(original: str) -> str:
    """Normalize relief description per protocol section 6."""
    cleaned = original.strip().lower()
    if cleaned in RELIEF_MAP:
        return RELIEF_MAP[cleaned]
    for key, val in RELIEF_MAP.items():
        if key in cleaned or cleaned in key:
            return val
    return "unknown"


def extract_count(text: str) -> int:
    """Extract mound count from description text.

    Checks for patterns like 'от 13 могили', 'Две могили', etc.
    Returns 0 if no count found.
    """
    # Numeric count: "от 13 могили" or "над 13 могили"
    m = re.search(r"(?:от|над|повече от)\s+(\d+)\s+могили", text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Direct numeric count: "10 могили"
    m = re.search(r"\b(\d+)\s+могили", text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # "от по 7...и 15 могили" -> sum
    parts = re.findall(r"по\s+(\d+).*?и\s+(\d+).*?могили", text, re.IGNORECASE)
    if parts:
        return sum(int(x) for x in parts[0])

    # Bulgarian word count before "могили"
    for word, num in BG_NUMBERS.items():
        if re.search(rf"\b{word}\s+могили", text, re.IGNORECASE):
            return num

    # "Няколко могили" -> estimate
    if re.search(r"няколко\s+могили", text, re.IGNORECASE):
        return -1  # unknown but present

    # "Една могила"
    if re.search(r"една\s+могила", text, re.IGNORECASE):
        return 1

    return 0


def parse_text_block(block: RawTextBlock) -> dict:
    """Parse a single text block into structured fields.

    Returns a dict suitable for DocxSample construction.
    """
    text = block.raw_text

    # Sheet 25k: "Картен лист 1:25 000: К-35-39-Г-г, обл. Стара Загора"
    # Province is known Bulgarian province, matched explicitly
    sheet_25k = ""
    province = ""
    m = re.search(
        r"Картен лист\s+1:\s*(?:25\s*000|25000)\s*:\s*([А-Яа-я0-9\-()]+)\s*,?\s*обл\.\s*(.+)",
        text,
        re.IGNORECASE,
    )
    if m:
        sheet_25k = m.group(1).strip()
        raw_prov = m.group(2).strip()
        # Match against known Bulgarian provinces
        for prov in [
            "София-област",
            "София",
            "Пловдив",
            "Варна",
            "Бургас",
            "Стара Загора",
            "Хасково",
            "Пазарджик",
            "Благоевград",
            "Перник",
            "Кюстендил",
            "Враца",
            "Монтана",
            "Видин",
            "Добрич",
            "Търговище",
            "Силистра",
            "Русе",
            "Велико Търново",
            "Габрово",
            "Ловеч",
            "Монтана",
            "Плевен",
            "Ямбол",
            "Сливен",
            "Разград",
            "Шумен",
        ]:
            if prov in raw_prov:
                province = prov
                break

    # Sheet 5k: "картен лист 1:5000: К-35-39-(253)" or "картни листа 1:5000: ..."
    # Extract full sheet ID including sheet letter and number
    sheet_5k = ""
    m5k = re.search(
        r"карт[еи]?н[и]?\s+лист[а]?\s+1:\s*(?:5\s*000|5000)\s*:\s*([^\n]+)",
        text,
        re.IGNORECASE,
    )
    if m5k:
        raw = m5k.group(1).strip()
        # Sheet 5k IDs match pattern: K-35-27-(80) or L-34-142-(230)
        # May include multiple: К-35-27-(80) и К-35-27-(86) or separated by semicolons
        # Also handles bare numbers: К-35-51-(25)
        sheet_refs = re.findall(
            r"[А-Яа-яA-Ll-l]-\d{2}-\d{2,3}[-][А-Яа-яA-Zа-я]*\(\d+\)|"
            r"[А-Яа-яA-Ll-l]-\d{2}-\d{2,3}\(\d+\)",
            raw,
            re.IGNORECASE,
        )
        if sheet_refs:
            sheet_5k = " ".join(sheet_refs)
        else:
            # Fallback: extract only parenthesized numbers
            nums = re.findall(r"\((\d+(?:\s*,\s*\d+)*)\)", raw)
            sheet_5k = " ".join(nums) if nums else raw[:60]  # Safety cap

    # Position: "ЮЗ част", "СИ квадрант", etc.
    position = ""
    pos_m = re.search(
        r"((?:Северозападна|Северна|Североизточна|Източна|Югоизточна|"
        r"Южна|Югозападна|Западна|Централна|Средна|З[а]?падна|"
        r"С[е]?верна|СИ|ЮЗ|ЮИ|СЗ|Ц|Ю|З|И|С)\s+(?:квадрант|част))",
        text,
        re.IGNORECASE,
    )
    if pos_m:
        position = pos_m.group(1).strip().rstrip("–—:")

    # Relief: "Релеф: равнинен" or "Релеф - полупланински"
    relief_original = ""
    rel_m = re.search(r"Релеф[^:]*:\s*(.+)", text, re.IGNORECASE)
    if rel_m:
        relief_original = rel_m.group(1).strip().rstrip(";.,")
    else:
        rel_m2 = re.search(r"Релеф\s*-\s*(.+)", text, re.IGNORECASE)
        if rel_m2:
            relief_original = rel_m2.group(1).strip().rstrip(";.,")

    # Notes / Особенности
    notes = ""
    notes_m = re.search(r"Особеност:\s*(.+)", text, re.IGNORECASE)
    if notes_m:
        notes = notes_m.group(1).strip().rstrip(";.,")

    # Description: extract text between sheet 5k reference and relief
    # Docx text has no newlines, so find the boundaries explicitly
    desc = ""
    # Find sheet 5k reference end
    m5k_end = re.search(
        r"карт[еи]н[и]?\s+лист\s+1:\s*(?:5\s*000|5000)\s*:\s*"
        r"[А-Яа-яA-Z]-\d{2}-\d{2,3}-\(\d+(?:\s*,\s*\d+)*\)",
        text,
        re.IGNORECASE,
    )
    if m5k_end:
        # Find relief start
        m_relief = re.search(r"Релеф", text[m5k_end.end() :], re.IGNORECASE)
        if m_relief:
            desc = text[m5k_end.end() : m5k_end.end() + m_relief.start()]
        else:
            desc = text[m5k_end.end() :]
    # Remove notes from description
    desc = re.sub(r"Особеност:\s*.*$", "", desc, flags=re.IGNORECASE)
    desc = re.sub(r"\s{2,}", " ", desc)
    desc = desc.strip().strip(";.,")

    # Derived flags
    text_lower = text.lower()
    contains_necropolis = "некропол" in text_lower
    contains_hard_negative = any(kw in text_lower for kw in HARD_NEGATIVE_KW)
    uncertainty = any(kw in text_lower for kw in UNCERTAINTY_KW)
    target_present = not any(kw in text_lower for kw in NEGATIVE_TARGET_KW)

    count = extract_count(text)
    if count == -1:
        count = 0  # "няколко" -> unknown

    contains_single_mound = bool(re.search(r"една\s+могила", text, re.IGNORECASE))

    return {
        "sheet_25k": sheet_25k,
        "sheet_5k": sheet_5k,
        "province": province,
        "position_on_25k_sheet": position,
        "original_description": desc,
        "relief_original": relief_original,
        "relief_normalized": normalize_relief(relief_original) if relief_original else "unknown",
        "target_present": target_present,
        "target_count_claimed": count,
        "contains_necropolis": contains_necropolis,
        "contains_single_mound": contains_single_mound,
        "contains_hard_negative": contains_hard_negative,
        "uncertainty": uncertainty,
        "notes": notes,
    }


def assign_difficulty(sample: DocxSample) -> None:
    """Heuristic difficulty assignment per protocol section 11."""
    desc_lower = sample.original_description.lower()
    relief = sample.relief_normalized

    hard_signals = 0

    if sample.contains_necropolis:
        hard_signals += 1
    if sample.target_count_claimed >= 7:
        hard_signals += 1
    if relief in ("mountain", "slope", "ridge"):
        hard_signals += 1

    crossing_kw = ["пресечен", "минава", "грид", "паралел", "меридиан", "изолини"]
    if any(kw in desc_lower for kw in crossing_kw):
        hard_signals += 1

    noise_kw = ["молив", "оцветя", "зацапан", "размазан", "нарушен", "надраскан"]
    if any(kw in desc_lower for kw in noise_kw):
        hard_signals += 1

    if sample.contains_hard_negative:
        hard_signals += 1

    if sample.uncertainty:
        hard_signals += 1

    if hard_signals >= 4:
        sample.difficulty = "very_hard"
    elif hard_signals >= 2:
        sample.difficulty = "hard"
    elif hard_signals >= 1:
        sample.difficulty = "medium"
    else:
        sample.difficulty = "easy"
