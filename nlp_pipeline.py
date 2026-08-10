"""
NLP pipeline: cleaning, tokenization/normalization, topic classification, and
rule-based NER for Thai/English disaster & accident alert posts.
"""
import re

try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    from pythainlp.util import normalize as th_normalize
    PYTHAINLP_OK = True
    THAI_STOPWORDS = thai_stopwords()
except Exception:
    PYTHAINLP_OK = False
    THAI_STOPWORDS = set()

EN_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "in",
    "on", "at", "to", "of", "for", "and", "or", "but", "with", "as", "by",
    "this", "that", "it", "from", "has", "have", "had", "will", "would",
    "there", "their", "we", "you", "he", "she", "they", "i", "his", "her",
}

# ---------- Regex & Cleansing ----------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\S+")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F000-\U0001F0FF☀-⛿]+"
)
HOTLINE_RE = re.compile(r"\b(1669|191|199|1784|1554|1646|1155|911)\b")
PHONE_RE = re.compile(r"\b0\d{1,2}[- ]?\d{3}[- ]?\d{3,4}\b")


def clean_text(text):
    noise = {
        "urls": URL_RE.findall(text),
        "mentions": MENTION_RE.findall(text),
        "hashtags": HASHTAG_RE.findall(text),
        "hotlines": list(set(HOTLINE_RE.findall(text))),
        "phone_numbers": list(set(PHONE_RE.findall(text))),
    }
    cleaned = URL_RE.sub(" ", text)
    cleaned = MENTION_RE.sub(" ", cleaned)
    cleaned = HASHTAG_RE.sub(" ", cleaned)
    cleaned = EMOJI_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, noise


def detect_language(text):
    thai_chars = len(re.findall(r"[฀-๿]", text))
    en_chars = len(re.findall(r"[A-Za-z]", text))
    if thai_chars == 0 and en_chars == 0:
        return "unknown"
    if thai_chars > 0 and en_chars > 0 and min(thai_chars, en_chars) / max(thai_chars, en_chars) > 0.2:
        return "mixed"
    return "th" if thai_chars >= en_chars else "en"


# ---------- Tokenization & Normalization ----------
def normalize_text(text, lang):
    if lang in ("th", "mixed") and PYTHAINLP_OK:
        text = th_normalize(text)
    # reduce repeated/elongated characters, e.g. "soooo" -> "soo", "มากกกก" -> "มากก"
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text


def tokenize(text, lang):
    if lang in ("th", "mixed") and PYTHAINLP_OK:
        tokens = word_tokenize(text, engine="newmm")
    else:
        tokens = re.findall(r"[A-Za-z']+|[฀-๿]+|\d+", text)
    return [t for t in tokens if t.strip()]


def remove_stopwords(tokens, lang):
    stopwords = THAI_STOPWORDS if lang in ("th", "mixed") else EN_STOPWORDS
    return [t for t in tokens if t.lower() not in stopwords and t not in stopwords and t.strip(" ,.") != ""]


# ---------- Topic Identification ----------
TOPIC_KEYWORDS = {
    "ไฟไหม้ / Fire": ["ไฟไหม้", "เพลิงไหม้", "ไฟลุก", "ไฟลุกไหม้", "fire", "burning", "blaze", "flames"],
    "น้ำท่วม / Flood": ["น้ำท่วม", "น้ำป่า", "น้ำหลาก", "flood", "flooding", "flash flood"],
    "อุบัติเหตุจราจร / Traffic accident": [
        "รถชน", "อุบัติเหตุ", "รถคว่ำ", "รถเสียหลัก", "ชนกัน", "รถพลิกคว่ำ",
        "accident", "crash", "collision", "car crash", "pile-up",
    ],
    "ตึกถล่ม/แผ่นดินไหว / Collapse-Earthquake": [
        "ตึกถล่ม", "อาคารถล่ม", "แผ่นดินไหว", "สิ่งปลูกสร้างถล่ม", "earthquake", "collapse", "collapsed",
    ],
    "ระเบิด / Explosion": ["ระเบิด", "explosion", "blast"],
    "อาชญากรรม / Crime": [
        "ปล้น", "จี้", "ทำร้าย", "ฆาตกรรม", "โจรกรรม", "ยิง",
        "robbery", "assault", "murder", "shooting", "stabbed",
    ],
}


def classify_topic(text):
    text_low = text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        count = sum(text_low.count(kw.lower()) for kw in keywords)
        if count:
            scores[topic] = count
    if not scores:
        return "อื่นๆ / Other", {"อื่นๆ / Other": 1}
    total = sum(scores.values())
    percentages = {k: round(v / total * 100, 1) for k, v in scores.items()}
    top_topic = max(scores, key=scores.get)
    return top_topic, percentages


# ---------- POS & NER (rule/dictionary based) ----------
THAI_PROVINCES = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร", "ขอนแก่น", "จันทบุรี",
    "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง", "ตราด",
    "ตาก", "นครนายก", "นครปฐม", "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี",
    "นราธิวาส", "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี",
    "พระนครศรีอยุธยา", "พะเยา", "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
    "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน", "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง",
    "ระยอง", "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน", "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล",
    "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี", "สุโขทัย",
    "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ",
    "อุดรธานี", "อุตรดิตถ์", "อุทัยธานี", "อุบลราชธานี",
]

LOCATION_PATTERN_RE = re.compile(
    r"(ถนน|ซอย|แยก|สะพาน|หมู่บ้าน|ตำบล|อำเภอ|ตึก|ห้าง)\s?([^\s,.\d]{1,20})"
)
KM_MARKER_RE = re.compile(r"กม\.?\s?\d+")

AGENCY_KEYWORDS = [
    "มูลนิธิร่วมกตัญญู", "มูลนิธิป่อเต็กตึ๊ง", "หน่วยกู้ภัย", "หน่วยกู้ชีพ", "ตำรวจจราจร", "ตำรวจ",
    "สภ.", "โรงพยาบาล", "รพ.", "เทศบาล", "อบต.", "อบจ.", "หน่วยดับเพลิง", "ดับเพลิง", "ทหาร",
    "ปภ.", "กรมทางหลวง", "การไฟฟ้า", "การประปา", "มูลนิธิกู้ภัย",
    "police", "fire department", "hospital", "rescue", "red cross", "ems", "highway police",
]

THAI_MONTHS = [
    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน", "กรกฎาคม",
    "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
]
THAI_WEEKDAYS = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
RELATIVE_TIME_WORDS = [
    "เมื่อคืน", "เมื่อวาน", "วันนี้", "เช้านี้", "บ่ายนี้", "ค่ำนี้", "ล่าสุด",
    "today", "yesterday", "last night", "this morning", "tonight",
]
EN_MONTHS = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]
EN_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

TIME_RE = re.compile(r"\b\d{1,2}[.:]\d{2}\s*(น\.?|am|pm|AM|PM)?\b")
DATE_TH_RE = re.compile(r"วันที่\s?\d{1,2}\s?(เดือน)?\s?(" + "|".join(THAI_MONTHS) + r")?")
DATE_EN_RE = re.compile(
    r"\b(" + "|".join(EN_MONTHS) + r")\s+\d{1,2}(st|nd|rd|th)?\b", re.IGNORECASE
)

CASUALTY_INJURED_TH_RE = re.compile(r"(บาดเจ็บ|เจ็บ)\D{0,6}(\d+)\s*(ราย|คน)")
CASUALTY_DEAD_TH_RE = re.compile(r"(เสียชีวิต|ตาย|ดับ)\D{0,6}(\d+)\s*(ราย|คน)")
CASUALTY_INJURED_EN_RE = re.compile(r"(\d+)\s*(people\s*)?(were\s*|was\s*)?(injured|hurt|wounded)", re.IGNORECASE)
CASUALTY_DEAD_EN_RE = re.compile(r"(\d+)\s*(people\s*)?(were\s*|was\s*)?(dead|killed|died)", re.IGNORECASE)


def extract_locations(text):
    locations = set()
    for province in THAI_PROVINCES:
        if province in text:
            locations.add(province)
    for match in LOCATION_PATTERN_RE.finditer(text):
        loc = (match.group(1) + match.group(2)).strip()
        if loc:
            locations.add(loc)
    for match in KM_MARKER_RE.finditer(text):
        locations.add(match.group(0))
    return sorted(locations)


def extract_datetime(text):
    found = set()
    for match in TIME_RE.finditer(text):
        found.add(match.group(0).strip())
    for match in DATE_TH_RE.finditer(text):
        if match.group(0).strip() != "วันที่":
            found.add(match.group(0).strip())
    for match in DATE_EN_RE.finditer(text):
        found.add(match.group(0).strip())
    text_low = text.lower()
    for day in THAI_WEEKDAYS:
        if day in text:
            found.add(day)
    for day in EN_WEEKDAYS:
        if day in text_low:
            found.add(day)
    for word in RELATIVE_TIME_WORDS:
        if word.lower() in text_low:
            found.add(word)
    return sorted(found)


def extract_casualties(text):
    injured, dead, notes = None, None, []

    for rx in (CASUALTY_INJURED_TH_RE, CASUALTY_INJURED_EN_RE):
        m = rx.search(text)
        if m:
            injured = int(next(g for g in m.groups() if g and g.isdigit()))
            break
    for rx in (CASUALTY_DEAD_TH_RE, CASUALTY_DEAD_EN_RE):
        m = rx.search(text)
        if m:
            dead = int(next(g for g in m.groups() if g and g.isdigit()))
            break

    if injured is None and re.search(r"บาดเจ็บ|injured|wounded|hurt", text, re.IGNORECASE):
        injured = "ไม่ระบุจำนวน / unspecified"
    if dead is None and re.search(r"เสียชีวิต|ตาย|ดับ|dead|killed|died", text, re.IGNORECASE):
        dead = "ไม่ระบุจำนวน / unspecified"
    if injured is None and dead is None:
        notes.append("ไม่พบข้อมูลผู้บาดเจ็บ/เสียชีวิตในข้อความ")

    return {"injured": injured, "dead": dead, "notes": notes}


def extract_agencies(text):
    text_low = text.lower()
    found = set()
    for kw in AGENCY_KEYWORDS:
        if kw.lower() in text_low:
            found.add(kw)
    return sorted(found)


def extract_entities(text):
    return {
        "locations": extract_locations(text),
        "datetime_mentions": extract_datetime(text),
        "casualties": extract_casualties(text),
        "agencies": extract_agencies(text),
    }


# ---------- Full pipeline ----------
def analyze(text):
    lang = detect_language(text)
    cleaned, noise = clean_text(text)
    normalized = normalize_text(cleaned, lang)
    tokens = tokenize(normalized, lang)
    tokens_no_stop = remove_stopwords(tokens, lang)
    topic, topic_scores = classify_topic(text)
    entities = extract_entities(text)

    severity = "แจ้งเตือน / Alert"
    dead = entities["casualties"]["dead"]
    injured = entities["casualties"]["injured"]
    if dead not in (None,):
        severity = "วิกฤต / Critical"
    elif injured not in (None,):
        severity = "เร่งด่วน / Urgent"

    return {
        "language": lang,
        "original_text": text,
        "cleaned_text": cleaned,
        "normalized_text": normalized,
        "noise_removed": noise,
        "tokens": tokens,
        "tokens_no_stopwords": tokens_no_stop,
        "topic": topic,
        "topic_scores": topic_scores,
        "entities": entities,
        "severity": severity,
    }
