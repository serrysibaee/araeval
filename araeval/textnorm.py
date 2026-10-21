"""Arabic-aware text normalisation used by answer extraction and string metrics."""
import re
import string
import unicodedata

TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
AR_PUNCT = "،؛؟«»…“”‘’٪٫٬"
PUNCT_RE = re.compile("[" + re.escape(string.punctuation + AR_PUNCT) + "]")

# Abjad order (أبجد هوز حطي كلمن سعفص), enough for 16-way classification tasks
ARABIC_LETTERS = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي", "ك", "ل", "م", "ن", "س", "ع"]
LATIN_LETTERS = list("ABCDEFGHIJKLMNOP")
# Variants a model may write for each Arabic option letter
AR_LETTER_ALIASES = {"أ": 0, "ا": 0, "إ": 0, "آ": 0, "ب": 1, "ج": 2, "د": 3, "هـ": 4, "ه": 4, "و": 5,
                     "ز": 6, "ح": 7, "ط": 8, "ي": 9, "ى": 9, "ك": 10, "ل": 11, "م": 12, "ن": 13, "س": 14, "ع": 15}


def to_ascii_digits(s: str) -> str:
    return s.translate(AR_DIGITS)


def normalize_arabic(s: str, strip_punct: bool = True) -> str:
    """Remove diacritics/tatweel, unify alef/ya/ta-marbuta, ASCII digits, lowercase, squeeze spaces."""
    s = unicodedata.normalize("NFKC", str(s))
    s = TASHKEEL.sub("", s).replace(TATWEEL, "")
    s = re.sub("[إأآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    s = to_ascii_digits(s).lower()
    if strip_punct:
        s = PUNCT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str) -> list[str]:
    return normalize_arabic(s).split()


def letter_index(ans: str) -> int | None:
    """Map 'A'/'b'/'أ'/'هـ'/'(ج)' to a 0-based option index."""
    a = str(ans).strip().strip("()[]{}.:،- ")
    if not a:
        return None
    if a in AR_LETTER_ALIASES:
        return AR_LETTER_ALIASES[a]
    if a[0] in AR_LETTER_ALIASES and (len(a) == 1 or a == "هـ"):
        return AR_LETTER_ALIASES[a[0]]
    if len(a) == 1 and a.upper() in LATIN_LETTERS:
        return LATIN_LETTERS.index(a.upper())
    return None
