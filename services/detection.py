"""Language detection and the script/direction metadata the formatter needs."""

from __future__ import annotations

from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect

# langdetect is probabilistic; a fixed seed keeps a given manuscript's result
# stable across runs and restarts.
DetectorFactory.seed = 0

# How much of the manuscript to feed the detector. The opening is representative
# of the book's language, and a full-length novel would otherwise dominate the
# request's runtime.
DETECTION_SAMPLE_CHARS = 50_000

UNKNOWN_LANGUAGE = "unknown"

# Latin-script languages, all left-to-right.
LATIN_LANGUAGES = frozenset({"en", "es", "fr", "de", "pt", "it", "nl"})
# Right-to-left scripts.
RTL_LANGUAGES = frozenset({"ar", "he", "ur", "fa"})
# CJK scripts, written left-to-right but needing their own line-breaking and
# glyph rules downstream.
CJK_LANGUAGES = frozenset({"zh-cn", "zh-tw", "ja", "ko"})
# Cyrillic, written left-to-right.
CYRILLIC_LANGUAGES = frozenset({"ru", "bg", "uk"})
# Greek, written left-to-right.
GREEK_LANGUAGES = frozenset({"el"})
# Devanagari, plus the scripts grouped with it here.
DEVANAGARI_LANGUAGES = frozenset({"hi", "mr", "ne", "bn"})

# langdetect emits a few codes that are aliases of, or spelled differently from,
# the codes used above.
_LANGUAGE_ALIASES = {"zh": "zh-cn", "iw": "he", "in": "id", "ji": "yi"}

LANGUAGE_NAMES = {
    # Latin
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    # RTL
    "ar": "Arabic",
    "he": "Hebrew",
    "ur": "Urdu",
    "fa": "Persian",
    # CJK
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    # Scripts with their own group above.
    "ru": "Russian",
    "uk": "Ukrainian",
    "bg": "Bulgarian",
    "el": "Greek",
    "hi": "Hindi",
    "mr": "Marathi",
    "ne": "Nepali",
    "bn": "Bengali",
    # Recognised but outside the script groups above; they still get a friendly
    # name even though they fall back to the latin script_type.
    "pl": "Polish",
    "cs": "Czech",
    "sk": "Slovak",
    "hr": "Croatian",
    "ro": "Romanian",
    "hu": "Hungarian",
    "tr": "Turkish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "ca": "Catalan",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "th": "Thai",
}


@dataclass(frozen=True)
class LanguageProfile:
    """A detected language plus the layout metadata derived from it."""

    code: str
    name: str
    text_direction: str
    script_type: str


def detect_language(text: str) -> LanguageProfile:
    """Identify `text`'s language and map it onto direction and script."""
    code = _detect_code(text)
    direction, script = _direction_and_script(code)
    return LanguageProfile(code, _name_for(code), direction, script)


def _detect_code(text: str) -> str:
    sample = text[:DETECTION_SAMPLE_CHARS].strip()
    if not sample:
        return UNKNOWN_LANGUAGE
    try:
        raw = detect(sample)
    except LangDetectException:
        # Raised when the sample holds no letters the detector recognises, e.g.
        # a file of numbers or symbols.
        return UNKNOWN_LANGUAGE
    code = raw.strip().lower().replace("_", "-")
    return _LANGUAGE_ALIASES.get(code, code)


def _direction_and_script(code: str) -> tuple[str, str]:
    if code in RTL_LANGUAGES:
        return "rtl", "rtl"
    if code in CJK_LANGUAGES:
        return "ltr", "cjk"
    if code in CYRILLIC_LANGUAGES:
        return "ltr", "cyrillic"
    if code in GREEK_LANGUAGES:
        return "ltr", "greek"
    if code in DEVANAGARI_LANGUAGES:
        return "ltr", "devanagari"
    if code in LATIN_LANGUAGES:
        return "ltr", "latin"
    # Fallback for anything unrecognised, and for languages outside these
    # groups: lay the text out as ordinary left-to-right Latin script.
    return "ltr", "latin"


def _name_for(code: str) -> str:
    if code == UNKNOWN_LANGUAGE:
        return "Unknown"
    name = LANGUAGE_NAMES.get(code)
    if name:
        return name
    # No friendly name on file, so title-case the code to keep the response
    # readable rather than leaking a bare "xx" to the client.
    return code.replace("-", " ").title()
