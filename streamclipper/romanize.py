"""Devanagari -> Roman letters (casual Hinglish style, e.g. "mai kyu aau nahi tu ja"), for when Whisper writes Hindi in Devanagari.

Rule-based: handles matras, conjuncts, nukta, anusvara and the common schwa deletions
(word-final, and the usual medial case). English loanwords Whisper writes in Devanagari
come from a small table; anything else is fixed later by the M5 glossary.
"""
import re

CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
    "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
    "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
    "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "\u0958": "q", "\u0959": "kh", "\u095a": "g", "\u095b": "z", "\u095c": "d", "\u095d": "dh",
    "\u095e": "f", "\u095f": "y",  # precomposed nukta letters
}
NUKTA = "\u093c"
NUKTA_MAP = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "d", "ढ": "dh", "फ": "f", "य": "y"}
INDEP = {"अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e",
         "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
MATRA = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai",
         "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e"}
VIRAMA, ANUSVARA, CHANDRABINDU, VISARGA = "\u094d", "\u0902", "\u0901", "\u0903"
LABIAL = set("पफबभम")
DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
RUN = re.compile(r"[\u0900-\u0963\u0966-\u097f]+")

# Common English words Whisper writes in Devanagari when it hears Hinglish.
LOANWORDS = {
    "गेम": "game", "गेमिंग": "gaming", "गेमर": "gamer", "प्लेयर": "player", "लाइव": "live",
    "स्ट्रीम": "stream", "स्ट्रीमर": "streamer", "वीडियो": "video", "चैनल": "channel",
    "सब्सक्राइब": "subscribe", "लाइक": "like", "शेयर": "share", "कमेंट": "comment", "लेवल": "level",
    "बॉस": "boss", "फाइट": "fight", "मैच": "match", "टीम": "team", "फ्रेंड": "friend",
    "मोबाइल": "mobile", "फोन": "phone", "इंटरनेट": "internet", "ऑनलाइन": "online",
    "ऑफलाइन": "offline", "कंट्रोलर": "controller", "ग्राफिक्स": "graphics", "स्कोर": "score",
    "ओके": "okay", "सॉरी": "sorry", "प्लीज": "please", "थैंक्स": "thanks", "हेलो": "hello",
    "गाइज": "guys", "स्टीम": "steam", "सर्वर": "server", "लैग": "lag", "रैंक": "rank",
    "वर्ल्ड": "world", "मोड": "mode", "ट्रेलर": "trailer", "रिव्यू": "review", "अपडेट": "update",
    "बग": "bug", "क्लिप": "clip", "रील": "reel", "वाओ": "wow", "सुपर": "super", "नाइस": "nice",
}


def has_devanagari(text: str) -> bool:
    return bool(RUN.search(text))


def _unit(c="", v=None, cons=False):
    return {"c": c, "v": v, "tail": "", "cons": cons}


def _romanize_word(w: str) -> str:
    w = w.replace("ज्ञ", "ग्य")  # jna -> gya, as spoken in Hindi
    units, i, n = [], 0, len(w)
    while i < n:
        ch = w[i]
        if ch in CONS:
            c = CONS[ch]
            if i + 1 < n and w[i + 1] == NUKTA:
                c, i = NUKTA_MAP.get(ch, c), i + 1
            units.append(_unit(c, None, True))
        elif ch in INDEP:
            units.append(_unit("", INDEP[ch]))
        elif ch in MATRA:
            if units and units[-1]["cons"] and units[-1]["v"] is None:
                units[-1]["v"] = MATRA[ch]
            else:
                units.append(_unit("", MATRA[ch]))
        elif ch == VIRAMA:
            if units and units[-1]["v"] is None:
                units[-1]["v"] = ""
        elif ch in (ANUSVARA, CHANDRABINDU):
            nxt = w[i + 1] if i + 1 < n else ""
            if not units:
                units.append(_unit())
            units[-1]["tail"] += "m" if nxt in LABIAL else "n"
        elif ch == VISARGA:
            if not units:
                units.append(_unit())
            units[-1]["tail"] += "h"
        elif ch != NUKTA:
            units.append(_unit(ch, ""))
        i += 1

    # Schwa deletion, right to left so "voiced" neighbours are already decided:
    # word-final schwa is silent ("kamal"); a medial one drops between voiced syllables
    # ("matlab", "samajhna", "ladki") unless the previous syllable ends in a nasal ("zindagi").
    last = len(units) - 1
    for idx in range(last, -1, -1):
        u = units[idx]
        if not u["cons"] or u["v"] is not None:
            continue
        if idx == last:
            u["v"] = "" if len(units) > 1 else "a"
            continue
        prev = units[idx - 1] if idx > 0 else None
        prev_ok = prev is not None and prev["v"] != "" and not prev["tail"]
        u["v"] = "" if (prev_ok and units[idx + 1]["v"] != "") else "a"
    # Casual spelling: a final nasal on a vowel is dropped, as people type it (nahi, mai, hai, aau)
    if units and units[last]["tail"] and units[last]["v"]:
        units[last]["tail"] = units[last]["tail"].replace("n", "").replace("m", "")
    return "".join(u["c"] + (u["v"] or "") + u["tail"] for u in units)


# Words where the plain rules give a stiff or odd spelling; these are how people type them.
CASUAL = {"क्यों": "kyu", "में": "mein", "कुछ": "kuch", "अच्छा": "accha", "अच्छी": "acchi",
          "अच्छे": "acche", "ठीक": "theek", "यार": "yaar", "बहुत": "bahut"}
# Roman spellings Whisper may produce itself, folded to the same casual style.
VARIANTS = {"kyun": "kyu", "kyon": "kyu", "nahin": "nahi", "hain": "hai", "aaun": "aau"}
_VARIANT_RE = re.compile(r"\b(" + "|".join(VARIANTS) + r")\b", re.I)


def casual_variants(text: str) -> str:
    def fix(m):
        new = VARIANTS[m.group().lower()]
        return new.capitalize() if m.group()[0].isupper() else new
    return _VARIANT_RE.sub(fix, text)


def romanize(text: str) -> str:
    text = text.replace("।", ".").replace("॥", ".").translate(DIGITS)
    return RUN.sub(lambda m: LOANWORDS.get(m.group()) or CASUAL.get(m.group()) or _romanize_word(m.group()), text)
