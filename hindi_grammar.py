"""
hindi_grammar.py — Hindi Gender Grammar Post-Processor

Fixes gender agreement errors in Romanized Hindi/Hinglish text.
Uses a 400+ word dictionary mapping nouns to grammatical gender.

Usage:
    from hindi_grammar import fix_grammar
    fixed = fix_grammar("Haan, aaj ka meeting toh hai")
    # → "Haan, aaj ki meeting toh hai"
"""

import re

# ══════════════════════════════════════════════════════════════════════════════
# HINDI GENDER DICTIONARY
# "f" = feminine (use ki, yi, nayi, acchi, thi, hui, meri, teri, hamari)
# "m" = masculine (use ka, ya, naya, accha, tha, hua, mera, tera, hamara)
# ══════════════════════════════════════════════════════════════════════════════

GENDER = {

    # ── Office / Work ─────────────────────────────────────────────────────
    "meeting": "f", "company": "f", "team": "f", "planning": "f",
    "report": "f", "call": "f", "timeline": "f", "list": "f",
    "file": "f", "presentation": "f", "sheet": "f", "mail": "f",
    "email": "f", "query": "f", "request": "f", "history": "f",
    "strategy": "f", "activity": "f", "priority": "f",
    "dependency": "f", "opportunity": "f", "quality": "f",
    "policy": "f", "salary": "f", "vacancy": "f", "duty": "f",
    "facility": "f", "delivery": "f", "entry": "f", "copy": "f",
    "supply": "f", "category": "f", "memory": "f", "gallery": "f",
    "industry": "f", "property": "f", "inventory": "f", "assembly": "f",
    "inquiry": "f", "summary": "f", "warranty": "f", "penalty": "f",
    "recovery": "f", "discovery": "f", "agency": "f", "frequency": "f",
    "currency": "f", "emergency": "f", "efficiency": "f", "accuracy": "f",
    "privacy": "f", "security": "f", "authority": "f", "capacity": "f",
    "community": "f", "university": "f", "society": "f", "library": "f",
    "directory": "f", "territory": "f", "factory": "f", "laboratory": "f",
    "ceremony": "f", "battery": "f", "survey": "f", "journey": "f",
    "story": "f", "theory": "f", "chemistry": "f", "energy": "f",
    "technology": "f", "methodology": "f", "responsibility": "f",
    "possibility": "f", "availability": "f", "visibility": "f",
    "accessibility": "f", "notification": "f", "application": "f",
    "configuration": "f", "integration": "f", "migration": "f",
    "automation": "f", "documentation": "f", "conversation": "f",
    "presentation": "f", "celebration": "f", "organization": "f",
    "information": "f", "communication": "f", "situation": "f",
    "condition": "f", "position": "f", "permission": "f",
    "discussion": "f", "session": "f", "profession": "f",
    "impression": "f", "expression": "f", "connection": "f",
    "direction": "f", "instruction": "f", "construction": "f",
    "production": "f", "introduction": "f", "suggestion": "f",
    "question": "f", "election": "f", "collection": "f",
    "protection": "f", "selection": "f", "reaction": "f",
    "transaction": "f", "attraction": "f", "satisfaction": "f",
    "performance": "f", "preference": "f", "reference": "f",
    "conference": "f", "difference": "f", "experience": "f",
    "audience": "f", "guidance": "f", "distance": "f",
    "maintenance": "f", "instance": "f", "importance": "f",
    "attendance": "f", "assistance": "f", "insurance": "f",
    "practice": "f", "service": "f", "notice": "f",
    "advice": "f", "invoice": "f", "office": "f",
    "table": "f", "trouble": "f", "issue": "m",

    "project": "m", "budget": "m", "sprint": "m", "agenda": "m",
    "client": "m", "update": "m", "status": "m",
    "task": "m", "plan": "m", "deal": "m", "ticket": "m",
    "code": "m", "server": "m", "feature": "m", "module": "m",
    "release": "m", "deployment": "m", "endpoint": "m",
    "review": "m", "milestone": "m", "blocker": "m", "risk": "m",
    "target": "m", "result": "m", "system": "m", "process": "m",
    "document": "m", "contract": "m", "product": "m", "model": "m",
    "feedback": "m", "schedule": "m", "platform": "m", "account": "m",
    "channel": "m", "dashboard": "m", "workflow": "m", "database": "m",
    "interface": "m", "component": "m", "template": "m", "program": "m",
    "test": "m", "build": "m", "error": "m", "bug": "m",
    "fix": "m", "patch": "m", "commit": "m", "branch": "m",
    "merge": "m", "backup": "m", "folder": "m", "link": "m",
    "tool": "m", "framework": "m", "package": "m", "plugin": "m",
    "widget": "m", "layout": "m", "design": "m", "format": "m",
    "method": "m", "function": "m", "parameter": "m", "variable": "m",
    "element": "m", "event": "m", "record": "m", "field": "m",
    "form": "m", "filter": "m", "graph": "m", "chart": "m",
    "token": "m", "session": "f", "alert": "m", "log": "m",
    "report": "f", "script": "m", "demo": "m", "proof": "m",
    "draft": "m", "version": "m", "sample": "m", "diagram": "m",
    "signal": "m", "source": "m", "level": "m", "role": "m",
    "goal": "m", "scope": "m", "phase": "m", "stage": "m",
    "topic": "m", "point": "m", "factor": "m", "aspect": "m",
    "concept": "m", "impact": "m", "effect": "m", "trend": "m",
    "insight": "m", "benchmark": "m", "standard": "m", "sector": "m",
    "market": "m", "profit": "m", "loss": "m", "revenue": "m",
    "expense": "m", "payment": "m", "loan": "m", "tax": "m",
    "rate": "m", "price": "m", "discount": "m", "offer": "m",
    "order": "m", "shipment": "m", "warehouse": "m",

    # ── Common Hindi words (Romanized) ────────────────────────────────────
    # Feminine
    "baat": "f", "jagah": "f", "taraf": "f", "zaroorat": "f",
    "madad": "f", "jankari": "f", "tayyari": "f", "zimmedari": "f",
    "mushkil": "f", "pareshani": "f", "dikkat": "f",
    "mehnat": "f", "koshish": "f", "ummeed": "f", "soch": "f",
    "chinta": "f", "galti": "f", "kami": "f", "takleef": "f",
    "ijazat": "f", "salah": "f", "shikayat": "f", "taarif": "f",
    "halat": "f", "suvidha": "f", "naukri": "f", "tankhah": "f",
    "chutti": "f", "subah": "f", "shaam": "f", "raat": "f",
    "dopahar": "f", "duniya": "f", "bhasha": "f", "kitab": "f",
    "kursi": "f", "mez": "f", "deewar": "f", "khidki": "f",
    "sadak": "f", "gaadi": "f", "rail": "f", "bus": "f",
    "chai": "f", "roti": "f", "sabzi": "f", "daal": "f",
    "cheez": "f", "tasveer": "f", "chabi": "f", "ghari": "f",
    "topi": "f", "jeb": "f", "zindagi": "f", "kahani": "f",
    "boli": "f", "dhoop": "f", "hawa": "f", "barish": "f",
    "aag": "f", "nadi": "f", "dharti": "f", "mitti": "f",
    "roshni": "f", "bijli": "f", "thandi": "f", "garmi": "f",
    "safai": "f", "dawai": "f", "bimaari": "f", "chot": "f",
    "takat": "f", "himmat": "f", "izzat": "f", "mohabbat": "f",
    "dosti": "f", "rishtedari": "f", "shaadi": "f", "talash": "f",
    "saza": "f", "wajah": "f", "vajah": "f", "adat": "f",
    "aadat": "f", "ibadat": "f", "shuruwat": "f", "tayaari": "f",
    "jaankari": "f", "khabar": "f", "soorat": "f", "sehat": "f",
    "umar": "f", "tasalli": "f", "khushi": "f", "udaasi": "f",
    "tanhai": "f", "awaaz": "f", "zuban": "f", "ungali": "f",
    "aankhh": "f", "naak": "f", "kamar": "f", "haddi": "f",

    # Masculine
    "kaam": "m", "din": "m", "waqt": "m", "samay": "m",
    "sawaal": "m", "jawaab": "m", "faisla": "m", "iraada": "m",
    "tajurba": "m", "natija": "m", "matlab": "m", "maqsad": "m",
    "khayal": "m", "sujhaav": "m", "badlaav": "m", "vikas": "m",
    "desh": "m", "shahar": "m", "ghar": "m", "kamra": "m",
    "darwaza": "m", "paisa": "m", "phone": "m", "computer": "m",
    "mobile": "m", "message": "m", "video": "m", "data": "m",
    "number": "m", "password": "m", "screen": "m", "network": "m",
    "khana": "m", "paani": "m", "doodh": "m", "anda": "m",
    "ped": "m", "phool": "m", "rang": "m", "kagaz": "m",
    "kapda": "m", "joota": "m", "bag": "m", "pen": "m",
    "sapna": "m", "bachpan": "m", "dhyan": "m", "gyaan": "m",
    "imtihaan": "m", "maidaan": "m", "aasmaan": "m", "toofaan": "m",
    "makaan": "m", "dukaan": "m", "bazaar": "m", "ilaaj": "m",
    "karz": "m", "kanoon": "m", "dhandha": "m", "karobaar": "m",
    "vyapaar": "m", "mausam": "m", "phal": "m", "ganna": "m",
    "chaand": "m", "suraj": "m", "taara": "m", "pahaad": "m",
    "samundar": "m", "jheel": "m", "raasta": "m", "pul": "m",
    "tala": "m", "sheesah": "m", "bartan": "m", "kapda": "m",
    "bistar": "m", "takiya": "m", "kambal": "m", "towel": "m",
    "haath": "m", "pair": "m", "sar": "m", "munh": "m",
    "kaan": "m", "dil": "m", "pet": "m", "khoon": "m",
    "dimaag": "m", "sharir": "m", "chehra": "m", "hoth": "m",
}


# ══════════════════════════════════════════════════════════════════════════════
# GRAMMAR FIXER
# ══════════════════════════════════════════════════════════════════════════════

# Precompile regex patterns for speed
_PATTERNS_F = {}  # feminine noun patterns
_PATTERNS_M = {}  # masculine noun patterns

for _word, _gender in GENDER.items():
    if _gender == "f":
        _PATTERNS_F[_word] = [
            (re.compile(rf'\bka\s+{_word}\b', re.IGNORECASE), f'ki {_word}'),
            (re.compile(rf'\bke\s+{_word}\b', re.IGNORECASE), f'ki {_word}'),
            (re.compile(rf'\b{_word}\s+tha\b', re.IGNORECASE), f'{_word} thi'),
            (re.compile(rf'\b{_word}\s+hua\b', re.IGNORECASE), f'{_word} hui'),
            (re.compile(rf'\baccha\s+{_word}\b', re.IGNORECASE), f'acchi {_word}'),
            (re.compile(rf'\bacha\s+{_word}\b', re.IGNORECASE), f'achi {_word}'),
            (re.compile(rf'\bpura\s+{_word}\b', re.IGNORECASE), f'puri {_word}'),
            (re.compile(rf'\bnaya\s+{_word}\b', re.IGNORECASE), f'nayi {_word}'),
            (re.compile(rf'\bbada\s+{_word}\b', re.IGNORECASE), f'badi {_word}'),
            (re.compile(rf'\bchhota\s+{_word}\b', re.IGNORECASE), f'chhoti {_word}'),
            (re.compile(rf'\bmera\s+{_word}\b', re.IGNORECASE), f'meri {_word}'),
            (re.compile(rf'\btera\s+{_word}\b', re.IGNORECASE), f'teri {_word}'),
            (re.compile(rf'\bhamara\s+{_word}\b', re.IGNORECASE), f'hamari {_word}'),
            (re.compile(rf'\btumhara\s+{_word}\b', re.IGNORECASE), f'tumhari {_word}'),
            (re.compile(rf'\buska\s+{_word}\b', re.IGNORECASE), f'uski {_word}'),
            (re.compile(rf'\bkonsa\s+{_word}\b', re.IGNORECASE), f'konsi {_word}'),
            (re.compile(rf'\bkaisa\s+{_word}\b', re.IGNORECASE), f'kaisi {_word}'),
        ]
    else:
        _PATTERNS_M[_word] = [
            (re.compile(rf'\bki\s+{_word}\b', re.IGNORECASE), f'ka {_word}'),
            (re.compile(rf'\b{_word}\s+thi\b', re.IGNORECASE), f'{_word} tha'),
            (re.compile(rf'\b{_word}\s+hui\b', re.IGNORECASE), f'{_word} hua'),
            (re.compile(rf'\bacchi\s+{_word}\b', re.IGNORECASE), f'accha {_word}'),
            (re.compile(rf'\bachi\s+{_word}\b', re.IGNORECASE), f'acha {_word}'),
            (re.compile(rf'\bpuri\s+{_word}\b', re.IGNORECASE), f'pura {_word}'),
            (re.compile(rf'\bnayi\s+{_word}\b', re.IGNORECASE), f'naya {_word}'),
            (re.compile(rf'\bbadi\s+{_word}\b', re.IGNORECASE), f'bada {_word}'),
            (re.compile(rf'\bchhoti\s+{_word}\b', re.IGNORECASE), f'chhota {_word}'),
            (re.compile(rf'\bmeri\s+{_word}\b', re.IGNORECASE), f'mera {_word}'),
            (re.compile(rf'\bteri\s+{_word}\b', re.IGNORECASE), f'tera {_word}'),
            (re.compile(rf'\bhamari\s+{_word}\b', re.IGNORECASE), f'hamara {_word}'),
            (re.compile(rf'\btumhari\s+{_word}\b', re.IGNORECASE), f'tumhara {_word}'),
            (re.compile(rf'\buski\s+{_word}\b', re.IGNORECASE), f'uska {_word}'),
            (re.compile(rf'\bkonsi\s+{_word}\b', re.IGNORECASE), f'konsa {_word}'),
            (re.compile(rf'\bkaisi\s+{_word}\b', re.IGNORECASE), f'kaisa {_word}'),
        ]

# Self-referential verb patterns (Sam is female)
_SELF_VERB_PATTERNS = [
    (re.compile(r'\bmain\s+karta\s+hoon\b', re.IGNORECASE), 'main karti hoon'),
    (re.compile(r'\bmain\s+karta\s+hun\b', re.IGNORECASE), 'main karti hun'),
    (re.compile(r'\bmain\s+dekhta\s+hoon\b', re.IGNORECASE), 'main dekhti hoon'),
    (re.compile(r'\bmain\s+samjhta\s+hoon\b', re.IGNORECASE), 'main samjhti hoon'),
    (re.compile(r'\bmain\s+sochta\s+hoon\b', re.IGNORECASE), 'main sochti hoon'),
    (re.compile(r'\bmain\s+jaanta\s+hoon\b', re.IGNORECASE), 'main jaanti hoon'),
    (re.compile(r'\bmain\s+rakhta\s+hoon\b', re.IGNORECASE), 'main rakhti hoon'),
    (re.compile(r'\bmain\s+batata\s+hoon\b', re.IGNORECASE), 'main bataati hoon'),
    (re.compile(r'\bmain\s+dhundhta\s+hoon\b', re.IGNORECASE), 'main dhundhti hoon'),
    (re.compile(r'\bmain\s+chahta\s+hoon\b', re.IGNORECASE), 'main chahti hoon'),
    (re.compile(r'\bmain\s+manta\s+hoon\b', re.IGNORECASE), 'main manti hoon'),
    (re.compile(r'\bmain\s+sunta\s+hoon\b', re.IGNORECASE), 'main sunti hoon'),
    (re.compile(r'\bmain\s+likhta\s+hoon\b', re.IGNORECASE), 'main likhti hoon'),
    (re.compile(r'\bmain\s+padhta\s+hoon\b', re.IGNORECASE), 'main padhti hoon'),
    (re.compile(r'\bmain\s+bolta\s+hoon\b', re.IGNORECASE), 'main bolti hoon'),
    (re.compile(r'\bmain\s+chalta\s+hoon\b', re.IGNORECASE), 'main chalti hoon'),
    (re.compile(r'\bmain\s+aata\s+hoon\b', re.IGNORECASE), 'main aati hoon'),
    (re.compile(r'\bmain\s+jaata\s+hoon\b', re.IGNORECASE), 'main jaati hoon'),
    (re.compile(r'\bmain\s+deta\s+hoon\b', re.IGNORECASE), 'main deti hoon'),
    (re.compile(r'\bmain\s+leta\s+hoon\b', re.IGNORECASE), 'main leti hoon'),
    (re.compile(r'\bmain\s+kehta\s+hoon\b', re.IGNORECASE), 'main kehti hoon'),
    (re.compile(r'\bmain\s+maanta\s+hoon\b', re.IGNORECASE), 'main maanti hoon'),
    (re.compile(r'\bmain\s+lagta\s+hoon\b', re.IGNORECASE), 'main lagti hoon'),
    (re.compile(r'\bmain\s+karta\s+tha\b', re.IGNORECASE), 'main karti thi'),
    (re.compile(r'\bmain\s+dekhta\s+tha\b', re.IGNORECASE), 'main dekhti thi'),
    (re.compile(r'\bmain\s+jaanta\s+tha\b', re.IGNORECASE), 'main jaanti thi'),
    # Catch-all: main [any English word] karta → karti
    (re.compile(r'\bmain\s+(\w+)\s+karta\b', re.IGNORECASE), r'main \1 karti'),
    # main [word] karta tha → karti thi
    (re.compile(r'\bmain\s+(\w+)\s+karta\s+tha\b', re.IGNORECASE), r'main \1 karti thi'),
]


def fix_grammar(text: str) -> str:
    """
    Fix Hindi gender agreement errors in Romanized Hindi/Hinglish text.
    Applies noun gender corrections and feminine self-referential verb fixes.
    Returns corrected text. Runs in <1ms.
    """
    result = text

    # Apply noun gender fixes
    for patterns in _PATTERNS_F.values():
        for pattern, replacement in patterns:
            result = pattern.sub(replacement, result)

    for patterns in _PATTERNS_M.values():
        for pattern, replacement in patterns:
            result = pattern.sub(replacement, result)

    # Apply self-referential verb fixes
    for pattern, replacement in _SELF_VERB_PATTERNS:
        result = pattern.sub(replacement, result)

    if result != text:
        print(f"[Grammar] Fixed: \"{text}\" → \"{result}\"")

    return result


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "Haan, aaj ka meeting toh hai CRM integrations ke liye",
        "Dekho, hamara company bahut accha hai",
        "Main check karta hoon abhi",
        "Yeh ka team ka report ready hai",
        "Mera naukri bahut accha hai",
        "Main project dekhta hoon",
        "Pura list ready hai",
        "Naya opportunity aayi hai",
        "Is ka budget ki status kya hai",
        "Hello, how are you?",  # English — should pass through unchanged
    ]
    print(f"HINDI_GENDER: {len(GENDER)} words\n")
    for t in tests:
        fixed = fix_grammar(t)
        if fixed != t:
            print(f"  ✅ \"{t}\"\n  →  \"{fixed}\"\n")
        else:
            print(f"  — \"{t}\" (no change)\n")
