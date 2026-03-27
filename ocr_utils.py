
from __future__ import annotations

import re
import functools
import importlib as _importlib
import importlib.util as _importlib_util
import logging
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QUANTITY UNITS (Global constants for OCR parsing)
# ---------------------------------------------------------------------------
QUANTITY_UNITS = (
    r"kg|kgs?|kilogram|kilograms?"
    r"|g|gm|gms?|gr|gram|grams?"
    r"|ltr|ltrs?|liter|liters?|litre|litres?|lit(?!\w)"
    r"|ml|milliliter|millilitre"
    r"|pcs?|piece|pieces?"
    r"|pkt|pkts?|packet|packets?"
    r"|box|boxes"
    r"|bottle|bottles?"
    r"|can|cans?"
    r"|bunch|bunches?"
    r"|dozen|doz"
    r"|bag|bags?"
    r"|pack|packs?"
    r"|roll|rolls?"
    r"|jar|jars?"
    r"|strip|strips?"
    r"|unit|units?"
    r"|no|nos"
    r"|tray|trays?"
    r"|sachet|sachets?"
    r"|tin|tins?"
    r"|loaf|loaves"
    r"|litr"
    r"|pk|pkt"
    r"|pints?"
)

_QTY_FULL_RE = re.compile(
    r"^(?:"
    r"x\s*\d+\.?\d*"
    r"|\d+\.?\d*\s*x"
    r"|[×✕]\s*\d+\.?\d*"
    r"|\d+\.?\d*\s*[×✕]"
    r"|\d+\.?\d*\s*(?:" + QUANTITY_UNITS + r")"
    r")$",
    re.IGNORECASE,
)
_UNIT_ONLY_RE  = re.compile(r"^(?:" + QUANTITY_UNITS + r")$", re.IGNORECASE)
_PRICE_RE      = re.compile(
    r"^(₹|rs\.?|inr|usd|\$|£|€|aed|%)?\s*\d{1,6}(\.\d{1,2})?$"
    r"|^\d{1,6}\.\d{2}$",
    re.IGNORECASE,
)
_BARE_NUM_RE   = re.compile(r"^\d+\.?\d*$")
_WEIGHT_QTY_RE = re.compile(r"^\d{1,3}\.\d{1,3}$")


# ---------------------------------------------------------------------------
# AUTO-INSTALL HELPER
# ---------------------------------------------------------------------------

AUTO_INSTALL: bool = True   # set False to suppress pip calls


def _ensure(package: str, import_name: str | None = None) -> bool:
    """
    Return True if *import_name* (defaults to *package*) is importable.
    When AUTO_INSTALL is True and the package is absent, attempt pip install.
    """
    import_name = import_name or package
    if _importlib_util.find_spec(import_name) is not None:
        return True
    if not AUTO_INSTALL:
        return False
    logger.info("Auto-installing %s …", package)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return _importlib_util.find_spec(import_name) is not None
    except Exception as exc:
        logger.warning("Could not install %s: %s", package, exc)
        return False


# ---------------------------------------------------------------------------
# LIBRARY BOOTSTRAP
# ---------------------------------------------------------------------------

# ── inflect ──────────────────────────────────────────────────────────────────
INFLECT_AVAILABLE: bool = _ensure("inflect")
if INFLECT_AVAILABLE:
    import inflect as _inflect_lib          # type: ignore
    _inflect_engine = _inflect_lib.engine()

    def _to_singular(word: str) -> str:
        singular = _inflect_engine.singular_noun(word.lower())
        return singular if singular else word.lower()

else:
    _FALLBACK_IRREGULARS: dict[str, str] = {
        "leaves": "leaf", "knives": "knife", "lives": "life",
        "wolves": "wolf", "halves": "half", "loaves": "loaf",
        "shelves": "shelf", "calves": "calf", "scarves": "scarf",
        "children": "child", "men": "man", "women": "woman",
        "teeth": "tooth", "feet": "foot", "mice": "mouse",
        "geese": "goose", "oxen": "ox",
        "tomatoes": "tomato", "potatoes": "potato", "mangoes": "mango",
        "avocados": "avocado", "heroes": "hero", "echoes": "echo",
        "cookies": "cookie", "brownies": "brownie", "smoothies": "smoothie",
    }

    def _to_singular(word: str) -> str:
        w = word.lower()
        if not w or len(w) < 3:
            return w
        if not w.endswith("s"):
            return w
        if w in _FALLBACK_IRREGULARS:
            return _FALLBACK_IRREGULARS[w]
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("ves") and len(w) > 4:
            return w[:-3] + "f"
        if re.search(r"(ss|x|z|ch|sh)es$", w):
            return w[:-1]
        if w.endswith("es") and len(w) > 4:
            stem_s  = w[:-1]
            stem_es = w[:-2]
            if stem_s.endswith("e"):
                return stem_s
            if stem_es and stem_es[-1] in "aeiou":
                return stem_es
            return stem_s
        if not w.endswith("ss") and len(w) > 3:
            return w[:-1]
        return w


# ── spaCy ────────────────────────────────────────────────────────────────────
SPACY_AVAILABLE: bool = _ensure("spacy")
_nlp = None
if SPACY_AVAILABLE:
    try:
        import spacy as _spacy              # type: ignore

        def _load_spacy_model():
            try:
                return _spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except OSError:
                from spacy.cli import download as _dl
                _dl("en_core_web_sm")
                return _spacy.load("en_core_web_sm", disable=["parser", "ner"])

        _nlp = _load_spacy_model()
    except Exception as _spacy_err:
        logger.warning("spaCy model load failed: %s", _spacy_err)
        SPACY_AVAILABLE = False


def _lemmatize(text: str) -> str:
    if not SPACY_AVAILABLE or _nlp is None:
        return text
    doc = _nlp(text)
    parts = []
    for tok in doc:
        if tok.is_alpha and tok.lemma_ and tok.lemma_ != "-PRON-":
            parts.append(tok.lemma_.lower())
        else:
            parts.append(tok.text.lower())
    return " ".join(parts)


# ── rapidfuzz ────────────────────────────────────────────────────────────────
RAPIDFUZZ_AVAILABLE: bool = _ensure("rapidfuzz")
if RAPIDFUZZ_AVAILABLE:
    from rapidfuzz import fuzz as _rfuzz, process as _rprocess  # type: ignore
else:
    import difflib as _difflib

    class _rfuzz:                               # type: ignore[no-redef]
        @staticmethod
        def token_set_ratio(a: str, b: str) -> float:
            aw = sorted(set(a.lower().split()))
            bw = sorted(set(b.lower().split()))
            return 100 * _difflib.SequenceMatcher(
                None, " ".join(aw), " ".join(bw)
            ).ratio()

    class _rprocess:                            # type: ignore[no-redef]
        @staticmethod
        def extractOne(query, choices, scorer=None, score_cutoff=0):
            best, best_score = None, 0.0
            for c in choices:
                sc = _rfuzz.token_set_ratio(query, c)
                if sc > best_score:
                    best, best_score = c, sc
            return (best, best_score, None) if best_score >= score_cutoff else None


# ── indic_transliteration ─────────────────────────────────────────────────────
_indic_sanscript = None
_itrans          = None
INDIC_TRANS_AVAILABLE: bool = False

if _ensure("indic-transliteration", "indic_transliteration"):
    try:
        _importlib.import_module("indic_transliteration")
        _indic_sanscript = _importlib.import_module("indic_transliteration.sanscript")
        _itrans = getattr(_indic_sanscript, "transliterate", None)
        INDIC_TRANS_AVAILABLE = _itrans is not None
    except Exception as _it_err:
        logger.warning("indic_transliteration load failed: %s", _it_err)


# ── nltk / WordNet ───────────────────────────────────────────────────────────
_wn   = None
_nltk = None
WORDNET_AVAILABLE: bool = False

if _ensure("nltk"):
    try:
        _nltk        = _importlib.import_module("nltk")
        _nltk_corpus = _importlib.import_module("nltk.corpus")
        _wn          = getattr(_nltk_corpus, "wordnet", None)
        if _wn is not None:
            try:
                _wn.synsets("food")
            except Exception:
                _nltk.download("wordnet", quiet=True)
                _nltk.download("omw-1.4",  quiet=True)
                _nltk_corpus = _importlib.import_module("nltk.corpus")
                _wn = getattr(_nltk_corpus, "wordnet", None)
            WORDNET_AVAILABLE = _wn is not None
    except Exception as _nltk_err:
        logger.warning("nltk load failed: %s", _nltk_err)


# ---------------------------------------------------------------------------
# FOOD VOCABULARY
# ---------------------------------------------------------------------------

_FOOD_SEEDS: list[str] = [
    "potato", "onion", "tomato", "capsicum", "carrot", "cabbage",
    "cauliflower", "broccoli", "spinach", "beetroot", "radish",
    "pumpkin", "okra", "eggplant", "corn", "sweet corn", "peas",
    "green peas", "green beans", "french bean", "cluster bean", "drumstick",
    "garlic", "ginger", "shallot", "spring onion",
    "curry leaf", "coriander leaf", "fenugreek leaf", "mint leaf",
    "bottle gourd", "bitter gourd", "ridge gourd", "ivy gourd",
    "colocasia", "yam", "raw banana", "raw papaya",
    "green chilli", "dry red chilli", "red chilli", "bell pepper", "leek",
    "celery", "asparagus", "zucchini", "mushroom", "avocado",
    "jackfruit", "lotus stem", "baby corn", "sorrel leaves",
    "lemon juice", "lime juice", "cowpea",
    "mango", "apple", "banana", "orange", "grapes", "guava",
    "papaya", "pineapple", "watermelon", "pomegranate",
    "kiwi", "strawberry", "blueberry", "cherry", "plum",
    "peach", "pear", "coconut", "lemon", "lime", "dates", "fig",
    "tamarind", "custard apple", "dragon fruit", "sapota",
    "milk", "yogurt", "curd", "paneer", "butter", "ghee", "clarified butter",
    "cream", "cheese", "cottage cheese", "buttermilk",
    "coconut milk", "coconut water", "almond milk", "condensed milk",
    "egg", "chicken", "chicken breast", "mutton", "lamb", "beef", "pork",
    "fish", "fish fillet", "prawn", "prawns", "shrimp", "salmon", "tuna",
    "rice", "basmati rice", "sona masoori rice", "wheat flour", "all purpose flour",
    "rice flour", "chickpea flour", "gram flour", "besan", "semolina", "oats",
    "millet", "foxtail millet", "pearl millet", "finger millet", "barley",
    "quinoa", "flattened rice", "bread", "roti", "chapati",
    "pasta", "noodles", "vermicelli", "puffed rice", "sago", "idli rice",
    "moong dal", "masoor dal", "toor dal", "chana dal", "urad dal",
    "bengal gram", "black gram",
    "kidney beans", "chickpeas", "kabuli chana", "lentils", "cowpea",
    "black chickpeas", "black gram", "besan", "gram flour", "chickpea flour",
    "moong sprout", "green moong sprouts",
    "urad papad",
    "cumin", "coriander", "coriander seeds", "turmeric", "red chilli powder",
    "black pepper", "cardamom", "cloves", "cinnamon", "fennel seeds",
    "mustard seeds", "fenugreek seeds", "methi seeds", "asafoetida",
    "bay leaf", "garam masala", "sambar powder", "chaat masala",
    "goda masala", "salt", "black salt", "tamarind paste",
    "dry mango powder", "pickle", "chutney", "green chutney", "sweet chutney",
    "sunflower oil", "groundnut oil", "mustard oil", "coconut oil",
    "olive oil", "sesame oil", "rice bran oil", "cooking oil",
    "sugar", "jaggery", "honey",
    "almonds", "cashews", "pistachios", "walnuts", "peanuts",
    "raisins", "fox nuts", "chia seeds", "flaxseed", "dry fruit mix",
    "tomato ketchup", "soy sauce", "vinegar", "mayonnaise",
    "jam", "peanut butter", "tomato puree",
    "biscuit", "cracker", "chips", "popcorn", "cornflakes",
    "chocolate", "croissant", "muffin", "marie biscuit",
    "banana chips", "jackfruit chips", "pani puri", "papad", "boondi",
    "kachori", "dhokla mix", "idli dosa batter", "breakfast cereal",
    "tea", "coffee", "green tea", "juice", "mineral water",
    "dishwash", "detergent", "toothpaste", "shampoo", "soap",
    "mosquito repellent", "dish soap", "camphor",
    "protein powder", "isabgol", "chamomile",
    "baking soda", "baking powder", "yeast", "vanilla essence",
    "desiccated coconut", "food colour",
    # FIX: added common items that were being dropped
    "fresh cream", "amul cream", "cooking cream",
    "pure ghee", "cow ghee", "buffalo ghee",
    "full cream milk", "toned milk", "skimmed milk",
    "greek yogurt", "hung curd",
    # Descriptor compound forms (JioMart / grocery receipts)
    "soy chunks", "soy protein",
    "black peppercorn", "black peppercorns", "peppercorn", "peppercorns",
    "cardamom pods", "cinnamon sticks", "star anise pods", "bay leaves",
]


def _wordnet_expand(seeds: list[str]) -> list[str]:
    """Automatically expand the seed vocabulary using WordNet hyponyms."""
    if not WORDNET_AVAILABLE or _wn is None:
        return seeds
    food_synsets = (
        list(_wn.synsets("food",          pos=_wn.NOUN))
        + list(_wn.synsets("ingredient",  pos=_wn.NOUN))
        + list(_wn.synsets("vegetable",   pos=_wn.NOUN))
        + list(_wn.synsets("fruit",       pos=_wn.NOUN))
        + list(_wn.synsets("grain",       pos=_wn.NOUN))
        + list(_wn.synsets("spice",       pos=_wn.NOUN))
        + list(_wn.synsets("dairy_product", pos=_wn.NOUN))
        + list(_wn.synsets("meat",        pos=_wn.NOUN))
        + list(_wn.synsets("seafood",     pos=_wn.NOUN))
    )
    wn_terms: set[str] = set()
    visited:  set      = set()
    queue = list(food_synsets)
    depth: dict = {s: 0 for s in food_synsets}
    while queue:
        syn = queue.pop(0)
        if syn in visited or depth[syn] > 3:
            continue
        visited.add(syn)
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if len(name) >= 3 and name.isascii():
                wn_terms.add(name)
        for hypo in syn.hyponyms():
            if hypo not in depth:
                depth[hypo] = depth[syn] + 1
            queue.append(hypo)
    return list(dict.fromkeys(seeds + sorted(wn_terms)))


FOOD_VOCAB: list[str]    = _wordnet_expand(_FOOD_SEEDS)
_FOOD_VOCAB_SET: set[str] = set(FOOD_VOCAB)

# ── Thresholds ────────────────────────────────────────────────────────────────
# FIX: lowered from 85/88 — OCR typos typically score 75-84, were being dropped
FUZZY_MATCH_THRESHOLD: int = 78   # rapidfuzz token_set_ratio  (was 85)
_DIFFLIB_THRESHOLD:    int = 82   # stdlib SequenceMatcher      (was 88)


@functools.lru_cache(maxsize=4096)
def _fuzzy_match_food(query: str) -> Optional[str]:
    if not query or len(query) < 3:
        return None
    if query in _FOOD_VOCAB_SET:
        return query
    threshold = FUZZY_MATCH_THRESHOLD if RAPIDFUZZ_AVAILABLE else _DIFFLIB_THRESHOLD
    choices = _rprocess.extract(
        query, FOOD_VOCAB,
        scorer=_rfuzz.token_set_ratio,
        score_cutoff=threshold,
        limit=5
    )
    if not choices:
        return None
    # Tie-break: Prefer longer matches that exist in the query
    choices.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    return choices[0][0]




# ---------------------------------------------------------------------------
# TRANSLITERATION MAP
# ---------------------------------------------------------------------------

_TRANS_SEEDS: dict[str, str] = {
    # Hindi vegetables
    "aloo": "potato", "alu": "potato", "aaloo": "potato",
    "pyaz": "onion", "piyaz": "onion", "kanda": "onion",
    "tamatar": "tomato",
    "bhindi": "okra", "vendakkai": "okra",
    "baingan": "eggplant", "brinjal": "eggplant", "vangi": "eggplant",
    "karela": "bitter gourd",
    "lauki": "bottle gourd", "ghiya": "bottle gourd",
    "turai": "ridge gourd", "tori": "ridge gourd",
    "tindora": "ivy gourd", "kovakkai": "ivy gourd",
    "arbi": "taro", "arvi": "taro",
    "suran": "yam", "jimikand": "yam",
    "saag": "spinach", "palak": "spinach",
    "methi": "fenugreek seeds",
    "shimla mirch": "capsicum",
    "lal mirch": "red chilli", "lal mirchi": "red chilli",
    "hari mirch": "green chilli", "hari mirchi": "green chilli",
    "kali mirch": "black pepper",
    "kaddu": "pumpkin",
    "sem": "broad bean",
    "gawar": "cluster bean",
    "parwal": "pointed gourd",
    "kathal": "jackfruit",
    # Hindi spices
    "haldi": "turmeric",
    "jeera": "cumin", "zeera": "cumin", "jira": "cumin",
    "dhaniya": "coriander", "dhania": "coriander",
    "hing": "asafoetida",
    "ajwain": "carom seeds",
    "imli": "tamarind",
    "amchur": "dry mango powder",
    "elaichi": "cardamom", "ilaichi": "cardamom",
    "laung": "cloves", "lavang": "cloves",
    "dalchini": "cinnamon",
    "tej patta": "bay leaf",
    "saunf": "fennel seeds", "badishep": "fennel seeds",
    "rai": "mustard seeds",
    "kalonji": "nigella seeds",
    "chakra phool": "star anise",
    "javitri": "mace", "jaiphal": "nutmeg",
    # Hindi grains / flours
    
    "sooji": "semolina", "rava": "semolina", "suji": "semolina",
    "chawal": "rice", "chaawal": "rice",
    "basmati": "basmati rice",
    "poha": "flattened rice", "avalakki": "flattened rice",
    "sabudana": "sago", "sabudan": "sago",
    "bajra": "pearl millet",
    "jowar": "sorghum",
    "ragi": "finger millet",
    # Hindi pulses
    "rajma": "kidney beans",
    "kabuli chana": "chickpeas", "chole": "chickpeas",
    "kala chana": "black chickpeas",
    "black gram": "black gram",
    # Hindi dairy & misc
    "doodh": "milk", "dudh": "milk",
    "dahi": "yogurt",
    "paneer": "paneer",
    "makkhan": "butter",
    "shakkar": "sugar", "cheeni": "sugar",
    "namak": "salt",
    "tel": "oil",
    # Hindi dry fruits
    "badam": "almonds", "badaam": "almonds",
    "kaju": "cashews",
    "pista": "pistachios",
    "akhrot": "walnuts",
    "moongfali": "peanuts", "mungfali": "peanuts",
    "kismis": "raisins", "kishmish": "raisins",
    "makhana": "fox nuts",
    # OCR corruptions / typos
    "supar": "sugar",
    "jeer": "cumin",
    "avccado": "avocado", "avccados": "avocado",
    "bestroot": "beetroot",
    "soojj": "semolina",
    "ra": "mustard seeds",
    # Compound item protections
    "chicken breast":      "chicken breast",
    "chicken breasts":     "chicken breast",
    "fish fillet":         "fish fillet",
    "fish fillets":        "fish fillet",
    "coconut milk":        "coconut milk",
    "coconut water":       "coconut water",
    "foxtail millet":      "foxtail millet",
    "basmati rice":        "basmati rice",
    "sona masoori rice":   "sona masoori rice",
    "sona masoori":        "sona masoori rice",
    "prawn":               "prawn",
    "prawns":              "prawn",
    "green beans":         "green beans",
    "green peas":          "green peas",
    "green chilli":        "green chilli",
    "green chillies":      "green chilli",
    "curry leaf":          "curry leaf",
    "curry leaves":        "curry leaf",
    "coriander leaf":      "coriander leaf",
    "coriander leaves":    "coriander leaf",
    "dry red chilli":      "dry red chilli",
    "dry red chillies":    "dry red chilli",
    "red chilli powder":   "red chilli powder",
    "mustard seeds":       "mustard seeds",
    "methi seeds":         "fenugreek seeds",
    "fenugreek seeds":     "fenugreek seeds",
    "coriander seeds":     "coriander seeds",
    "fennel seeds":        "fennel seeds",
    "fenugreek seeds":     "fenugreek seeds",
    "mustard seeds":       "mustard seeds",
    "sesame seeds":        "sesame seeds",
    "cumin seeds":         "cumin seeds",
    "rice flour":          "rice flour",
    "wheat flour":         "wheat flour",
    "gram flour":          "gram flour",
    "all purpose flour":   "all purpose flour",
    "maida":               "all purpose flour",
    "atta":                "wheat flour",
    "black pepper":        "black pepper",
    "black salt":          "black salt",
    "black gram":          "black gram",
    "mustard seeds":       "mustard seeds",
    "methi seeds":         "fenugreek seeds",
    "coriander seeds":     "coriander seeds",
    "fennel seeds":        "fennel seeds",
    "cooking oil":         "cooking oil",
    "lemon juice":         "lemon juice",
    "sweet chutney":       "sweet chutney",
    "green chutney":       "green chutney",
    "dry fruit mix":       "dry fruit mix",
    "dry fruits mix":      "dry fruit mix",
    "breakfast cereal":    "breakfast cereal",
    "tomato ketchup":      "tomato ketchup",
    "urad papad":          "urad papad",
    "idli dosa batter":    "idli dosa batter",
    "green moong sprouts": "green moong sprouts",
    "green moong dal":     "moong dal",
    "moong sprouts":       "moong sprout",
    "sorrel leaves":       "sorrel leaves",
    "sweet corn":          "sweet corn",
    "garam masala":        "garam masala",
    "goda masala":         "goda masala",
    "chaat masala":        "chaat masala",
    # Descriptor variants — preserve compound names from receipts
    "soy chunks":          "soy chunks",
    "soya chunks":         "soy chunks",
    "textured soy":        "soy chunks",
    "black peppercorns":   "black peppercorn",
    "black peppercorn":    "black peppercorn",
    "peppercorns":         "peppercorn",
    "cardamom pods":       "cardamom",
    "cardamom pod":        "cardamom",
    "cinnamon sticks":     "cinnamon",
    "cinnamon stick":      "cinnamon",
    "bay leaves":          "bay leaf",
    "star anise pods":     "star anise",
    "star anises":         "star anise",
    # D-Mart Image 10: brand + code → canonical item
    "l moong dal":         "moong dal",
    "l latur turdal":      "toor dal",
    "latur turdal":        "toor dal",
    "latur turda":         "toor dal",
    "turdal":              "toor dal",
    "l sugar":             "sugar",
    "l masoor":            "masoor dal",
    "satyam masurda":      "masoor dal",
    "satyam masoor":       "masoor dal",
    "satyam modng":        "moong dal",
    "satyam moong":        "moong dal",
    "lijjat udad pa":      "urad papad",
    "lijjat udad":         "urad papad",
    "udad pa":             "urad papad",
    "dkjeera":             "cumin",
    "dkjeera powd":        "cumin",
    "dk jeera":            "cumin",
    "dk jeera powd":       "cumin",
    "dkhaldi":             "turmeric",
    "dkhaldi powd":        "turmeric",
    "dkhaldi powo":        "turmeric",
    "dk haldi":            "turmeric",
    "dk haldi powd":       "turmeric",
    "dkgaram":             "garam masala",
    "dkgaram masa":        "garam masala",
    "dk garam masa":       "garam masala",
    "dk goda masala":      "goda masala",
    "saffola gold":        "sunflower oil",
    "saffola gold bl":     "sunflower oil",
    "pdha jad":            "flattened rice",
    "pdha jada":           "flattened rice",
    "hem soham cham":      "chamomile",
    "hem soham":           "chamomile",
    "shree sidhivin":      "vinegar",
    "shree sidhiv":        "vinegar",
    "actii chilli":        "red chilli",
    "actii chilli su":     "red chilli",
    "kissan tomato":       "tomato ketchup",
    "veeba eggless":       "mayonnaise",
    "veeba":               "mayonnaise",
    "garden mini dh":      "dhokla mix",
    "garden bh":           "dhokla mix",
    "garden dhokla":       "dhokla mix",
    "chhedas kachdr":      "kachori",
    "chhedas kachori":     "kachori",
    "kelloggs cho":        "cornflakes",
    "kelloggs choc":       "cornflakes",
    "kelloggs cho v":      "cornflakes",
    "britannia vita":      "marie biscuit",
    "britannia vitam":     "marie biscuit",
    "parle krackjac":      "cracker",
    "parle krackjack":     "cracker",
    "parle g glucos":      "biscuit",
    "parle g":             "biscuit",
    "mangalam":            "camphor",
    "mangalam pure":       "camphor",
    "colgate str th":      "toothpaste",
    "colgate str":         "toothpaste",
    "exo ant rdund":       "dish soap",
    "exo ant round":       "dish soap",
    "sunny green":         "coconut oil",
    "sunny green pre":     "coconut oil",
    "layer nogirl":        "soap",
    "surf excel":          "detergent",
    "rin ala fabri":       "detergent",
    "comfort fab":         "detergent",
    "harpic powerps":      "detergent",
    "park avenue":         "soap",
    # Image 11: handwritten/printed receipt
    "beet root":                   "beetroot",
    "brinjal black":               "eggplant",
    "brinjal black big":           "eggplant",
    "p ginger":                    "ginger",
    "sambhar onion":               "shallot",
    "sambhar onion small":         "shallot",
    "sambhar onion small pack":    "shallot",
    "garlic indian":               "garlic",
    "string less bean":            "french bean",
    "string less beans":           "french bean",
    "stringless bean":             "french bean",
    "stringless beans":            "french bean",
    "banana njalipooyan":          "banana",
    "chilli green":                "green chilli",
    "orange kg":                   "orange",
    "cabbage ea":                  "cabbage",
    "cowpea kuttipayar":           "cowpea",
    "cowpea kuttipa":              "cowpea",
}


def _build_transliteration_map(seeds: dict[str, str]) -> dict[str, str]:
    """
    Automatically expand the seed map with:
      • plural forms of values (via inflect or naive +s)
      • plural forms of keys
      • qualifier-stripped variants
      • prefix-truncated OCR variants
      • Devanagari script forms (via indic_transliteration if installed)
    """
    result: dict[str, str] = dict(seeds)
    _QUALIFIER_WORDS = {
        "green", "white", "red", "black", "yellow", "whole", "split",
        "roast", "roasted", "raw", "dried", "fresh", "powder", "powd",
        "indian", "local", "loose", "packed", "mini", "large", "small",
        "skinned", "skinless",
    }

    for k, v in list(seeds.items()):
        if INFLECT_AVAILABLE:
            pl = _inflect_engine.plural(v)
            if pl and pl != v:
                result.setdefault(pl, v)
        else:
            result.setdefault(v + "s", v)
        result.setdefault(k + "s", v)

    for k, v in list(seeds.items()):
        parts   = k.split()
        stripped = [p for p in parts if p not in _QUALIFIER_WORDS]
        if stripped and stripped != parts:
            stripped_key = " ".join(stripped)
            # FIX: do not map "salt" to "black salt" if "salt" is already a valid standalone item.
            # ALSO: do not use bare units (gram, kg, ltr) as expansion keys.
            is_unit = bool(_UNIT_ONLY_RE.match(stripped_key))
            if stripped_key not in _FOOD_SEEDS and not is_unit:
                result.setdefault(stripped_key, v)

    for k, v in list(seeds.items()):
        if len(k) > 6 and len(k[:-1]) >= 3:
            result.setdefault(k[:-1], v)
        if len(k) > 8 and len(k[:-2]) >= 3:
            result.setdefault(k[:-2], v)

    if INDIC_TRANS_AVAILABLE and _itrans is not None and _indic_sanscript is not None:
        _HK         = getattr(_indic_sanscript, "HK",         "hk")
        _DEVANAGARI = getattr(_indic_sanscript, "DEVANAGARI", "devanagari")
        for k, v in list(seeds.items()):
            try:
                dev = _itrans(k, _HK, _DEVANAGARI)
                if dev != k:
                    result.setdefault(dev, v)
            except Exception:
                pass
    return result


TRANSLITERATION_MAP: dict[str, str] = _build_transliteration_map(_TRANS_SEEDS)
_MULTI_WORD_TRANS  = {k: v for k, v in TRANSLITERATION_MAP.items() if " " in k}
_SINGLE_WORD_TRANS = {k: v for k, v in TRANSLITERATION_MAP.items() if " " not in k}


def _apply_transliteration(text: str) -> str:
    """Apply transliteration map, checking 4/3/2/1-word sequences."""
    words  = text.lower().split()
    result: list[str] = []
    i, n   = 0, len(words)
    while i < n:
        matched = False
        if not matched and i + 4 <= n:
            four = " ".join(words[i:i + 4])
            if four in _MULTI_WORD_TRANS:
                result.append(_MULTI_WORD_TRANS[four]); i += 4; matched = True
        if not matched and i + 3 <= n:
            three = " ".join(words[i:i + 3])
            if three in _MULTI_WORD_TRANS:
                result.append(_MULTI_WORD_TRANS[three]); i += 3; matched = True
        if not matched and i + 2 <= n:
            two = " ".join(words[i:i + 2])
            if two in _MULTI_WORD_TRANS:
                result.append(_MULTI_WORD_TRANS[two]); i += 2; matched = True
        if not matched:
            result.append(_SINGLE_WORD_TRANS.get(words[i], words[i])); i += 1
    return " ".join(result)


# ---------------------------------------------------------------------------
# BRAND NORMALISER
# ---------------------------------------------------------------------------

_KNOWN_BRANDS: set[str] = {
    "kissan", "veeba", "parle", "britannia", "kelloggs", "chhedas",
    "garden", "lijjat", "saffola", "pdha", "hem", "shree", "actii",
    "satyam", "latur", "mangalam", "dk", "harpic", "colgate",
    "surf", "rin", "comfort", "exo", "sunny", "layer", "park",
    "protinex", "glucon", "maggi", "nestle", "haldirams", "amul",
    "dkhaldi", "dkjeera", "dkgaram", "avenue",
}

_SINGLE_CHAR_BRAND_PREFIXES: set[str] = {"l"}


def _strip_brand_prefix(text: str) -> str:
    words = text.lower().split()
    while words and words[0] in _KNOWN_BRANDS:
        words.pop(0)
    if len(words) >= 2 and words[0] in _SINGLE_CHAR_BRAND_PREFIXES:
        words.pop(0)
    return " ".join(words)


def _resolve_brand_item(text: str) -> Optional[str]:
    first    = text.split()[0].lower() if text.split() else ""
    is_brand = (
        first in _KNOWN_BRANDS
        or (first in _SINGLE_CHAR_BRAND_PREFIXES and len(text.split()) >= 2)
    )
    if not is_brand:
        return None
    remainder = _strip_brand_prefix(text)
    if not remainder or len(remainder) < 3:
        return None
    if remainder in TRANSLITERATION_MAP:
        return TRANSLITERATION_MAP[remainder]
    return _fuzzy_match_food(remainder)


# ---------------------------------------------------------------------------
# SKIP SETS
# ---------------------------------------------------------------------------

_SKIP_TAXONOMY: dict[str, list[str]] = {
    "totals": [
        "total", "subtotal", "grand total", "net amount", "amount due",
        "balance due", "amount payable", "sub total", "net total", "bill total",
        "order total", "paid", "ordertotal", "subtotal", "grandtotal",
        "total amount", "final amount", "payment",
    ],
    "charges": [
        "delivery charge", "delivery charges", "delivery fee",
        "shipping charge", "shipping fee", "shipping cost",
    ],
    "taxes": ["tax", "gst", "vat", "cgst", "sgst", "igst", "cess", "gst breakup"],
    "table_headers": [
        "mrp", "rate", "price", "unit price", "sp", "dp",
        "qty", "qnty", "quantity", "nos",
        "item", "items", "description", "product", "particulars",
        "value", "net", "hsn", "code", "n/rate", "qty/kg",
        "sr", "sno", "s.no", "no.", "#",
        "hsn code", "item description", "net price",
    ],
    "payment": ["paid", "cash", "card", "upi", "online", "wallet", "change"],
    "document": [
        "invoice", "bill", "receipt", "order", "summary", "slip",
        "order summary", "order details", "order confirmation",
        "track order", "track your order", "order no", "invoice no",
        "bill no", "receipt no", "jio mart", "jiomart", "jiomart.com",
        "paid", "pay", "bial pay",
    ],
    "navigation": [
        "grocery list", "shopping list", "my list",
        "date", "time", "store", "branch", "outlet", "counter",
        "cashier", "server", "table", "waiter", "operator",
    ],
    "section_headers": [
        "vegetables & greens", "vegetables and greens",
        "fruits & vegetables", "fruits and vegetables",
        "meat & seafood", "meat and seafood",
        "dairy & refrigerated", "dairy and refrigerated",
        "dairy products", "grains pulses and flours",
        "grains, pulses & flours", "grains pulses flours",
        "oils and liquids", "spices and seasonings", "snacks and beverages",
        "bakery & confectionery", "frozen foods", "household items",
        "personal care", "idli and dosa essentials", "other ingredients",
        "dairy refrigerate", "dairy refrigerated", "dairy section",
        "frozen section", "chilled section", "refrigerated items",
        "bakery section", "produce section", "meat section",
        "grain pulse flour", "grain pulse", "grains pulses",
    ],
    "stores_india": [
        "dmart", "d-mart", "big bazaar", "reliance fresh", "reliance smart",
        "more supermarket", "spar", "spencers", "nilgiris", "heritage fresh",
        "star bazaar", "blinkit", "zepto", "swiggy instamart", "dunzo",
        "grofers", "bigbasket", "amazon fresh", "jiomart", "lulu",
        "carrefour", "hypercity", "vishal megamart",
    ],
    "stores_global": [
        "walmart", "tesco", "costco", "kroger", "aldi", "lidl",
        "sainsburys", "asda", "morrisons", "waitrose", "m&s",
        "whole foods", "trader joe", "target", "publix", "wegmans", "safeway",
    ],
    "social": [
        "thank you", "thanks", "welcome", "customer copy",
        "thank you for shopping", "thank you visit again",
        "please visit", "we accept", "follow us",
    ],
    "app_ui_labels": [
        "order total", "add ingredient", "inventory health",
        "register item", "item name", "quantity unit",
        "use soon", "urgent", "expired",
        "expiry", "status", "action", "qtyng rate value",
        "qtyng", "inventory", "health",
        # FIX: removed bare "fresh" from app_ui_labels — it is a food word
    ],
}


def _build_skip_exact(taxonomy: dict[str, list[str]]) -> set[str]:
    out: set[str] = set()
    for terms in taxonomy.values():
        for t in terms:
            t_l = t.lower().strip()
            if not t_l:
                continue
            out.add(t_l)
            out.add(re.sub(r"[^\w\s]", "", t_l).strip())
            out.add(t_l.replace("&", "and"))
            out.add(t_l.replace("and", "&"))
    return out


_SKIP_EXACT: set[str] = _build_skip_exact(_SKIP_TAXONOMY)

_SKIP_PREFIXES: tuple[str, ...] = (
    "www.", "http", "fssai", "gstin:", "cin:", "gstin ",
    "phone:", "mob:", "mobile:", "email:", "e-mail:",
    "+91", "+1", "+44", "+971",
    "address:", "add:", "date:", "time:", "page ",
    "order no", "order id", "order #", "order date",
    "invoice no", "bill no", "receipt no", "vou. no",
    "delivery:", "estimated delivery", "expected delivery",
    "table ", "server:", "cashier:", "waiter:", "cover:", "seat ",
    "sub total", "net total", "bill total",
    "plot no", "flat no", "shop no",
)

_HEADER_WORDS: set[str] = {
    "item", "items", "qty", "quantity", "price", "rate", "mrp",
    "amount", "description", "product", "particulars",
    "nos", "sr", "sno", "s.no", "no.", "#", "total",
    "value", "net", "hsn", "code", "n/rate", "qty/kg",
}

_BILLING_SUFFIX_RE = re.compile(
    r"\b("
    r"delivery\s+charge[s]?|delivery\s+fee"
    r"|service\s+charge[s]?|service\s+fee"
    r"|convenience\s+fee|platform\s+fee|handling\s+fee"
    r"|shipping\s+charge[s]?|packing\s+charge[s]?"
    r"|packaging\s+charge[s]?|packaging\s+fee"
    r"|cgst|sgst|igst|gst|vat|cess|surcharge"
    r"|discount|cashback|coupon|promo\s+code"
    r"|gratuity|toll\s+charge|levy"
    r")\s*$",
    re.IGNORECASE,
)

_BILLING_SUBSTRINGS: tuple[str, ...] = (
    "delivery charge", "delivery fee",
    "service charge", "service fee",
    "convenience fee", "platform fee",
    "handling fee", "shipping charge",
    "packing charge", "packaging charge",
    "packaging fee", "discount applied",
    "coupon applied", "promo applied",
)

# FIX: removed "fresh", "value", "net" — these appear in legitimate food names.
# Only keep words that are unambiguously non-food billing terms.
_SKIP_WORD_SET: frozenset[str] = frozenset({
    "total", "subtotal", "sub", "grand", "paid",
    "tax", "gst", "cgst", "sgst", "igst", "vat", "cess",
    "amount", "balance", "due", "payable", "bill", "receipt",
    "invoice", "order", "summary", "discount", "cashback",
    "charge", "charges", "fee", "fees", "delivery", "shipping",
    "handling", "packing", "packaging", "surcharge", "levy",
    "mrp", "rate", "price", "qty", "quantity", "nos",
    "sr", "sno", "description",
    "hsn", "code", "coupon", "promo",
    "drage", "bial", "bitral", "pay",
})

_COLUMN_BLEED_TOKENS: set[str] = set()

# FIX: removed "fresh", "natural", "pure" from noise seeds.
# These words form part of common product names (Fresh Cream, Pure Ghee, etc.)
# and stripping them hollows out the item name entirely.
_NOISE_SEEDS: set[str] = {
    "small", "large", "medium", "big", "mini",
    "extra", "special",
    "regular", "standard", "imported", "seasonal",
    "raw", "frozen", "chilled", "dried", "loose", "packed",
    "branded", "buy", "get", "offer", "new",
}


def _auto_extend_noise_words(seeds: set[str]) -> set[str]:
    """Automatically discover more noise adjectives via spaCy POS tagging."""
    if not SPACY_AVAILABLE or _nlp is None:
        return seeds
    _QUALIFIER_CANDIDATES = [
        "extra large", "super fresh",
        "frozen pack", "specially selected", "locally grown",
        "standard grade", "imported variety",
    ]
    extended = set(seeds)
    for phrase in _QUALIFIER_CANDIDATES:
        doc = _nlp(phrase)
        for tok in doc:
            if tok.pos_ == "ADJ" and tok.lemma_.lower() not in _FOOD_VOCAB_SET:
                extended.add(tok.lemma_.lower())
    return extended


_NOISE_WORDS: set[str] = _auto_extend_noise_words(_NOISE_SEEDS)


# ---------------------------------------------------------------------------
# THRESHOLDS
# ---------------------------------------------------------------------------

MIN_CONFIDENCE: float = 0.50

_SYMBOL_ONLY_RE = re.compile(r"^[^a-zA-Z\u0900-\u097F]+$")
_TAX_BRACKET_RE = re.compile(
    r"^\d+\)\s*(cgst|sgst|igst|vat)\s*@"
    r"|\d\s*(cgst|sgst)\s*\d"
    r"|\d+(cgst|sgst)\d",
    re.IGNORECASE,
)

# FIX: removed bare "fresh" and "urgent" — they match inside product names.
# All app-UI patterns now require explicit surrounding context words or
# word-boundary anchors to avoid killing "Fresh Cream", "Fresh Chicken" etc.
_LEGAL_LINE_RE = re.compile(
    r"(avenue supermarts|cin\s*:|gstin\s*:|fssai|l\d{5}[a-z]+\d+"
    r"|tax invoice|bill\s*no|bill\s*dt|vou\.?\s*no|cashier"
    r"|items\s*:\s*\d|qty\s*:\s*\d|koparkhairane|navi mumbai"
    r"|plot\s*no|phone\s*:|gst\s*breakup|supermarts"
    r"|metro\s*cash|cash\s*&\s*carry|horeca|invoice\s*no"
    r"|table\s*\d|server\s*:|cover\s*\d|order\s*#|order\s*id"
    r"|carrefour\s*uae|al\s*ghurair|lulu\s*hypermarket"
    r"|apollo\s*pharmacy|monginis|nilgiri|theobroma"
    r"|eastern\s*spices|cafe\s*mocha|\bthank\s*you\s*for\b"
    r"|jiomart\.com|jio\s*mart|\border\s*summary\b|\btrack\s*order\b"
    r"|dmartindia|www\.jiomart|www\.dmart|barcode|\|\|\|\|"
    r"|\b\d{6,}[a-z]"
    r"|phone\d{7,}|tel\d{7,}"
    r"|d_mart|d\*mart"
    r"|mart\s+kha"
    r"|\border\s+total\b|\badd\s+ingredient\b|\binventory\s+health\b"
    r"|\bregister\s+item\b|\bitem\s+name\b|\bquantity\s*\/\s*unit\b"
    r"|\buse\s+soon\b"
    r")",
    re.IGNORECASE,
)

_TRAILING_UNIT_RE = re.compile(
    r"\s+(?:kg|kgs?|gm?|gram|grams?|liter|litre|ltr|ml|pcs?|pkt|packet|"
    r"pack|box|bottle|can|bunch|bag|roll|jar|unit|ea|each|no|nos|"
    r"piece|pieces?|dozen|tin|loaf|loaves)\s*$",
    re.IGNORECASE,
)

_EMBEDDED_DESCRIPTOR_RE = re.compile(
    r"\s*[-–—]\s*(?:\d+\s*(?:" + QUANTITY_UNITS + r")"
    r"|white|brown|whole|skim|full\s*fat|plain|sweet|spicy"
    r"|salted|unsalted|regular|classic|original"
    r"|\d+\s*pcs?|\d+\s*pack|\d+\s*bottle|\d+\s*box)\s*$",
    re.IGNORECASE,
)

# Compiled once at module level
_UI_BLEED_RE = re.compile(
    r"\s+(?:order\s+total|add\s+ingredient|inventory\s+health"
    r"|register\s+item|use\s+soon|expired"
    r"|expiry|status|action)\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# PUBLIC HELPER PREDICATES
# ---------------------------------------------------------------------------

def is_price(t: str) -> bool:
    return bool(_PRICE_RE.match(t.strip()))


def is_weight_qty(t: str) -> bool:
    """True for decimal weight tokens (1–3 d.p.); 2 d.p. treated as price."""
    t = t.strip()
    if not _WEIGHT_QTY_RE.match(t):
        return False
    parts = t.split(".")
    if len(parts) == 2 and len(parts[1]) == 2:
        return False
    return True


def is_qty(t: str) -> bool:
    t2 = re.sub(r"(\d+\.?\d*)([A-Za-z]+)", r"\1 \2", t.strip())
    return bool(_QTY_FULL_RE.match(t2))


def is_unit_only(t: str) -> bool:
    return bool(_UNIT_ONLY_RE.match(t.strip()))


def is_num(t: str) -> bool:
    return bool(_BARE_NUM_RE.match(t.strip()))


def is_skip_line(t: str) -> bool:
    t_clean = t.lower().strip()
    if not t_clean or len(t_clean) < 2:
        return True
    if is_price(t_clean):
        return True
    t_norm = re.sub(r"[_\-]", " ", t_clean)
    t_norm = re.sub(r"\s+", " ", t_norm).strip()
    if t_norm in _SKIP_EXACT or t_clean in _SKIP_EXACT:
        return True
    for prefix in _SKIP_PREFIXES:
        if t_clean.startswith(prefix) or t_norm.startswith(prefix):
            return True
    if _SYMBOL_ONLY_RE.match(t_clean):
        return True
    if _TAX_BRACKET_RE.search(t_clean):
        return True
    if _LEGAL_LINE_RE.search(t_clean):
        return True
    if _BILLING_SUFFIX_RE.search(t_clean) or _BILLING_SUFFIX_RE.search(t_norm):
        return True
    for substr in _BILLING_SUBSTRINGS:
        if substr in t_clean or substr in t_norm:
            return True
    words = t_clean.split()
    # FIX: only skip if ALL words are in the (now smaller) skip set
    if words and all(w in _SKIP_WORD_SET for w in words):
        return True
    return False


def is_header_row(texts: list[str]) -> bool:
    lower = {t.lower().strip() for t in texts}
    if lower and lower.issubset(_HEADER_WORDS):
        return True
    all_words = set(re.findall(r"\b\w+\b", " ".join(texts).lower()))
    return len(all_words & _HEADER_WORDS) >= 3


def normalize_quantity(text: str) -> str:
    return re.sub(r"(\d+\.?\d*)([A-Za-z]+)", r"\1 \2", text.strip())


# ---------------------------------------------------------------------------
# CORE ITEM NAME NORMALISATION
# ---------------------------------------------------------------------------

def clean_item_name(raw: str) -> str:
    item = raw.lower().strip()
    if not item:
        return ""

    # Stage 2 — OCR character swaps
    item = re.sub(r"@", "a", item)
    item = re.sub(r"\b0([a-z])", r"o\1", item)
    item = re.sub(r"\bI([A-Z])", r"l\1", item)

    # Stage 3 — strip embedded quantities
    item = re.sub(
        r"\b\d+\.?\d*\s*(?:" + QUANTITY_UNITS + r")\b",
        " ", item, flags=re.IGNORECASE,
    )
    item = re.sub(r"\bx\s*\d+|\d+\s*x\b", " ", item, flags=re.IGNORECASE)

    # Stage 3b — strip embedded descriptors BEFORE bare-digit strip
    item = _EMBEDDED_DESCRIPTOR_RE.sub("", item).strip()

    # Strip bare digits (including trailing ones)
    item = re.sub(r"\b\d+\.?\d*\b", " ", item)
    item = re.sub(r"\s+\d+\s*$", " ", item)

    # Stage 4 — strip punctuation / brand separators
    item = re.sub(r"\([^)]{1,5}\)", " ", item)
    item = re.sub(r"[^\w\s]", " ", item)
    item = re.sub(r"\s+", " ", item).strip()

    # Stage 5 — remove noise adjectives (auto-extended via spaCy when available)
    # FIX: only strip noise words when at least 1 content word will remain
    # (prevents hollowing out short names like "Pure Ghee" → "")
    words = item.split()
    filtered = [w for w in words if w not in _NOISE_WORDS and len(w) > 1]
    if filtered:  # FIX: guard — don't strip if it empties the name
        item = " ".join(filtered).strip()
    if not item:
        return ""

    # Stage 5.5 — transliteration (4/3/2/1-word sequences)
    item = _apply_transliteration(item)

    # Stage 5.8 — Early Vocab protection
    # If the item already matches perfectly (e.g. "bengal gram")
    # we stop here to prevent Stage 6 from killing items ending in "gram"/"kg".
    if item in _FOOD_VOCAB_SET:
        return item

    # Stage 6 — strip trailing unit words
    item = _TRAILING_UNIT_RE.sub("", item).strip()
    if not item:
        return ""

    # Stage 6b — strip trailing UI label bleed
    item = _UI_BLEED_RE.sub("", item).strip()
    if not item:
        return ""

    # Stage 8 — strip leading lone letter (brand initial)
    parts = item.split()
    if len(parts) >= 2 and re.match(r"^[a-z]$", parts[0]):
        item = " ".join(parts[1:])

    # Stage 9 — spaCy lemmatisation (automated when spaCy is installed)
    if SPACY_AVAILABLE:
        item = _lemmatize(item)

    # Stage 10 — singularisation (inflect or fallback)
    parts = item.split()
    if parts:
        parts[-1] = _to_singular(parts[-1])
    item = " ".join(parts).strip()
    if not item or len(item) < 2:
        return ""

    # Stage 11a — second transliteration pass (catches fruits/plurals)
    item2 = _apply_transliteration(item)
    if item2 != item:
        item = item2

    # Stage 11b — brand-prefix resolution
    brand_resolved = _resolve_brand_item(item)
    if brand_resolved:
        return brand_resolved
    
    # Stage 11c — absolute skip check BEFORE fuzzy match
    # Prevents billing words (Total, Pay) from shadowing valid food items
    if is_skip_line(item):
        return ""

    # Stage 12 — fuzzy food vocab matching (rapidfuzz or difflib)
    fuzzy_result = _fuzzy_match_food(item)
    if fuzzy_result:
        return fuzzy_result

    # Stage 13 — plausibility gate
    parts_check = item.split()
    has_digit   = bool(re.search(r"\d", item))
    all_short   = all(len(w) <= 2 for w in parts_check)
    digit_fused = bool(re.search(r"[a-z]\d|\d[a-z]{2,}", item))
    if has_digit or all_short or digit_fused:
        return ""

    return item


# ---------------------------------------------------------------------------
# SPATIAL ROW GROUPING
# ---------------------------------------------------------------------------

def _group_into_rows(paddle_page: list, y_tolerance: int = 15) -> list[dict]:
    """Group PaddleOCR word tokens into horizontal rows by y-centre proximity."""
    rows: list[dict] = []
    for word_info in paddle_page:
        box        = word_info[0]
        text, conf = word_info[1][0], word_info[1][1]
        if not text.strip():
            continue
        y_center = (box[0][1] + box[2][1]) / 2
        x_left   = box[0][0]
        placed   = False
        for row in rows:
            if abs(row["y"] - y_center) <= y_tolerance:
                row["tokens"].append((x_left, text, conf))
                placed = True
                break
        if not placed:
            rows.append({"y": y_center, "tokens": [(x_left, text, conf)]})
    rows.sort(key=lambda r: r["y"])
    for row in rows:
        row["tokens"].sort(key=lambda t: t[0])
    return rows


# ---------------------------------------------------------------------------
# QUANTITY DETECTION
# ---------------------------------------------------------------------------

def _find_qty_in_tokens(tokens: list[str]) -> tuple[Optional[str], set[int]]:
    n = len(tokens)

    for i in range(n - 1, -1, -1):
        if is_qty(tokens[i]) and not is_price(tokens[i]):
            return normalize_quantity(tokens[i]), {i}

    for i in range(n - 1):
        t1, t2 = tokens[i], tokens[i + 1]
        if is_num(t1) and is_unit_only(t2):
            return normalize_quantity(f"{t1} {t2}"), {i, i + 1}
        if is_unit_only(t1) and is_num(t2):
            return normalize_quantity(f"{t2} {t1}"), {i, i + 1}

    for i in range(n - 1, -1, -1):
        if is_weight_qty(tokens[i]):
            val = float(tokens[i])
            if val == 0.0:
                continue
            qty_str = ("%d g" % round(val * 1000)) if val < 1.0 else ("%.3f kg" % val)
            return qty_str, {i}

    for i in range(n - 1, -1, -1):
        if is_num(tokens[i]):
            val = float(tokens[i])
            if val <= 20:
                return f"{int(val)} pc", {i}
            if val <= 50 and "." not in tokens[i]:
                return f"{int(val)} pc", {i}
            break

    return None, set()


# ---------------------------------------------------------------------------
# ROW-LEVEL PARSER
# ---------------------------------------------------------------------------

def _parse_row(tokens_with_conf: list[tuple[str, float]]):
    filtered = [
        (t.strip(), c)
        for t, c in tokens_with_conf
        if c >= MIN_CONFIDENCE and t.strip()
    ]
    if not filtered:
        return None

    texts = [t for t, _ in filtered]

    # FIX: Handle rows containing ONLY numbers or weight (e.g. "0.500")
    if all(is_num(t) or is_price(t) for t in texts):
        weight_tok = next((t for t in texts if is_weight_qty(t)), None)
        if weight_tok:
            val = float(weight_tok)
            if val > 0.0:
                qty_str = ("%d g" % round(val * 1000)) if val < 1.0 else ("%.3f kg" % val)
                return {"weight_qty": qty_str}
        small_int = next(
            (t for t in texts if "." not in t and is_num(t) and 1 <= int(t) <= 20),
            None,
        )
        if small_int:
            return {"weight_qty": small_int + " pc"}
        return None

    # FIX: Handle horizontal bleed from receipt summary boxes (e.g. JioMart) or multi-column layouts
    # If "Wheat Flour" and "Total" are on the same Y-axis, we split and process both.
    _split_headers = {
        "subtotal", "sub total", "delivery charge", "delivery fee",
        "order total", "paid", "paid:", "total", "total:", "mrp"
    }
    
    # 1. Detect Split Points (Headers or Price-clusters)
    split_indices = []
    for i, t in enumerate(texts):
        t_clean = t.lower().strip()
        if t_clean in _split_headers or any(t_clean.startswith(b) for b in _split_headers):
            split_indices.append(i)
        elif i > 0 and (is_price(t) or is_qty(t)) and i + 1 < len(texts) and not is_qty(texts[i+1]) and not is_price(texts[i+1]):
             # [..., Price/Qty, ItemPart, ...] -> Split at ItemPart
             split_indices.append(i + 1)
    
    # 2. Extract multiple clusters
    segments = []
    last_idx = 0
    for idx in split_indices:
        if idx > last_idx:
            segments.append(texts[last_idx:idx])
        last_idx = idx
    segments.append(texts[last_idx:])
    
    # 3. Process each segment
    results = []
    for parts in segments:
        if not parts:
            continue
        
        # Clean parts — strip leading/trailing non-food tokens
        while parts and (is_price(parts[0]) or is_num(parts[0])):
            parts.pop(0)
        while parts and (is_price(parts[-1]) or is_num(parts[-1])):
            parts.pop(-1)
            
        if not parts:
            continue
            
        full_line = " ".join(parts)
        if is_header_row(parts) or is_skip_line(full_line):
            continue

        qty, qty_indices = _find_qty_in_tokens(parts)
        item_tokens = [t for i, t in enumerate(parts) if i not in qty_indices]
        item_raw    = " ".join(item_tokens).strip()

        if not item_raw or is_skip_line(item_raw):
            continue

        cleaned = clean_item_name(item_raw)
        if not cleaned or len(cleaned) < 2:
            continue

        results.append((cleaned, qty if qty else "1 pc"))

    return results if results else None


# ---------------------------------------------------------------------------
# TOKEN STREAM SEGMENTER
# ---------------------------------------------------------------------------

def _segment_token_stream(tokens: list[str]) -> list[tuple[str, str]]:
    results:    list[tuple[str, str]] = []
    item_parts: list[str]             = []
    i, n = 0, len(tokens)
    while i < n:
        t = tokens[i]
        if is_skip_line(t) or is_price(t):
            if item_parts:
                results.append((" ".join(item_parts), "1 pc"))
                item_parts = []
            i += 1
            continue
        if is_num(t) and i + 1 < n and is_unit_only(tokens[i + 1]):
            qty = f"{t} {tokens[i + 1]}"
            if item_parts:
                results.append((" ".join(item_parts), qty))
                item_parts = []
            i += 2
            continue
        if is_qty(t):
            if item_parts:
                results.append((" ".join(item_parts), t))
                item_parts = []
            i += 1
            continue
        if is_unit_only(t):
            qty = f"1 {t}"
            if item_parts:
                results.append((" ".join(item_parts), qty))
                item_parts = []
            i += 1
            continue
        item_parts.append(t)
        i += 1
    if item_parts:
        results.append((" ".join(item_parts), "1 pc"))
    return results


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def extract_grocery_items(
    ocr_output,
    conf_threshold: float = MIN_CONFIDENCE,
) -> list[list[str]]:
    """
    Extract grocery items and quantities from any OCR output format.

    Args:
        ocr_output     : PaddleOCR result (list of pages), flat list of strings,
                         or newline-separated plain string.
        conf_threshold : Minimum OCR confidence to accept a token (default 0.50).

    Returns:
        List of [item_name, quantity] pairs.
    """
    _conf_threshold = conf_threshold   # local copy — never mutates module constant

    def _is_token_stream(strings: list[str]) -> bool:
        if not strings:
            return False
        avg_len = sum(len(s) for s in strings) / len(strings)
        spaced  = sum(1 for s in strings if " " in s.strip())
        return avg_len < 12 and spaced < len(strings) * 0.25

    def _make_line_page(lines: list[str]) -> list:
        page = []
        for idx, line in enumerate(lines):
            line = str(line).strip()
            if line:
                y = idx * 20
                page.append(
                    [[[0, y], [200, y], [200, y + 15], [0, y + 15]], (line, 0.95)]
                )
        return page

    def _row_parse_with_threshold(tokens_with_conf: list[tuple[str, float]]):
        filtered = [
            (t.strip(), c)
            for t, c in tokens_with_conf
            if c >= _conf_threshold and t.strip()
        ]
        if not filtered:
            return None
        return _parse_row([(t, 1.0) for t, _ in filtered])

    # Input normalisation
    if isinstance(ocr_output, str):
        pages = [_make_line_page([ln for ln in ocr_output.splitlines() if ln.strip()])]
    elif isinstance(ocr_output, list) and ocr_output:
        first = ocr_output[0]
        if isinstance(first, str):
            strings = [s for s in ocr_output if str(s).strip()]
            if _is_token_stream(strings):
                results: list[list[str]] = []
                for raw_item, qty in _segment_token_stream(strings):
                    cleaned = clean_item_name(raw_item)
                    if cleaned and len(cleaned) >= 2 and not is_skip_line(cleaned):
                        results.append([cleaned, normalize_quantity(qty)])
                return results
            else:
                pages = [_make_line_page(strings)]
        elif isinstance(first, list):
            pages = ocr_output
        else:
            return []
    else:
        return []

    results:        list[list[str]] = []
    # FIX: pending_weight now also tracks which item row it belongs to.
    # It is only applied to the very next real item row, then cleared.
    pending_weight: Optional[str]   = None

    # FIX: stop_right_column tracks if we've reached a summary box (like "Total")
    # on the right side. After this, we ignore the secondary segment of rows.
    stop_right_column = False

    for page in pages:
        if not page:
            continue
        for row in _group_into_rows(page, y_tolerance=15):
            tokens_with_conf = [(text, conf) for _, text, conf in row["tokens"]]
            parsed = _row_parse_with_threshold(tokens_with_conf)
            if parsed is None:
                continue
            
            # Detect "Total" on the right to trigger stop_right_column
            texts_lower = [t.lower().strip() for t, _ in tokens_with_conf]
            if any(h in texts_lower for h in ["total", "subtotal", "order total", "grand total"]):
                # If "Total" is not at the very start, it's likely a right-side total
                if "total" in texts_lower and texts_lower.index("total") > 0:
                    stop_right_column = True
            
            if isinstance(parsed, dict) and "weight_qty" in parsed:
                pending_weight = parsed["weight_qty"]
                continue
            
            if isinstance(parsed, list):
                # Calculate row start X to detect if it's a "right column only" row
                row_x = min(t[0] for t in row["tokens"]) if row["tokens"] else 0
                
                # Heuristic: if row starts far to the right, it's a secondary column
                # Paddle typically returns coords in pixels. 300 is a safe threshold for multi-column.
                is_right_row = (row_x > 300) 
                
                if stop_right_column:
                    if is_right_row:
                        # Skip standalone items on the right in the summary area
                        continue
                    if len(parsed) > 1:
                        # Strip the right-side item from a multi-column row
                        parsed = parsed[:1]
                    
                for item, qty in parsed:
                    # Apply pending_weight only once and immediately clear it
                    if pending_weight is not None:
                        qty = pending_weight
                        pending_weight = None
                    results.append([item, qty])
            else:
                # Fallback for unexpected types (though should be list or dict)
                pass

    # 4. Filter & Deduplicate
    # JioMart receipts often repeat items in a summary box on the right.
    # We favor specific names (e.g. "Coriander Seeds") over generic ones (e.g. "Coriander")
    # if they appear in the same bill.
    final_results = []
    # Sort by length descending to process specific names first
    results.sort(key=lambda x: len(x[0]), reverse=True)
    
    seen_sets = []
    for item, qty in results:
        is_redundant = False
        item_words = set(item.split())
        
        for seen_set in seen_sets:
            # If "coriander" is a subset of "coriander seeds" (and they are same item root)
            if item_words.issubset(seen_set):
                # Only suppress if the generic one has a default quantity (likely summary)
                if qty == "1 pc" or qty == "1 unit":
                    is_redundant = True
                    break
        
        if not is_redundant:
            final_results.append([item, qty])
            seen_sets.append(item_words)

    return final_results


def extract_grocery_items_low_conf(ocr_output) -> list[list[str]]:
    """
    Same as extract_grocery_items() with conf_threshold=0.35.
    Recommended for wrinkled, damaged, or low-quality receipt scans.
    """
    return extract_grocery_items(ocr_output, conf_threshold=0.35)


def process_grocery_list(words: list) -> list[list[str]]:
    """Legacy entry point — delegates to extract_grocery_items()."""
    return extract_grocery_items(words)


# ---------------------------------------------------------------------------
# IMAGE PRE-PROCESSING  (requires opencv-python)
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str) -> str:
    """
    Apply OpenCV pre-processing (deskew + adaptive threshold + denoise + rescale)
    for difficult receipt scans.  Gracefully returns original path when cv2 is absent.
    """
    try:
        import cv2          # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        logger.warning("OpenCV not available — skipping image pre-processing.")
        return image_path

    import os

    img = cv2.imread(image_path)
    if img is None:
        return image_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deskew
    try:
        thresh_for_angle = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            blockSize=15, C=4,
        )
        coords = np.column_stack(np.where(thresh_for_angle > 0))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5:
                h, w   = gray.shape
                center = (w // 2, h // 2)
                M      = cv2.getRotationMatrix2D(center, angle, 1.0)
                gray   = cv2.warpAffine(
                    gray, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
    except Exception:
        pass

    thresh   = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31, C=11,
    )
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)
    h, w     = denoised.shape
    if max(h, w) < 1000:
        scale    = 1500 / max(h, w)
        denoised = cv2.resize(
            denoised, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    proc_path = os.path.splitext(image_path)[0] + "_processed.png"
    cv2.imwrite(proc_path, denoised)
    return proc_path


# ---------------------------------------------------------------------------
# RUNTIME VOCABULARY EXTENSION API
# ---------------------------------------------------------------------------

def register_food_term(term: str, aliases: "list[str] | None" = None) -> None:
    """
    Add a new canonical food term (and optional aliases) to the live vocabulary.
    The fuzzy-match LRU cache is cleared automatically so the new term is
    immediately discoverable on the next call.
    """
    t = term.lower().strip()
    if t and t not in _FOOD_VOCAB_SET:
        FOOD_VOCAB.append(t)
        _FOOD_VOCAB_SET.add(t)
        _fuzzy_match_food.cache_clear()
    for alias in aliases or []:
        a = alias.lower().strip()
        if a:
            TRANSLITERATION_MAP[a] = t
            (_MULTI_WORD_TRANS if " " in a else _SINGLE_WORD_TRANS)[a] = t


def register_brand(brand_name: str) -> None:
    """Register a new brand prefix for automatic stripping during normalisation."""
    _KNOWN_BRANDS.add(brand_name.lower().strip())


# ---------------------------------------------------------------------------
# DEBUG HELPER  (set GROCERY_OCR_DEBUG=1 to enable)
# ---------------------------------------------------------------------------

import os as _os

if _os.environ.get("GROCERY_OCR_DEBUG"):
    _orig_skip = is_skip_line

    def is_skip_line(t: str) -> bool:  # type: ignore[no-redef]
        result = _orig_skip(t)
        if result:
            logger.debug("SKIPPED LINE: %r", t)
        return result

    _orig_clean = clean_item_name

    def clean_item_name(raw: str) -> str:  # type: ignore[no-redef]
        result = _orig_clean(raw)
        if not result:
            logger.debug("DROPPED ITEM: %r → empty after cleaning", raw)
        else:
            logger.debug("CLEANED: %r → %r", raw, result)
        return result