"""Generate the experiment's made-up words with UniPseudo and save its reply.

Usage, from the project root:
    python tools/make_nonce_words.py  [output.json]

UniPseudo (https://www.lexique.org/django/unipseudo/) builds pseudowords by chaining
letter bigrams or trigrams taken from real words of the same length. This script does
what its web page does: it asks for the English source words of the chosen length (they
come from the English Lexicon Project), then asks UniPseudo to generate pseudowords from
them with the page's default algorithm and constraints for English.

A pseudoword is then dropped if

* it is a real word according to the system word list /usr/share/dict/words, as it
  stands or once a final "s", "es", "d" or "ed" is removed (so "rales" goes because
  "rale" is listed), or
* it contains the first three letters of a color or shape used in the experiment (for
  example "blu" or "squ"), which could hint at a rule.

The first KEEP pseudowords that remain, in the order UniPseudo returned them, are the
made-up words. The script prints them as a list to paste into config.yaml
(puzzles.nonce_words).

UniPseudo is random and takes no seed, so running this again gives different words. The
reply that the experiment's words came from is kept in reference/unipseudo_words.json.
Do not edit that file by hand.
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hh.config import load_config  # noqa: E402

SITE = "https://www.lexique.org"
PAGE = SITE + "/django/unipseudo/"
LANGUAGE = "English"
LENGTH = 5          # letters per pseudoword
REQUESTED = 100     # pseudowords asked for
KEEP = 40           # pseudowords kept after screening
WORD_LIST = Path("/usr/share/dict/words")

output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reference" / "unipseudo_words.json"
features = load_config(ROOT / "config.yaml").features

# --- ask UniPseudo, in the same three steps as its web page ---------------------------
session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (research script)"
page = session.get(PAGE, timeout=60).text
token = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', page).group(1)
settings = json.loads(re.search(r'<script id="unipseudo-config" type="application/json">(.*?)</script>',
                                page, re.S).group(1))
urls = {name: SITE + path for name, path in settings["urls"].items()}

language = session.get(urls["language"], params={"language": LANGUAGE}, timeout=60).json()
source_words = session.get(urls["words"], params={"language": LANGUAGE, "length": LENGTH, "gram_class": ""},
                           timeout=60).json()["words"]
algorithm = settings["algorithms"][str(LENGTH)]["default"]
filters = {name: "" if value is None else str(value) for name, value in language["presets"].items()}
reply = session.post(
    urls["generate"],
    json={"n": str(REQUESTED), "length": str(LENGTH), "algorithm": algorithm, "language": LANGUAGE,
          "words": "\n".join(source_words), "filters": filters},
    headers={"X-CSRFToken": token, "Referer": PAGE},
    timeout=180,
)
reply.raise_for_status()
reply = reply.json()
pseudowords = [row["p"] for row in reply["rows"]]

# --- screen ---------------------------------------------------------------------------
real_words = {line.strip().lower() for line in WORD_LIST.read_text().splitlines()} if WORD_LIST.exists() else None


def real_form(word):
    """The listed word that ``word`` is or is an inflection of, or None."""
    stems = [word] + [word[:-len(ending)] for ending in ("s", "es", "d", "ed") if word.endswith(ending)]
    return next((stem for stem in stems if stem in real_words), None)


hints = sorted({str(value)[:3].lower() for values in features.values() for value in values})
kept, dropped = [], {}
for word in pseudowords:
    if word in kept or word in dropped:
        dropped.setdefault(word, "repeated")
    elif not re.fullmatch(r"[a-z]+", word):
        dropped[word] = "not made of the letters a to z"
    elif real_words is not None and real_form(word):
        dropped[word] = f"{real_form(word)!r} is in {WORD_LIST}"
    elif any(hint in word for hint in hints):
        dropped[word] = "contains " + ", ".join(repr(hint) for hint in hints if hint in word)
    elif len(kept) < KEEP:
        kept.append(word)
if len(kept) < KEEP:
    sys.exit(f"only {len(kept)} of {len(pseudowords)} pseudowords passed the screen; raise REQUESTED")

record = {
    "source": {
        "tool": "UniPseudo",
        "url": PAGE,
        "date": date.today().isoformat(),
        "language": LANGUAGE,
        "length": LENGTH,
        "algorithm": algorithm,
        "constraints": filters,
        "requested": REQUESTED,
        "word_database": language["database"],
        "n_source_words": len(source_words),
        "generated_by": "tools/make_nonce_words.py",
    },
    "screen": {
        "real_word_list": str(WORD_LIST) if real_words is not None else None,
        "hints_excluded": hints,
        "kept_first": KEEP,
    },
    "reply": reply,
    "dropped": dropped,
    "kept": kept,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(record, indent=1) + "\n")
print(f"wrote {output}: {len(pseudowords)} pseudowords from UniPseudo, {len(dropped)} dropped, {len(kept)} kept")
print("dropped:", dropped)
print("\nnonce_words: [" + ", ".join(kept) + "]")
