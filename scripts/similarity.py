"""Free "too close" checks for one swap (original -> replacement).

Daniel, 2026-10-21: the generated pairs are mostly good; the flaw is substitutes that keep too
much of the original (@hector_rv_1987 -> @hugo_lm_1984 keeps its template). Instead of a paid
LLM judge, compare each pair structurally and by spelling, flag the subset, and fix or drop it.

    checker = Checker(all_swaps)            # learns which shared tokens are generic
    checker.too_close(original, replacement) -> list of reasons ([] = fine)

Calibration notes (batch-r1, 2026-10-21) are in docs/spec.md, "Too-close checks". The first,
naive version flagged 3,043 of 4,930 pairs, almost all false alarms on shared generic words
("University of", "mailbox.ph", "Dr."), dates that keep their year (the generator is told to),
and the fictional 555 phone block.
"""
import re
from collections import Counter

from rapidfuzz.distance import JaroWinkler, Levenshtein

YEAR = re.compile(r"(?<!\d)(19\d\d|20\d\d)(?!\d)")
MONTHS = re.compile(r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|janv|févr|mär|mai|juni|juli|okt|dez|"
                    r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|sept|octubre|noviembre|diciembre|"
                    r"gennaio|giugno|luglio|ottobre|julie|年|月", re.I)
GENERIC_MIN = 4  # a token kept unchanged in >= this many distinct swaps is generic, not identifying
# First words of place / organisation names and honorifics, which legitimately survive a swap
# (Kiryat Motzkin -> Kiryat Gat). Not learned from the data: first NAMES are kept so often by the
# generator that a frequency rule learned "Tobias" as generic and hid the flaw it was meant to find.
PREFIXES = set("""klinikum kiryat derech ramat givat beit kfar neve tel ein bad rue via calle avenida rua strada
saint st san santa santo sankt mount mt lake port fort new north south east west upper lower old great little
hotel cafe café clinic hospital university college school dr mr mrs ms herr frau doctor professor prof uncle aunt
auntie kuya tita tito nonna nonno saba savta rabbi father sister brother mama papa abuela abuelo tía tío
grandad grandpa grandma granny nana universitätsklinikum""".split())
DATE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")


def tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[\s\W_]+", s.lower()) if t]


def shape(s: str) -> str:
    """Letters -> a, digits -> 9, runs collapsed; separators and @ . _ - kept."""
    return re.sub(r"\d+", "9", re.sub(r"[^\W\d_]+", "a", s))


def mixed(s: str) -> bool:
    """Handle / username / email / ID-like: letters plus digits or separators, no spaces."""
    return " " not in s.strip() and bool(re.search(r"[^\W\d_]", s)) and bool(re.search(r"[\d_@]|\.\w", s))


class Checker:
    def __init__(self, swaps: list[tuple[str, str]]):
        kept = Counter()
        for o, n in set(swaps):
            for t in set(tokens(o)) & set(tokens(n)):
                kept[t] += 1
        self.generic = {t for t, c in kept.items() if c >= GENERIC_MIN or len(t) <= 2 or t.isdigit()}
        # Words that the generator swapped ON THEIR OWN somewhere in the data: mostly first names
        # and surnames ("Tobias", "Vazquez"). A kept word only counts as a leak if it is one of
        # these; organisation prefixes ("Banco", "Grupo", "Agios") never appear alone.
        self.names = {o.lower() for o, _ in swaps if o.isalpha() and len(o) >= 3}

    def too_close(self, o: str, n: str) -> list[str]:
        reasons = []
        to, tn = tokens(o), tokens(n)
        shared = (set(to) & set(tn)) - self.generic

        # A name part survives the swap. Two narrow cases, because "kept any rare word" fired on
        # ordinary nouns (Highway, Markt, Auto Lease, month names):
        #  - person-like names (2-3 capitalised words) keeping their FIRST word: Tobias Vogt -> Tobias Wendt
        #  - handles / emails keeping a local-part chunk: yara.lindqvist -> yara.lindgren, gabriel.santos88 -> gustavo.santos88
        words = o.split()
        if 2 <= len(words) <= 3 and all(w[:1].isupper() and w.isalpha() for w in words) \
                and words[0].lower() in set(to) & set(tn) and words[0].lower() not in PREFIXES \
                and words[0].lower() in self.names:
            reasons.append("kept:" + words[0])
        elif mixed(o) and mixed(n):
            local_o, local_n = (tokens(o.split("@")[0]) if "@" in o.lstrip("@") else to), \
                               (tokens(n.split("@")[0]) if "@" in n.lstrip("@") else tn)
            kept = [t for t in set(local_o) & set(local_n)
                    if t not in self.generic and len(t) >= 3 and re.sub(r"\d+$", "", t) in self.names]
            if kept and set(local_o) != set(local_n):
                reasons.append("kept:" + "|".join(sorted(kept)))

        # Spelling of what actually changed: Carlo -> Carlos, Hoffmann -> Hofmann, Mickey -> Micky.
        do = " ".join(t for t in to if t not in tn and not t.isdigit())
        dn = " ".join(t for t in tn if t not in to and not t.isdigit())
        if len(do) >= 4 and dn and not MONTHS.search(do):
            if JaroWinkler.normalized_similarity(do, dn) >= 0.92 or Levenshtein.normalized_similarity(do, dn) >= 0.8:
                reasons.append("spelling")

        # Template + a birth-year-like number shifted by 1-3: @hector_rv_1987 -> @hugo_lm_1984.
        # The same year is left alone (event years, case numbers keep their year by design).
        if mixed(o) and mixed(n) and shape(o) == shape(n):
            oy, ny = YEAR.findall(o), YEAR.findall(n)
            if oy and ny and 0 < abs(int(oy[0]) - int(ny[0])) <= 3:
                reasons.append("template+year")

        # Numbers (phones, IDs, accounts) only: strings that are mostly digits, not dates, times,
        # addresses or money. Fictional blocks (555, UK 07700 900 / 01632 960, Ofcom drama ranges)
        # are stripped first; then flag when 3 of the last 4 digits match.
        alnum = re.sub(r"[\W_]", "", o)
        numeric = alnum and sum(c.isdigit() for c in alnum) / len(alnum) >= 0.6
        if numeric and not (YEAR.search(o) or DATE.search(o) or MONTHS.search(o) or ":" in o or re.search(r"\d,\d{3}", o)):
            def core(x):
                x = re.sub(r"\D", "", x)
                for block in ("07700900", "01632960", "555"):
                    x = x.replace(block, "")
                return x
            od, nd = core(o), core(n)
            # >= 6 digits: shorter strings are postcodes and small counts, close by design.
            if len(od) >= 6 and len(nd) >= 6 and sum(a == b for a, b in zip(od[-4:], nd[-4:])) >= 3:
                reasons.append("digits")
        return reasons
