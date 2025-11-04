"""Shared safer spell-check utilities for OCR pipelines.

This module prefers SymSpell when provided (context-aware lookup_compound).
When falling back to pyspellchecker it uses candidates() and lightweight
frequency + edit-distance heuristics instead of always calling correction(),
which can be over-aggressive.
"""
from typing import List, Optional
import re

def _edit_distance_limited(a: str, b: str, limit: int = 2) -> int:
    """Compute Levenshtein distance but bail out when > limit."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    m, n = len(a), len(b)
    # initialize row
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        ai = a[i - 1]
        best = curr[0]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + cost
            curr[j] = ins if ins < delete else delete
            if replace < curr[j]:
                curr[j] = replace
            if curr[j] < best:
                best = curr[j]
        if best > limit:
            return limit + 1
        prev = curr
    return prev[n]


_CONTRACTIONS = {
    "ive": "I've", "dont": "don't", "cant": "can't", "wont": "won't",
    "im": "I'm", "its": "it's", "thats": "that's"
}


def _preserve_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if len(src) > 1 and src[0].isupper() and src[1:].islower():
        return repl.capitalize()
    return repl


def load_whitelist(path: str) -> set:
    """Load whitelist file (one token per line). Returns set of lowercased tokens."""
    s = set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for ln in f:
                w = ln.strip()
                if not w:
                    continue
                s.add(w.lower())
    except Exception:
        return set()
    return s


def normalize_lines(lines: List[str], final_pass: bool = False) -> List[str]:
    """Apply lightweight OCR normalizations.

    - Replace underscores with spaces
    - Fix common digit/letter confusions (0->o,1->l,9->g) when between letters
    - Collapse multiple spaces, normalize spacing around punctuation
    - If final_pass True, fix capitalization at line starts
    """
    out = []
    for line in lines:
        if line is None:
            out.append(line)
            continue
        s = line.replace('_', ' ')

        # fix digit/letter confusions when embedded in words
        s = re.sub(r'(?<=[A-Za-z])0(?=[A-Za-z])', 'o', s)
        s = re.sub(r'(?<=[A-Za-z])1(?=[A-Za-z])', 'l', s)
        s = re.sub(r'(?<=[A-Za-z])9(?=[A-Za-z])', 'g', s)

        # remove weird repeated punctuation
        s = re.sub(r'[\s]+', ' ', s)
        s = re.sub(r'\s+([.,!?:;])', r'\1', s)
        s = re.sub(r"([.,!?:;])\s*([A-Za-z0-9\"'])", r"\1 \2", s)

        s = s.strip()
        if final_pass and s:
            # Capitalize first alpha char of the line
            first = re.search(r'[A-Za-z]', s)
            if first and first.start() == 0:
                s = s[0].upper() + s[1:]
        out.append(s)
    return out


def apply_spell_check_lines(lines: List[str], spell_checker=None, symspell=None, max_edit_distance: int = 2, whitelist: Optional[set] = None) -> List[str]:
    """Safer line-based spell correction.

    - If `symspell` is provided use its compound lookup for context-aware fixes.
    - Otherwise, if `spell_checker` (pyspellchecker) is available, use
      candidates() plus edit-distance and optional frequency heuristics.
    - Preserve punctuation and case. Be conservative: only apply fixes when
      edit distance <= max_edit_distance and candidate shares first or last
      character (helps avoid bizarre swaps).
    """
    if not lines:
        return lines

    # Normalize whitelist to lowercase set for comparisons
    if whitelist:
        wlset = set(w.lower() for w in whitelist if isinstance(w, str))
    else:
        wlset = set()

    # Prefer SymSpell when available
    try:
        if symspell is not None:
            out = []
            from symspellpy import Verbosity
            for line in lines:
                tokens = re.findall(r"[A-Za-z']+|\d+|[^A-Za-z\d']+", line)
                words = [t for t in tokens if re.fullmatch(r"[A-Za-z']+", t)]
                if words:
                    joined = ' '.join(w.lower() for w in words)
                    comp = symspell.lookup_compound(joined, max_edit_distance=max_edit_distance)
                    if comp:
                        corr = comp[0].term.split()
                        if len(corr) == len(words):
                            wi = 0
                            for i, t in enumerate(tokens):
                                if re.fullmatch(r"[A-Za-z']+", t):
                                    # preserve whitelisted tokens
                                    if t.lower() in wlset:
                                        wi += 1
                                        continue
                                    replacement = corr[wi]
                                    # handle common contractions
                                    if t.lower() in _CONTRACTIONS:
                                        replacement = _CONTRACTIONS[t.lower()]
                                    else:
                                        replacement = _preserve_case(t, replacement)
                                    tokens[i] = replacement
                                    wi += 1
                            out.append(''.join(tokens))
                            continue
                # Fallback per-token
                for i, t in enumerate(tokens):
                    if not re.fullmatch(r"[A-Za-z']+", t):
                        continue
                    lw = t.lower()
                    if lw in wlset:
                        continue
                    if lw in _CONTRACTIONS:
                        tokens[i] = _CONTRACTIONS[lw]
                        continue
                    suggs = symspell.lookup(lw, Verbosity.CLOSEST, max_edit_distance=max_edit_distance, include_unknown=True)
                    best = suggs[0] if suggs else None
                    if best and best.term != lw and best.distance <= max_edit_distance:
                        rep = best.term
                        tokens[i] = _preserve_case(t, rep)
                out.append(''.join(tokens))
            return out
    except Exception:
        # If symspell has problems, fall through to safer fallback
        pass

    # If no spell_checker available, return original
    if not spell_checker:
        return lines

    # Build a small token frequency map to avoid changing repeated forms
    freq = {}
    for ln in lines:
        for t in re.findall(r"[A-Za-z]+", ln or ""):
            freq[t.lower()] = freq.get(t.lower(), 0) + 1

    def choose_candidate(orig: str):
        core = re.sub(r'[^A-Za-z]', '', orig)
        if not core or len(core) < 2:
            return None
        low = core.lower()
        # If word appears in whitelist, preserve it
        if low in wlset:
            return None
        # If word appears often, assume it's intentional (e.g., a name/term)
        if freq.get(low, 0) > 1:
            return None
        # Contractions and short common words
        if low in _CONTRACTIONS:
            return _CONTRACTIONS[low]

        # Try candidates from pyspellchecker rather than correction()
        try:
            cands = spell_checker.candidates(low)
        except Exception:
            # fallback to correction() if candidates() not supported
            try:
                corr = spell_checker.correction(low)
                return corr if corr and corr != low else None
            except Exception:
                return None

        if not cands:
            return None

        best = None
        best_score = (max_edit_distance + 1, -1.0)  # (edit_dist, frequency)
        for c in cands:
            if c == low:
                continue
            d = _edit_distance_limited(low, c, limit=max_edit_distance)
            if d > max_edit_distance:
                continue
            # small heuristic: demand same first or last char to avoid bad swaps
            if c[0] != low[0] and c[-1] != low[-1]:
                continue
            freq_score = 0.0
            try:
                if hasattr(spell_checker, 'word_frequency'):
                    freq_score = spell_checker.word_frequency.frequency(c)
            except Exception:
                freq_score = 0.0
            # Lower edit distance is better; higher freq is better
            score = (d, -freq_score)
            if score < best_score:
                best_score = score
                best = c
        return best

    out_lines = []
    for line in lines:
        tokens = re.findall(r"[A-Za-z']+|\d+|[^A-Za-z\d']+", line)
        for i, t in enumerate(tokens):
            if not re.fullmatch(r"[A-Za-z']+", t):
                continue
            # skip ALL CAPS abbreviations and mixed alphanum
            if t.isupper() and len(t) <= 5:
                continue
            # skip whitelist tokens
            if t.lower() in wlset:
                continue
            cand = choose_candidate(t)
            if cand:
                tokens[i] = _preserve_case(t, cand)
        out_lines.append(''.join(tokens))

    return out_lines
