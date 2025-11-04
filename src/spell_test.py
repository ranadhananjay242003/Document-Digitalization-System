"""Quick demo for the new spell checking helper.

Run from project root:
    python src\\spell_test.py

The script will try to use SymSpell (if installed and dictionaries present)
or fall back to pyspellchecker. It prints before/after lines for sample OCR
fragments so you can evaluate improvements without running full OCR pipelines.
"""
import sys
import os

# Ensure src is importable when running from repository root
here = os.path.dirname(os.path.abspath(__file__))
if here not in sys.path:
    sys.path.insert(0, here)

from spell_utils import apply_spell_check_lines

def try_load_components():
    sym = None
    spell = None
    try:
        from symspellpy import SymSpell
        # Try to locate dictionary in repo data folder
        dict_path = None
        cand = os.path.join(os.path.dirname(here), 'data', 'frequency_dictionary_en_82_765.txt')
        if os.path.exists(cand):
            dict_path = cand
        if dict_path:
            s = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            s.load_dictionary(dict_path, 0, 1)
            sym = s
    except Exception:
        sym = None

    try:
        from spellchecker import SpellChecker
        spell = SpellChecker()
    except Exception:
        spell = None

    return sym, spell


def make_mock_spellchecker():
    """Create a tiny mock SpellChecker-like object for demo when deps missing.

    It implements candidates(word) and provides a minimal word_frequency
    interface used by the helper; kept intentionally small to demonstrate
    expected corrections without external packages.
    """
    from spell_utils import _edit_distance_limited

    vocab = {
        'the': 1000, 'quick': 800, 'brown': 700, 'fox': 600, 'jumps': 500, 'over': 900, 'lazy': 400, 'dog': 700,
        'you': 900, 'never': 800, 'would': 700, 'receive': 600, 'occurred': 300, 'civilization': 200,
        'don\'t': 900, "I'm": 800, 'rajendra': 50, 'brings': 120, 'so': 900, 'much': 850,
        'manifestation': 100, 'their': 950, 'support': 300
    }

    class MockFreq:
        def frequency(self, w):
            return float(vocab.get(w, 1))

    class MockSpell:
        def __init__(self):
            self.word_frequency = MockFreq()

        def candidates(self, word):
            word = word.lower()
            c = set()
            for w in vocab.keys():
                if _edit_distance_limited(word, w.lower(), limit=2) <= 2:
                    c.add(w.lower())
            return c

        def correction(self, word):
            # naive: return highest freq candidate within edit distance 2
            cands = self.candidates(word)
            if not cands:
                return word
            return max(cands, key=lambda x: vocab.get(x, 0))

    return MockSpell()


SAMPLES = [
    "teh quick brwn fox jumps ovr teh lazy dog",
    "Yevnever was able to woule recieve teh package",
    "recieve occured recieveing",
    "Cvi Lution led to great changes",
    "Dont worry, im fine",
    "DONT CHANGE NASA or ALL CAPS",
    "rajendm went to the market",
    "The car was fast and the brinas cost was low",
    "somuch of the text is joinedlike this",
    "thier mainifestaration was clear"
]

def main():
    sym, spell = try_load_components()
    # If neither real component is available, use the lightweight mock for demo
    used_mock = False
    if not sym and not spell:
        spell = make_mock_spellchecker()
        used_mock = True

    print(f"SymSpell available: {bool(sym)}; pyspellchecker available: {not used_mock and bool(spell)}; using_mock: {used_mock}")

    corrected = apply_spell_check_lines(SAMPLES, spell_checker=spell, symspell=sym)

    print('\n--- BEFORE ---')
    for s in SAMPLES:
        print(s)

    print('\n--- AFTER ---')
    for s in corrected:
        print(s)

if __name__ == '__main__':
    main()
