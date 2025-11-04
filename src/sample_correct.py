"""Apply spell_utils correction to a provided OCR sample and print before/after.

Run from project root:
    python src\\sample_correct.py
"""
import sys, os
here = os.path.dirname(os.path.abspath(__file__))
if here not in sys.path:
    sys.path.insert(0, here)

from spell_utils import apply_spell_check_lines, load_whitelist

SAMPLE = [
    "T Gitla can bo seen as the maun",
    "Iiterary Support for the great religious",
    "Civilization of India, the oldest Survivin",
    "Culture in he world The present",
    "Lranslation and commentary is another",
    "Manifestation of the permanent living",
    "Importance of Geeta Swami Bhaktivedar",
    "Brinjs totbe West Salutary reminder",
    "That our highly activistic and One Sided",
    "Culture is faced with crisis that may",
    "Erd in Self-destruction because it lack",
    "The inner Aeftb ofan 9uthentic meta",
    "Physical Cosciousness without such",
    "Deptl, Our moral and Political protesthion",
    "Are Just So much verblage",
    "Thomas Merton) Rajendra Bittins'",
    "Author funt Author",
]

def try_load_components():
    sym = None
    spell = None
    try:
        from symspellpy import SymSpell
        cand = os.path.join(os.path.dirname(here), '..', 'data', 'frequency_dictionary_en_82_765.txt')
        if os.path.exists(cand):
            s = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            s.load_dictionary(cand, 0, 1)
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
    # lightweight mock tuned for this sample
    from spell_utils import _edit_distance_limited
    vocab = {
        'geeta': 200, 'swami': 150, 'bhaktivedan': 20, 'rajendra': 50, 'thomas': 300, 'merton': 300,
        'the':1000,'civilization':500,'india':600,'culture':700,'translation':400,'manifestation':200,
        'importance':300,'support':250,'author':200,'west':400,'salutary':50,'authentic':150,'depth':200,
        'physical':300,'consciousness':300,'political':200,'protection':100,'verbiage':80,'self-destruction':60
    }
    class MockFreq:
        def frequency(self,w):
            return float(vocab.get(w,1))
    class MockSpell:
        def __init__(self):
            self.word_frequency = MockFreq()
        def candidates(self, word):
            word = word.lower()
            c=set()
            for w in vocab:
                if _edit_distance_limited(word, w.lower(), limit=2) <=2:
                    c.add(w.lower())
            return c
        def correction(self, word):
            cands=self.candidates(word)
            if not cands:
                return word
            return max(cands, key=lambda x: vocab.get(x,0))
    return MockSpell()

def main():
    sym, spell = try_load_components()
    if not sym and not spell:
        spell = make_mock_spellchecker()
        used_mock = True
    else:
        used_mock = False
    # Load whitelist from repository data directory (if present)
    repo_root = os.path.dirname(here)
    # data folder is at repo root/data
    wl_path = os.path.join(repo_root, 'data', 'whitelist.txt')
    whitelist = load_whitelist(wl_path)

    print('--- ORIGINAL ---')
    print('\n'.join(SAMPLE))
    print('\n--- CORRECTED ---')
    out = apply_spell_check_lines(SAMPLE, spell_checker=spell, symspell=sym, whitelist=whitelist)
    print('\n'.join(out))
    print('\nused_mock_spellchecker=', used_mock)

if __name__ == "__main__":
    main()
