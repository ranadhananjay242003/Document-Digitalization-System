"""Auto-extract likely names/terms from a pasted OCR sample, append to
data/whitelist.txt, and run sample_correct to show improved corrections.

This is conservative: it selects capitalized tokens and multi-word
capitalized sequences, filters common words, and appends unique new
entries to the whitelist.
"""
import re
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from spell_utils import load_whitelist

SAMPLE = '''Serie
Simple writing
T Gitla can bo seen as the maun
Iiterary Support for the great religious
Civilization of India, the oldest Survivin
Culture in he world The present
Lranslation and commentary is another
Manifestation of the permanent living
Importance of Geeta Swami Bhaktivedar
Brinjs totbe West Salutary reminder
That our highly activistic and One Sided
Culture is faced with crisis that may
Erd in Self-destruction because it lack
The inner Aeftb ofan 9uthentic meta
Physical Cosciousness without such
Deptl, Our moral and Political protesthion
Are Just So much verblage
Thomas Merton) Rajendra Bittins'
Author funt Author
'''

COMMON = set([w.lower() for w in (
    "The That Our This A An In On At For Of And Or But Is Was Be Are To It",
    "Simple Serie".split()
)] )

def extract_candidates(text):
    # Find multi-word capitalized sequences
    seqs = re.findall(r'(?:\b[A-Z][a-z]{2,}\b(?:\s+|\b)){1,4}', text)
    candidates = set()
    for s in seqs:
        cleaned = s.strip().strip(')\',"')
        if not cleaned:
            continue
        parts = cleaned.split()
        # skip if it's just a common word
        if all(p.lower() in COMMON for p in parts):
            continue
        candidates.add(cleaned)

    # Also single capitalized tokens (length >=4) that are not common
    singles = re.findall(r'\b([A-Z][a-z]{3,})\b', text)
    for s in singles:
        if s.lower() in COMMON:
            continue
        candidates.add(s)

    return sorted(candidates, key=lambda x: (-len(x), x))

def append_to_whitelist(path, items):
    existing = load_whitelist(path)
    to_add = [it for it in items if it.lower() not in existing]
    if not to_add:
        print('No new whitelist candidates to add.')
        return []
    with open(path, 'a', encoding='utf-8') as f:
        for it in to_add:
            f.write(it + '\n')
    print(f'Appended {len(to_add)} items to {path}:')
    for it in to_add:
        print('  -', it)
    return to_add

def main():
    wl_path = os.path.join(ROOT, 'data', 'whitelist.txt')
    if not os.path.exists(os.path.dirname(wl_path)):
        os.makedirs(os.path.dirname(wl_path), exist_ok=True)
    existing = load_whitelist(wl_path)
    print(f'Existing whitelist entries: {len(existing)}')
    cand = extract_candidates(SAMPLE)
    print('Candidates to consider:')
    for c in cand:
        print('  -', c)

    added = append_to_whitelist(wl_path, cand)

    # Re-run sample_correct to show results
    print('\nRe-running sample_correct.py to show corrected output...')
    os.system(f'python "{os.path.join(ROOT, "src", "sample_correct.py") }"')

if __name__ == '__main__':
    main()
