import json
import pandas as pd

raw_orig = json.load(open('raw_data/ropes_contrast_set_original_032820.json'))['data']
orig_raw_text = [r['paragraphs'] for r in raw_orig]

breakpoint()