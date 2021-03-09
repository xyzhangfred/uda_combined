aug = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/data/mctaco/mctaco_out_full.txt'
orig = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/data/mctaco/mctaco_unlabeled_orig.tsv'
combined = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/data/mctaco/mctaco_unlabeled.txt'

with open (aug, 'r' ) as f:
    aug_lines = f.readlines()

with open (orig, 'r' ) as f:
    orig_lines_raw = f.readlines()

orig_lines = ['\t'.join(l.strip().split('\t')[:3]) for l in orig_lines_raw  ]

with open (combined, 'w' ) as f:
    for orig_line, aug_line in zip(orig_lines, aug_lines):
        f.write(orig_line.strip() + '\t' + aug_line)
