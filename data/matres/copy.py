aug = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/data/matres/matres_out_full.txt'
orig = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/data/matres/matres_unlabeled_orig.txt'
combined = '/home/xiongyi/dataxyz/repos/Mine/uda_combined/data/matres/matres_unlabeled.txt'

with open (aug, 'r' ) as f:
    aug_lines = f.readlines()

with open (orig, 'r' ) as f:
    orig_lines = f.readlines()

with open (combined, 'w' ) as f:
    for orig_line, aug_line in zip(orig_lines, aug_lines):
        f.write(orig_line.strip() + '\t' + aug_line)
