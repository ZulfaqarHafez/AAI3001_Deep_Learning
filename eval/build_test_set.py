"""Rebuild the PPE test split with images.

The images of yolo-v8-new-model/test were never kept in git (*.jpg is ignored)
except in commit 238e70b5e6 (2025-11-25, "added new dataset"), an earlier
Roboflow export of the same project with the same 253 test source images.
The labels at HEAD come from the later v5 export (2025-11-28) that the models
were trained on, so this script pairs each HEAD label file with the historic
image of the same source id (the part of the filename before ".rf.").
YOLO labels are normalised, so the v5 512x512 stretch resize does not change
them. Two source ids have two files each; those are paired by label overlap.

Usage:
  git fetch --depth 1 origin 238e70b5e6dc4f61393f1e7fb7181a1d394a4bea
  mkdir -p /tmp/old && git archive 238e70b5e6dc4f61393f1e7fb7181a1d394a4bea \
      yolo-v8-new-model/test | tar -x -C /tmp/old
  python eval/build_test_set.py --old /tmp/old/yolo-v8-new-model/test --out /tmp/ppe_test
"""
import argparse, collections, glob, os, shutil

NAMES = ['Gloves', 'Helmet', 'No-Helmet', 'No-gloves', 'No-vest', 'Vest']


def sid(path):
    return os.path.basename(path).split('.rf.')[0]


def boxes(path):
    return [tuple(float(x) for x in l.split()) for l in open(path) if l.strip()]


def overlap(a, b):
    # count boxes in a that have a same-class box in b within 0.01 on all coords
    return sum(any(x[0] == y[0] and all(abs(x[i] - y[i]) < 0.01 for i in range(1, 5)) for y in b) for x in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', required=True, help='historic test dir with images/ and labels/')
    ap.add_argument('--new-labels', default='yolo-v8-new-model/test/labels')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    old = collections.defaultdict(list)
    for img in glob.glob(os.path.join(a.old, 'images', '*.jpg')):
        lab = os.path.join(a.old, 'labels', os.path.basename(img)[:-4] + '.txt')
        old[sid(img)].append((img, boxes(lab)))
    os.makedirs(os.path.join(a.out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'labels'), exist_ok=True)
    used, n = set(), 0
    for lab in sorted(glob.glob(os.path.join(a.new_labels, '*.txt'))):
        cands = [c for c in old[sid(lab)] if c[0] not in used]
        assert cands, f'no historic image for {lab}'
        nb = boxes(lab)
        img = max(cands, key=lambda c: overlap(nb, c[1]))[0]
        used.add(img)
        stem = os.path.basename(lab)[:-4]
        shutil.copy(img, os.path.join(a.out, 'images', stem + '.jpg'))
        shutil.copy(lab, os.path.join(a.out, 'labels', stem + '.txt'))
        n += 1
    with open(os.path.join(a.out, 'data.yaml'), 'w') as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images\nval: images\ntest: images\nnc: 6\nnames: {NAMES}\n")
    print(f'paired {n} label files with images -> {a.out}')


if __name__ == '__main__':
    main()
