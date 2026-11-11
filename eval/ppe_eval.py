"""Evaluate the PPE YOLOv8 checkpoints on the test split (CPU) and list failures.

Usage:
  python eval/ppe_eval.py --data /tmp/ppe_test/data.yaml --out eval/results \
      --weights deployed=path/best_deep.pt readme_production=path/best_100Epoch.pt \
                repo=model_eval/ppe-v8s-final/weights/bestV8_50Epoch.pt \
      --analyse deployed --exclude-stems eval/results/train_test_stem_overlap.txt

Outputs (in --out):
  metrics.json            per-model, per-class P, R, AP50, AP50-95 + confusion matrix
  failures_<tag>.json     every FN / FP / class confusion at the app threshold
  val_<tag>/              Ultralytics plots (confusion matrix, PR curves)
"""
import argparse, glob, json, os, random, time

import numpy as np
from PIL import Image
from ultralytics import YOLO

NAMES = ['Gloves', 'Helmet', 'No-Helmet', 'No-gloves', 'No-vest', 'Vest']
APP_CONF = 0.5          # backend/backend_api.py confidence_threshold
IOU_MATCH = 0.5
LOW_CONF = 0.1         # FNs matched by a same-class box between LOW_CONF and APP_CONF


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_gt(label_path, W, H):
    out = []
    for l in open(label_path):
        if l.strip():
            c, x, y, w, h = map(float, l.split())
            out.append((int(c), [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]))
    return out


def val_model(tag, weights, data, out, split='test', conf=0.001):
    model = YOLO(weights)
    t0 = time.time()
    m = model.val(data=data, split=split, imgsz=640, batch=16, device='cpu', conf=conf, iou=0.7,
                  plots=True, project=os.path.abspath(out), name=f'val_{tag}', exist_ok=True, verbose=False)
    dt = time.time() - t0
    b = m.box
    per_class = {}
    for i, c in enumerate(m.ap_class_index):
        per_class[NAMES[c]] = {'precision': float(b.p[i]), 'recall': float(b.r[i]),
                               'AP50': float(b.ap50[i]), 'AP50_95': float(b.ap[i])}
    cm = m.confusion_matrix.matrix.astype(int).tolist()
    return {
        'weights': weights,
        'all': {'precision': float(b.mp), 'recall': float(b.mr), 'mAP50': float(b.map50), 'mAP50_95': float(b.map)},
        'per_class': per_class,
        'confusion_matrix': {'note': f'Ultralytics layout: rows = predicted, cols = true; last index = background; built from predictions with conf >= {conf}',
                             'labels': NAMES + ['background'], 'matrix': cm},
        'val_seconds': round(dt, 1),
    }


def brightness(img, box=None):
    g = img.convert('L')
    if box is not None:
        g = g.crop(tuple(int(v) for v in box))
    a = np.asarray(g, dtype=np.float32)
    return float(a.mean()) if a.size else float('nan')


def analyse(tag, weights, img_dir, lab_dir, out):
    """Match predictions at the app threshold to ground truth and tag each error."""
    model = YOLO(weights)
    fails = []
    for img_path in sorted(glob.glob(os.path.join(img_dir, '*.jpg'))):
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        gt = read_gt(os.path.join(lab_dir, os.path.basename(img_path)[:-4] + '.txt'), W, H)
        r = model.predict(img_path, imgsz=640, conf=LOW_CONF, device='cpu', verbose=False)[0]
        allp = [(int(c), bb.tolist(), float(s)) for c, bb, s in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf)]
        preds = [p for p in allp if p[2] >= APP_CONF]
        low = [p for p in allp if p[2] < APP_CONF]
        img_b = brightness(img)
        gt_used, pr_used = set(), set()
        # greedy same-class matching by IoU
        pairs = sorted(((iou(g[1], p[1]), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(preds)
                        if g[0] == p[0]), reverse=True)
        for v, gi, pi in pairs:
            if v >= IOU_MATCH and gi not in gt_used and pi not in pr_used:
                gt_used.add(gi); pr_used.add(pi)
        for gi, (c, box) in enumerate(gt):
            if gi in gt_used:
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            rel = area / (W * H)
            best_other = max(((iou(box, p[1]), p) for pi, p in enumerate(preds) if pi not in pr_used), default=(0, None))
            same_cls_loc = max((iou(box, p[1]) for p in preds if p[0] == c), default=0)
            overl = max((iou(box, g2[1]) for gj, g2 in enumerate(gt) if gj != gi), default=0)
            tags = []
            lowm = max((p[2] for p in low if p[0] == c and iou(box, p[1]) >= IOU_MATCH), default=None)
            if lowm is not None:
                tags.append(f'detected_below_app_threshold(conf {lowm:.2f})')
            if best_other[0] >= IOU_MATCH and best_other[1][0] != c:
                tags.append(f'class_confusion(pred {NAMES[best_other[1][0]]})')
            if 0.1 <= same_cls_loc < IOU_MATCH:
                tags.append('poor_localisation')
            if rel < 0.005:
                tags.append('small_object(<0.5% of image)')
            if overl >= 0.3:
                tags.append('crowded_or_overlapping_gt')
            bb = brightness(img, box)
            if bb < 60:
                tags.append('dark')
            elif bb > 200:
                tags.append('overexposed')
            fails.append({'type': 'FN', 'image': 'images/' + os.path.basename(img_path), 'class': NAMES[c], 'box': [round(v, 1) for v in box],
                          'rel_area': round(rel, 5), 'crop_brightness': round(bb, 1), 'tags': tags or ['other']})
        for pi, (c, box, s) in enumerate(preds):
            if pi in pr_used:
                continue
            best = max(((iou(box, g[1]), g) for g in gt), default=(0, None))
            tags = []
            if best[0] >= IOU_MATCH and best[1][0] != c:
                tags.append(f'class_confusion(gt {NAMES[best[1][0]]})')
            elif best[0] >= IOU_MATCH:
                tags.append('duplicate_detection')
            elif 0.1 <= best[0] < IOU_MATCH:
                tags.append('poor_localisation')
            else:
                tags.append('no_gt_nearby(background or missing label)')
            fails.append({'type': 'FP', 'image': 'images/' + os.path.basename(img_path), 'class': NAMES[c], 'conf': round(s, 3),
                          'box': [round(v, 1) for v in box], 'tags': tags, 'image_brightness': round(img_b, 1)})
    with open(os.path.join(out, f'failures_{tag}.json'), 'w') as f:
        json.dump(fails, f, indent=1)
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--weights', nargs='+', required=True, help='tag=path')
    ap.add_argument('--analyse', default=None, help='tag of the model to run failure analysis on')
    ap.add_argument('--exclude-stems', default=None, help='file of filename stems; re-run val without them')
    ap.add_argument('--out', default='eval/results')
    a = ap.parse_args()
    random.seed(0); np.random.seed(0)
    os.makedirs(a.out, exist_ok=True)
    root = os.path.dirname(os.path.abspath(a.data))
    res = {'data': a.data, 'n_images': len(glob.glob(os.path.join(root, 'images', '*.jpg'))), 'models': {}}
    weights = dict(w.split('=', 1) for w in a.weights)
    for tag, w in weights.items():
        res['models'][tag] = val_model(tag, w, a.data, a.out)
    if a.exclude_stems:
        stems = {l.strip() for l in open(a.exclude_stems) if l.strip()}
        keep = [p for p in sorted(glob.glob(os.path.join(root, 'images', '*.jpg')))
                if os.path.basename(p).split('.rf.')[0] not in stems]
        lst = os.path.join(root, 'images_no_train_stem_overlap.txt')
        open(lst, 'w').write('\n'.join(keep) + '\n')
        yml = os.path.join(root, 'data_no_overlap.yaml')
        open(yml, 'w').write(f"path: {root}\ntrain: {lst}\nval: {lst}\ntest: {lst}\nnc: 6\nnames: {NAMES}\n")
        res['sensitivity_no_train_stem_overlap'] = {'n_images': len(keep), 'models': {}}
        for tag, w in weights.items():
            r = val_model(tag + '_nooverlap', w, yml, a.out)
            res['sensitivity_no_train_stem_overlap']['models'][tag] = {'all': r['all'], 'per_class': r['per_class']}
    with open(os.path.join(a.out, 'metrics.json'), 'w') as f:
        json.dump(res, f, indent=1)
    if a.analyse:
        r = val_model(a.analyse + f'_conf{APP_CONF}', weights[a.analyse], a.data, a.out, conf=APP_CONF)
        res['app_threshold'] = {'model': a.analyse, 'conf': APP_CONF, 'all': r['all'],
                                'confusion_matrix': r['confusion_matrix']}
        with open(os.path.join(a.out, 'metrics.json'), 'w') as f:
            json.dump(res, f, indent=1)
        fails = analyse(a.analyse, weights[a.analyse], os.path.join(root, 'images'), os.path.join(root, 'labels'), a.out)
        print(f'{len(fails)} errors at conf={APP_CONF}')
    for tag, r in res['models'].items():
        print(tag, r['all'])


if __name__ == '__main__':
    main()
