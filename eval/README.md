# PPE detector: test-set evaluation, error analysis and ONNX export (November 2026)

Everything here was run on CPU on 11 November 2026 with Ultralytics 8.4.210,
torch 2.15.0+cpu, onnxruntime 1.31.1, Python 3.13.16 (4 threads, Intel Xeon
2.1 GHz). Scripts are in this folder; raw outputs are in `results/`.

## What was evaluated

**Checkpoints** (all YOLOv8s, 11.1M parameters, the same 6 classes in this order:
`Gloves, Helmet, No-Helmet, No-gloves, No-vest, Vest`):

| Tag | File | Where | Why |
|---|---|---|---|
| deployed | `best_deep.pt` | Hugging Face `iMaximusiV/yolo-ppe-detector` | default in `backend/backend_api.py` |
| readme_production | `best_100Epoch.pt` | same Hugging Face repo | named "Production model" in the README |
| repo | `bestV8_50Epoch.pt` | `model_eval/ppe-v8s-final/weights/` | the only 6-class checkpoint in git |

`ppe-4080-v12/weights/best.pt` is a different 14-class model (Fall-Detected,
Ladder, Mask, Safety Cone and others) trained on another dataset, so it does not
match this test split and was not evaluated.

**Test split.** `yolo-v8-new-model/test` holds 255 label files but no images:
`*.jpg` is in `.gitignore`. The images survive only in commit `238e70b5e6`
("added new dataset", 25 Nov 2025), an earlier Roboflow export of the same
project with the same 253 source images. Its labels were later re-annotated
(1,001 boxes then, 1,237 now). `build_test_set.py` pairs each current label
file with the historic image of the same source id. YOLO labels are
normalised, so the later 512 x 512 stretch resize does not change them; a
visual spot check confirmed the boxes line up. As a cross-check, the
deployed model scores mAP50 0.474 on the historic images with their own
historic labels, against 0.468 on the paired set, so the pairing is not what
drives the results.

Test set: **255 images, 1,237 boxes** (Gloves 93, Helmet 414, No-Helmet 244,
No-gloves 157, No-vest 191, Vest 138).

## Headline (test split, imgsz 640, Ultralytics val, conf 0.001, IoU 0.7)

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| deployed (`best_deep.pt`) | 0.488 | 0.547 | **0.468** | **0.217** |
| readme_production (`best_100Epoch.pt`) | 0.699 | 0.499 | 0.481 | 0.278 |
| repo (`bestV8_50Epoch.pt`) | 0.694 | 0.498 | 0.469 | 0.270 |

These are well below the README's figures (mAP50 0.62 to 0.64, mAP50-95
0.35 to 0.36). The README numbers match the validation metrics stored inside
the checkpoints during training (for example `best_deep.pt`: mAP50 0.640,
mAP50-95 0.342 on the validation split), not a held-out test. The validation
images are not in git, so those numbers could not be re-run here.

### Per class, AP50 / AP50-95

| Class (test boxes) | deployed | readme_production | repo |
|---|---|---|---|
| Gloves (93) | 0.434 / 0.173 | 0.491 / 0.228 | 0.502 / 0.231 |
| Helmet (414) | **0.776** / 0.369 | 0.247 / 0.186 | 0.253 / 0.189 |
| No-Helmet (244) | 0.218 / 0.066 | 0.405 / 0.152 | 0.351 / 0.129 |
| No-gloves (157) | 0.369 / 0.214 | 0.539 / 0.333 | 0.551 / 0.341 |
| No-vest (191) | 0.298 / 0.129 | 0.356 / 0.204 | 0.364 / 0.207 |
| Vest (138) | 0.716 / 0.351 | **0.848** / 0.566 | 0.792 / 0.525 |

The deployed model is the only one that finds helmets reliably (Helmet recall
0.708 against 0.200 for the other two), but it is the weakest on every
"No-" class. For a compliance product the "No-" classes are the ones that
raise alerts, so the choice between checkpoints is a real trade-off, not a
clear win.

### Sensitivity checks

- **Possible train/test overlap.** 40 of the 253 test source ids also appear
  as filenames in train (`results/train_test_stem_overlap.txt`). Some may be
  name collisions between merged source datasets; the train images are not in
  git, so this cannot be checked pixel by pixel. Without those images
  (215 images left): deployed mAP50 0.462 / mAP50-95 0.216, readme_production
  0.444 / 0.226, repo 0.431 / 0.216. The conclusions do not change.
- **Older labels.** On the historic export (same images, the 1,001 older
  labels): deployed 0.474 / 0.228, repo 0.580 / 0.340. The repo model is
  sensitive to which annotation round is used; the deployed model is not.

## Error analysis of the deployed model at the app's threshold

The backend runs the detector at `confidence_threshold=0.5`, so the failure
list uses conf 0.5, same-class IoU 0.5 greedy matching
(`results/failures_deployed_best_deep.json`, one record per error with image
path, box, class and tags).

At conf 0.5: **365 true positives, 872 false negatives, 410 false positives**
(recall 29.5%, precision 47.1%, micro-averaged over boxes). Per-class recall:
Helmet 23.4%, No-Helmet 15.2%, Gloves 34.4%, No-gloves 24.2%, No-vest 59.2%,
Vest 34.8%.

Confusion matrix from Ultralytics at conf 0.5 (rows predicted, columns true;
its matcher differs slightly from the one above):

| pred \ true | Gloves | Helmet | No-Helmet | No-gloves | No-vest | Vest | background |
|---|---|---|---|---|---|---|---|
| Gloves | 37 | 0 | 0 | 0 | 0 | 0 | 17 |
| Helmet | 0 | 95 | 0 | 0 | 0 | 0 | 3 |
| No-Helmet | 0 | 2 | 41 | 0 | 0 | 0 | 103 |
| No-gloves | 0 | 0 | 0 | 39 | 0 | 0 | 11 |
| No-vest | 0 | 1 | 1 | 0 | 114 | 23 | 240 |
| Vest | 0 | 0 | 0 | 0 | 1 | 58 | 8 |
| background | 56 | 316 | 202 | 118 | 76 | 57 | 0 |

Class mix-ups are rare; almost all errors are against background (missed
objects or detections with no label).

### False negatives (872), grouped by cause

Tags are computed automatically and can overlap; "first tag" counts each FN once.

| Cause | First tag | Any tag | How it is measured |
|---|---|---|---|
| Detected, but below the 0.5 threshold | 382 | 382 | same-class box with IoU >= 0.5 at conf 0.1 to 0.5 |
| Small object | 132 | 289 | box under 0.5% of the image area |
| Poor localisation | 95 | 99 | best same-class box has IoU 0.1 to 0.5 |
| Class confusion | 23 | 28 | other-class box at IoU >= 0.5 (24 are Vest called No-vest) |
| Dark crop | 18 | 69 | mean grey level of the box under 60 |
| Overexposed crop | 2 | 22 | mean grey level over 200 |
| Crowded / overlapping labels | 3 | 5 | another label box overlaps at IoU >= 0.3 |
| None of the above | 217 | 217 | see manual review below |

**The biggest single cause is the threshold**: 382 of 872 misses (44%) are
found by the model at a lower confidence. Helmets in particular are often
scored 0.3 to 0.45 (for example `images/ppe_0720_jpg.rf.1eb4387afd27ed538e0e10eaca9773bd.jpg`
has 13 labelled helmets, all detected between 0.27 and 0.44, none above 0.5).
Lowering the app threshold, or a per-class threshold, would recover many of
these at some cost in false positives.

Occlusion cannot be measured from box labels alone; the overlap proxy only
fires 5 times, so occlusion is not a measured cause here.

**Manual review** of 12 random untagged misses (seed 0): 6 are clear misses of
large, well-lit objects (a red helmet in `images/005376_jpg.rf.9a23f7594769b3fac08fddabadb442bf.jpg`,
hi-vis vests in `images/ppe_0038_jpg.rf.afdef980056852107b522e26673c380c.jpg`
and `images/ppe_0577_jpg.rf.f33679d24f6947086c5935a4bac38b63.jpg`), 4 are
low-resolution or motion-blurred factory CCTV frames (`images/PP02img322_jpg.rf.dcdb1805e930aa7723e3246c4f215161.jpg`,
`images/PP02img413_jpg.rf.c2bfc753e6457255d20ff09e538cf339.jpg`; one has a
detection box burned into the pixels), and 2 have questionable labels (a
No-gloves box over a timestamp overlay in `images/img120_jpg.rf.579f1db833b2314f2fa7527f745ca5bb.jpg`).

By image source (filename prefix), recall at conf 0.5:

| Source | Images | Boxes | Recall | FP |
|---|---|---|---|---|
| `PP02img*` (factory CCTV, gloves) | 12 | 73 | 9.6% | 2 |
| six-digit ids (mostly press photos) | 90 | 447 | 20.8% | 298 |
| `ppe_*` | 99 | 544 | 34.4% | 90 |
| `img*` | 6 | 31 | 41.9% | 6 |
| other | 48 | 142 | 45.8% | 14 |

### False positives (410), grouped by cause

| Cause | Count |
|---|---|
| No label nearby (IoU under 0.1 with every label) | 280 |
| Poor localisation (IoU 0.1 to 0.5 with a same- or other-class label) | 101 |
| Class confusion (IoU >= 0.5 with another class; 24 are No-vest on a labelled Vest) | 28 |
| Duplicate detection | 1 |

Of the 280 "no label nearby", 228 are No-vest and 215 come from the
six-digit press-photo images. **Manual review** of 12 random ones (seed 0):
in all 12 the person really does lack the predicted item (officials in suits
or shirts without a vest, people without a helmet), for example
`images/005420_jpg.rf.b8f46961ead3e922ed0ab00aa44b8ee1.jpg` and
`images/005346_jpg.rf.da4f4bbf4418a9a29443522f37a8bee9.jpg`. The test labels
mark No-vest on some people in a crowd but not all, so a large share of
these "false positives" are missing annotations. Measured precision for
No-vest and No-Helmet is therefore pessimistic, by an amount this sample is
too small to estimate.

## ONNX export for a browser demo

Exported from the deployed checkpoint (`best_deep.pt`) with
`eval/export_onnx.py`. Files are **not** in git (size); they live outside
the repo. Opset 12, static batch 1, no NMS in the graph, `simplify=True`.

| File | Size | Input | Output | Test mAP50 / mAP50-95 | CPU latency p50 / p95 |
|---|---|---|---|---|---|
| `ppe_yolov8s_640.onnx` | 44.75 MB (44,754,287 B) | `images` float32 [1,3,640,640] | `output0` float32 [1,10,8400] | 0.461 / 0.215 | 93.3 / 296.8 ms |
| `ppe_yolov8s_640_int8dyn.onnx` | 11.49 MB (11,492,847 B) | `images` float32 [1,3,640,640] | `output0` float32 [1,10,8400] | 0.455 / 0.211 | 130.3 / 158.8 ms |
| `ppe_yolov8s_320.onnx` | 44.63 MB (44,628,205 B) | `images` float32 [1,3,320,320] | `output0` float32 [1,10,2100] | 0.419 / 0.191 | 22.5 / 28.2 ms |

Latency: onnxruntime 1.31.1 `CPUExecutionProvider`, 4 intra-op threads,
10 warm-up runs then 100 timed runs over real test images, model forward
only (no preprocessing, no NMS). The fp32 640 run had a long tail (mean
128.2 ms against p50 93.3 ms) on this shared VM, so treat its p95 with care.
The int8 file is dynamic weight quantisation (`quantize_dynamic`, QUInt8); it
is 4x smaller and loses 0.006 mAP50 here, but runs slower than fp32 on this
CPU. It uses `ConvInteger`/`DynamicQuantizeLinear`, which was not tested in
onnxruntime-web. An fp16 variant was attempted with onnxconverter-common 1.16
and did not load in onnxruntime (Resize cast type error), so none is
provided.

**Decoding `output0` in the browser**

- Preprocess: RGB, letterbox to 640 x 640 (or 320) keeping aspect ratio,
  pad with grey 114, scale to 0..1, layout CHW, batch 1.
- `output0[0]` is 10 x N (N = 8400 at 640, 2100 at 320). Column j is one
  candidate: rows 0 to 3 are box centre x, centre y, width, height in
  letterboxed input pixels; rows 4 to 9 are class scores, already sigmoid
  (0 to 1), in the order `Gloves, Helmet, No-Helmet, No-gloves, No-vest, Vest`.
  There is no separate objectness score.
- Score = max over rows 4 to 9, class = argmax. Keep score >= 0.5 to match
  the app, run class-wise NMS (IoU 0.7 matches Ultralytics defaults), then
  undo the letterbox: subtract the padding and divide by the resize ratio.
- `onnx_decode_check.py` does exactly this in numpy and matches
  Ultralytics' own boxes to within a pixel or two; borderline scores move by
  about 0.02 because Ultralytics letterboxes to a stride multiple with cv2
  resize, so a box near 0.5 may appear in one and not the other.
- Class names are also stored in the ONNX metadata (`names`).

## Dataset counts (measured from the label files in this repo)

| Split | Label files | Distinct source images | Boxes |
|---|---|---|---|
| train | 5,847 | 1,837 | 20,533 |
| valid | 265 | 264 | 1,640 |
| test | 255 | 253 | 1,237 |
| total | 6,367 | 2,354 | 23,410 |

Train boxes per class: Gloves 4,319, Helmet 3,979, No-Helmet 2,570,
No-gloves 3,435, No-vest 3,665, Vest 2,565. Train is about 3x augmented
(1,729 source ids appear 3 times); valid and test are not, which differs from
`docs/MODEL_TRAINING.md`, which says all splits were tripled. The Roboflow
export note says 6,376 images, 9 more than the label files present here.

## Reproduce

```
git fetch --depth 1 origin 238e70b5e6dc4f61393f1e7fb7181a1d394a4bea
mkdir -p /tmp/old && git archive 238e70b5e6dc4f61393f1e7fb7181a1d394a4bea yolo-v8-new-model/test | tar -x -C /tmp/old
python eval/build_test_set.py --old /tmp/old/yolo-v8-new-model/test --out /tmp/ppe_test
# best_deep.pt and best_100Epoch.pt from huggingface.co/iMaximusiV/yolo-ppe-detector
python eval/ppe_eval.py --data /tmp/ppe_test/data.yaml --out eval/results \
  --weights deployed_best_deep=best_deep.pt readme_best_100Epoch=best_100Epoch.pt \
            repo_bestV8_50Epoch=model_eval/ppe-v8s-final/weights/bestV8_50Epoch.pt \
  --analyse deployed_best_deep --exclude-stems eval/results/train_test_stem_overlap.txt
python eval/export_onnx.py --weights best_deep.pt --outdir /tmp/ppe_onnx \
  --data /tmp/ppe_test/data.yaml --report eval/results/onnx_export.json
```

The historic-label cross-check was a one-off `YOLO(w).val(data=<historic
test>/data.yaml, split="test", imgsz=640, batch=16, device="cpu")` on the
untouched folder from the `git archive` step. Wall clock: `ppe_eval.py` 3 min
56 s, `export_onnx.py` 2 min 7 s. Absolute scratch paths in the JSON outputs
were replaced with file names after the run.
