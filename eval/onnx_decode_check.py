"""Decode raw ONNX output in numpy (what a browser demo must do) and compare with Ultralytics.

Usage: python eval/onnx_decode_check.py <scratch dir holding ppe_eval/test_set, ppe_onnx/, ppe_eval/hf_weights/>
"""
import sys, glob, numpy as np, onnxruntime as ort
from PIL import Image
from ultralytics import YOLO
S = sys.argv[1]
img = sorted(glob.glob(f'{S}/ppe_eval/test_set/images/*.jpg'))[3]
im = Image.open(img).convert('RGB'); w, h = im.size
r = min(640 / w, 640 / h); nw, nh = round(w * r), round(h * r); px, py = (640 - nw) // 2, (640 - nh) // 2
c = Image.new('RGB', (640, 640), (114, 114, 114)); c.paste(im.resize((nw, nh), Image.BILINEAR), (px, py))
x = (np.asarray(c, np.float32).transpose(2, 0, 1)[None] / 255.0)
out = ort.InferenceSession(f'{S}/ppe_onnx/ppe_yolov8s_640.onnx', providers=['CPUExecutionProvider']).run(None, {'images': x})[0][0]
print('score range', float(out[4:].min()), float(out[4:].max()))
boxes, scores = out[:4].T, out[4:].T
cls = scores.argmax(1); conf = scores.max(1); keep = conf >= 0.5
b, s, k = boxes[keep], conf[keep], cls[keep]
xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
def nms(bx, sc, thr=0.7):
    o = sc.argsort()[::-1]; res = []
    while len(o):
        i = o[0]; res.append(i)
        xx1 = np.maximum(bx[i, 0], bx[o[1:], 0]); yy1 = np.maximum(bx[i, 1], bx[o[1:], 1])
        xx2 = np.minimum(bx[i, 2], bx[o[1:], 2]); yy2 = np.minimum(bx[i, 3], bx[o[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
        iou = inter / (a(bx[[i]]) + a(bx[o[1:]]) - inter)
        o = o[1:][iou < thr]
    return res
mine = []
for cl in np.unique(k):
    idx = np.where(k == cl)[0]
    for j in nms(xyxy[idx], s[idx]):
        bb = (xyxy[idx][j] - [px, py, px, py]) / r
        mine.append((int(cl), [round(float(v)) for v in bb], round(float(s[idx][j]), 3)))
ref = YOLO(f'{S}/ppe_eval/hf_weights/best_deep.pt').predict(img, imgsz=640, conf=0.5, device='cpu', verbose=False)[0]
ul = [(int(a), [round(v) for v in bb.tolist()], round(float(c_), 3)) for a, bb, c_ in zip(ref.boxes.cls, ref.boxes.xyxy, ref.boxes.conf)]
print('numpy decode:', sorted(mine)); print('ultralytics :', sorted(ul))
