"""Export the PPE detector to ONNX for onnxruntime-web and measure CPU latency.

Usage:
  python eval/export_onnx.py --weights best_deep.pt --outdir /some/dir/ppe_onnx \
      --data /tmp/ppe_test/data.yaml --report eval/results/onnx_export.json

Produces (static input, batch 1, opset 12, no NMS inside the graph):
  ppe_yolov8s_640.onnx          fp32, 1x3x640x640
  ppe_yolov8s_640_int8dyn.onnx  dynamic int8 weight quantisation
  ppe_yolov8s_320.onnx          fp32, 1x3x320x320
Each file is checked for mAP on the test split with Ultralytics and timed with
onnxruntime (CPUExecutionProvider) on real test images.
"""
import argparse, glob, json, os, shutil, statistics, time

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO

OPSET = 12


def letterbox(img, size):
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    canvas = Image.new('RGB', (size, size), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x)


def latency(path, size, images, warmup=10, runs=100, threads=4):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    xs = [letterbox(Image.open(p).convert('RGB'), size) for p in images]
    for i in range(warmup):
        sess.run(None, {name: xs[i % len(xs)]})
    ts = []
    for i in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: xs[i % len(xs)]})
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    q = lambda p: ts[min(len(ts) - 1, int(round(p * (len(ts) - 1))))]
    return {'runs': runs, 'threads': threads, 'p50_ms': round(q(0.5), 1), 'p95_ms': round(q(0.95), 1),
            'mean_ms': round(statistics.mean(ts), 1), 'note': 'model forward only; excludes preprocessing and NMS'}


def io_spec(path):
    m = onnx.load(path)
    dims = lambda t: [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]
    et = lambda t: onnx.TensorProto.DataType.Name(t.type.tensor_type.elem_type)
    meta = {p.key: p.value for p in m.metadata_props}
    return {'inputs': [{'name': t.name, 'shape': dims(t), 'dtype': et(t)} for t in m.graph.input],
            'outputs': [{'name': t.name, 'shape': dims(t), 'dtype': et(t)} for t in m.graph.output],
            'opset': [o.version for o in m.opset_import if o.domain in ('', 'ai.onnx')][0],
            'metadata': meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--report', required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    root = os.path.dirname(os.path.abspath(a.data))
    images = sorted(glob.glob(os.path.join(root, 'images', '*.jpg')))[:50]

    files = {}
    for size in (640, 320):
        src = YOLO(a.weights).export(format='onnx', imgsz=size, opset=OPSET, simplify=True, dynamic=False,
                                     half=False, nms=False, device='cpu')
        dst = os.path.join(a.outdir, f'ppe_yolov8s_{size}.onnx')
        shutil.move(src, dst)
        files[f'fp32_{size}'] = (dst, size)

    base = files['fp32_640'][0]
    # fp16 was tried with onnxconverter-common 1.16 (keep_io_types=True): the
    # converted graph fails to load in onnxruntime 1.31 (Resize cast type
    # mismatch), and Ultralytics only exports half precision on GPU. The int8
    # variant below covers the small-download case instead.

    from onnxruntime.quantization import QuantType, quantize_dynamic
    p8 = os.path.join(a.outdir, 'ppe_yolov8s_640_int8dyn.onnx')
    quantize_dynamic(base, p8, weight_type=QuantType.QUInt8)
    files['int8dyn_640'] = (p8, 640)

    report = {'weights': a.weights, 'opset': OPSET, 'onnxruntime': ort.__version__, 'variants': {}}
    for key, (path, size) in files.items():
        entry = {'path': path, 'size_bytes': os.path.getsize(path),
                 'size_mb': round(os.path.getsize(path) / 1e6, 2), 'imgsz': size, 'io': io_spec(path)}
        try:
            entry['latency_cpu'] = latency(path, size, images)
        except Exception as e:  # e.g. an op not supported by the CPU EP
            entry['latency_cpu'] = {'error': repr(e)}
        try:
            m = YOLO(path, task='detect').val(data=a.data, split='test', imgsz=size, batch=1, device='cpu',
                                              plots=False, verbose=False, project=os.path.join(a.outdir, 'val'),
                                              name=key, exist_ok=True)
            entry['test_metrics'] = {'mAP50': float(m.box.map50), 'mAP50_95': float(m.box.map),
                                     'precision': float(m.box.mp), 'recall': float(m.box.mr)}
        except Exception as e:
            entry['test_metrics'] = {'error': repr(e)}
        report['variants'][key] = entry
        print(key, entry['size_mb'], 'MB', entry.get('latency_cpu'), entry.get('test_metrics'))
    shutil.rmtree(os.path.join(a.outdir, 'val'), ignore_errors=True)
    with open(a.report, 'w') as f:
        json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
