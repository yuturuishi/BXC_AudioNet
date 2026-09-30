# -*- coding: utf-8 -*-
"""AudioSet 527 类音频打标 API 服务 (AST 模型, onnxruntime CPU 推理, 多标签)

POST /audio/tag   multipart 表单字段 file=<音频文件 wav/mp3/flac/ogg>
                 可选表单字段 topk=<返回条数, 默认 8>  threshold=<概率阈值, 默认 0.1>
返回: {code, msg, result:{tags:[{rank,label_en,label_cn,prob}], duration, count}}

接口返回多标签打标结果, 一条音频可同时命中多个标签。

依赖: flask transformers onnxruntime soundfile scipy numpy
启动(项目根目录下执行): python -m api.audioset_ast_api   (默认端口 8072)
"""
import io
import json
import os

import numpy as np
import onnxruntime as ort
import scipy.signal
import soundfile as sf
from flask import Flask, jsonify, request
from transformers import ASTFeatureExtractor

from audionet.labels.audioset_classes import EN_LIST, label_cn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根目录
MODEL_DIR = os.path.join(ROOT, "checkpoints", "audioset_ast")
ONNX_PATH = os.path.join(MODEL_DIR, "audioset_ast.onnx")
FE_DIR = os.path.join(MODEL_DIR, "preprocessor")
SR = 16000
CLIP_SECONDS = 10

app = Flask(__name__)

# 特征提取器: 优先加载本地配置(离线), 兜底走 HF 缓存
if os.path.isdir(FE_DIR):
    fe = ASTFeatureExtractor.from_pretrained(FE_DIR)
else:
    fe = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593", local_files_only=True)

labels_path = os.path.join(MODEL_DIR, "labels.json")
LABELS = json.load(open(labels_path, encoding="utf-8")) if os.path.exists(labels_path) else EN_LIST

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name


def load_audio_bytes(data):
    """bytes -> 16kHz mono float32"""
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        wav = scipy.signal.resample_poly(wav, SR, sr)
    return wav


def predict(wav, topk=8, threshold=0.1):
    """16kHz 波形 -> 多标签打标结果"""
    need = SR * CLIP_SECONDS
    if len(wav) < need:
        wav = np.tile(wav, int(np.ceil(need / max(len(wav), 1))))[:need]
    else:
        wav = wav[:need]
    fv = fe(wav, sampling_rate=SR)["input_values"][0]        # [1024, 128]
    logits = sess.run(None, {input_name: fv[None].astype(np.float32)})[0][0]
    probs = 1.0 / (1.0 + np.exp(-logits))                    # 多标签用 sigmoid
    order = np.argsort(probs)[::-1][:topk]
    tags = [{
        "rank": i + 1,
        "label_en": LABELS[idx],
        "label_cn": label_cn(LABELS[idx]),
        "prob": round(float(probs[idx]), 4),
    } for i, idx in enumerate(order) if probs[idx] >= threshold]
    return tags


@app.route("/", methods=["GET"])
def index():
    return "AudioSet 527-class audio tagging API"


@app.route("/audio/tag", methods=["POST"])
def tag():
    try:
        f = request.files.get("file")
        if f is None:
            return jsonify({"code": 0, "msg": "缺少文件字段 file"})
        topk = int(request.form.get("topk", 8))
        threshold = float(request.form.get("threshold", 0.1))
        wav = load_audio_bytes(f.read())
        if len(wav) < SR // 2:
            return jsonify({"code": 0, "msg": "音频过短（至少 0.5 秒）"})
        tags = predict(wav, topk, threshold)
        return jsonify({
            "code": 1000, "msg": "success",
            "result": {
                "count": len(tags),
                "duration": round(len(wav) / SR, 2),
                "tags": tags,
            },
        })
    except Exception as e:
        return jsonify({"code": 0, "msg": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8072)