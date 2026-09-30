# -*- coding: utf-8 -*-
"""AudioSet AST 模型 INT8 动态量化（仅 MatMul/Gemm，Conv 保持 fp32 以免 CPU EP 报 NOT_IMPLEMENTED）。
用法: python -m scripts.quantize_audioset_ast
"""
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic
import soundfile as sf

from transformers import ASTFeatureExtractor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "audioset_ast")
FP32 = os.path.join(CKPT, "audioset_ast.onnx")
INT8 = os.path.join(CKPT, "audioset_ast_int8.onnx")
LABELS = json.load(open(os.path.join(CKPT, "labels.json"), encoding="utf-8"))


def load_wav16(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        import scipy.signal
        wav = scipy.signal.resample_poly(wav, 16000, sr)
    return wav.astype(np.float32)


def infer(sess, fe, wav):
    fv = fe(wav, sampling_rate=16000)["input_values"][0]
    logits = sess.run(None, {sess.get_inputs()[0].name: fv[None].astype(np.float32)})[0][0]
    return 1.0 / (1.0 + np.exp(-logits))


def main():
    fe = ASTFeatureExtractor.from_pretrained(os.path.join(CKPT, "preprocessor"))

    # 量化
    m = onnx.load(FP32)
    quantize_dynamic(m, INT8, op_types_to_quantize=["MatMul", "Gemm"])
    print("INT8 导出: %.1fMB" % (os.path.getsize(INT8) / 1e6))

    fp32_sess = ort.InferenceSession(FP32, providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(INT8, providers=["CPUExecutionProvider"])

    tests = [os.path.join(ROOT, "test_audio", f) for f in
             ["dog.wav", "siren.wav", "crying_baby.wav", "thunderstorm.wav", "rooster.wav", "keyboard_typing.wav"]]

    print("\n%-26s | %s | %s | %s" % ("file", "fp32_top1", "int8_top1", "maxprobdiff"))
    worst = 0.0
    for t in tests:
        if not os.path.exists(t):
            continue
        wav = load_wav16(t)
        p32 = infer(fp32_sess, fe, wav)
        p8 = infer(int8_sess, fe, wav)
        md = float(np.abs(p32 - p8).max())
        worst = max(worst, md)
        o32 = np.argsort(p32)[::-1][:1][0]
        o8 = np.argsort(p8)[::-1][:1][0]
        print("%-26s | %s %.2f | %s %.2f | %.4f" % (
            os.path.basename(t), LABELS[o32], p32[o32], LABELS[o8], p8[o8], md))
    print("\nINT8 与 fp32 最大概率偏差: %.4f" % worst)
    # 推理耗时
    import time
    wav = load_wav16(tests[0])
    fv = fe(wav, sampling_rate=16000)["input_values"][0]
    t0 = time.time()
    for _ in range(3):
        int8_sess.run(None, {int8_sess.get_inputs()[0].name: fv[None].astype(np.float32)})
    print("INT8 单条推理(本机): %.2fs" % ((time.time() - t0) / 3))


if __name__ == "__main__":
    main()
