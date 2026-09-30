# -*- coding: utf-8 -*-
"""AudioSet 527 类打标模型导出 ONNX (含特征提取器配置与类别名, 供离线部署)

用法:
    python -m scripts.export_onnx_audioset_ast
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import ASTForAudioClassification, ASTFeatureExtractor

from audionet.data.fsd50k_multilabel import MODEL_ID, NUM_CLASSES
from audionet.labels.audioset_classes import EN_LIST

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根目录
CKPT = os.path.join(ROOT, "checkpoints", "audioset_ast")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=os.path.join(CKPT, "best_model_ast.pt"))
    ap.add_argument("--onnx", default=os.path.join(CKPT, "audioset_ast.onnx"))
    args = ap.parse_args()

    ckpt = torch.load(args.pt, map_location="cpu")
    model = ASTForAudioClassification.from_pretrained(
        MODEL_ID, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True,
        local_files_only=True)   # 离线环境: 直接用本地 HF 缓存
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("载入权重: epoch=%s val_lwlrap=%.4f" % (ckpt.get("epoch"), ckpt.get("val_lwlrap", 0)))

    dummy = torch.randn(1, 1024, 128)  # [batch, time, mel]
    with torch.no_grad():
        ref = model(input_values=dummy).logits

    if os.path.exists(args.onnx):
        os.remove(args.onnx)
    torch.onnx.export(
        model, (dummy,), args.onnx,
        input_names=["input_values"], output_names=["logits"],
        opset_version=14,
        dynamic_axes={"input_values": {0: "batch"}, "logits": {0: "batch"}},
    )

    out_dir = os.path.dirname(args.onnx)
    # 特征提取器配置 + 类别名落地 (API 离线加载用)
    ASTFeatureExtractor.from_pretrained(
        MODEL_ID, local_files_only=True).save_pretrained(os.path.join(out_dir, "preprocessor"))
    json.dump(EN_LIST, open(os.path.join(out_dir, "labels.json"), "w"), ensure_ascii=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input_values": dummy.numpy()})[0]
    diff = float(np.abs(out - ref.numpy()).max())
    print("ONNX 导出完成: %s (%.1fMB)" % (args.onnx, os.path.getsize(args.onnx) / 1e6))
    print("torch vs onnxruntime 最大偏差: %.2e" % diff)
    assert diff < 1e-3, "偏差过大"


if __name__ == "__main__":
    main()