# -*- coding: utf-8 -*-
"""AudioSet 527 类打标模型测试入口: 在 FSD50K eval 集上算 lwlrap / mAP / F1

用法 (项目根目录下执行):
    python tests.py
    python tests.py --split both                      # 同时看 dev val
    python tests.py --pt checkpoints/audioset_ast/best_model_ast.pt

产物:
    checkpoints/audioset_ast/test_report.md   人读报告
    checkpoints/audioset_ast/test_report.json 机器可读指标
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ASTForAudioClassification

from audionet.data.fsd50k_multilabel import MODEL_ID, NUM_CLASSES, MultiLabelCache, predict
from audionet.labels.audioset_classes import EN_LIST, label_cn
from audionet.utils.metrics import lwlrap, mean_ap, macro_f1_at_topk

ROOT = os.path.dirname(os.path.abspath(__file__))


def eval_split(model, cache_dir, split, device, batch_size=64):
    ds = MultiLabelCache(cache_dir, split)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    truth = np.load(os.path.join(cache_dir, "labels_%s.npy" % split))
    t0 = time.time()
    probs = predict(model, loader, device)
    lw, per_class_lw = lwlrap(truth, probs)
    mp, aps, idx = mean_ap(truth, probs)
    f1, prec, rec = macro_f1_at_topk(truth, probs, k=3)
    print("[%s] n=%d | lwlrap %.4f | mAP %.4f | top3-F1 %.4f (P %.4f / R %.4f) | %.1fs" % (
        split, len(truth), lw, mp, f1, prec, rec, time.time() - t0))
    return {"split": split, "n": int(len(truth)), "lwlrap": lw, "mAP": mp,
            "top3_f1": f1, "top3_precision": prec, "top3_recall": rec,
            "per_class_lwlrap": per_class_lw, "per_class_ap": aps,
            "ap_class_idx": idx, "probs": probs, "truth": truth}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="D:/datasets/FSD50K/ast_cache")
    ap.add_argument("--save_dir", default=os.path.join(ROOT, "checkpoints", "audioset_ast"))
    ap.add_argument("--pt", default="")
    ap.add_argument("--split", default="eval", choices=["eval", "val", "both"])
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    pt = args.pt or os.path.join(args.save_dir, "best_model_ast.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(pt, map_location="cpu")
    print("载入权重: %s | epoch=%s val_lwlrap=%.4f" % (
        pt, ckpt.get("epoch"), ckpt.get("val_lwlrap", float("nan"))))

    model = ASTForAudioClassification.from_pretrained(
        MODEL_ID, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True,
        local_files_only=True)   # 离线环境: 直接用本地 HF 缓存
    model.load_state_dict(ckpt["state_dict"])
    model.gradient_checkpointing_disable()      # 推理不需要激活重算, 关掉更快
    model.to(device).eval()

    splits = ["val", "eval"] if args.split == "both" else [args.split]
    results = {}
    for sp in splits:
        path = os.path.join(args.cache_dir, "labels_%s.npy" % sp)
        if not os.path.exists(path):
            print("缺少缓存, 跳过:", path)
            continue
        results[sp] = eval_split(model, args.cache_dir, sp, device, args.batch_size)

    # ---------- 报告 ----------
    os.makedirs(args.save_dir, exist_ok=True)
    main_res = results.get("eval") or results.get("val")
    aps, idx = main_res["per_class_ap"], main_res["ap_class_idx"]
    order = np.argsort(aps)[::-1]
    truth_support = main_res["truth"].sum(axis=0)

    md = ["# AudioSet 527 类打标模型 测试报告", "",
          "- 模型: AST (MIT/ast-finetuned-audioset-10-10-0.4593) 在 FSD50K 上部分微调",
          "- 测试时间: %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
          "- 权重: %s (epoch %s)" % (pt, ckpt.get("epoch")),
          "- 输入: 16kHz 单声道 -> tile 到 10s -> kaldi fbank 128 mel -> 1024x128", ""]
    for sp, r in results.items():
        md += ["## %s 集指标 (n=%d)" % (sp, r["n"]), "",
               "| 指标 | 数值 |", "| --- | --- |",
               "| lwlrap | %.4f |" % r["lwlrap"],
               "| mAP | %.4f |" % r["mAP"],
               "| top-3 micro-F1 | %.4f |" % r["top3_f1"],
               "| top-3 precision | %.4f |" % r["top3_precision"],
               "| top-3 recall | %.4f |" % r["top3_recall"], ""]

    md += ["## %s 集 单类 AP: 最好 15 类" % main_res["split"], "",
           "| 类别 | 中文 | 测试样本数 | AP |", "| --- | --- | --- | --- |"]
    for i in order[:15]:
        j = int(idx[i])
        md.append("| %s | %s | %d | %.4f |" % (
            EN_LIST[j], label_cn(EN_LIST[j]), int(truth_support[j]), aps[i]))
    md += ["", "## %s 集 单类 AP: 最差 15 类 (样本数>=5)" % main_res["split"], "",
           "| 类别 | 中文 | 测试样本数 | AP |", "| --- | --- | --- | --- |"]
    weak = [i for i in order[::-1] if truth_support[idx[i]] >= 5][:15]
    for i in weak:
        j = int(idx[i])
        md.append("| %s | %s | %d | %.4f |" % (
            EN_LIST[j], label_cn(EN_LIST[j]), int(truth_support[j]), aps[i]))

    # 逐条样例: 取测试集前 12 条, 展示真实标签与预测 top-3
    names = json.load(open(os.path.join(args.cache_dir, "names_%s.json" % main_res["split"])))
    probs, truth = main_res["probs"], main_res["truth"]
    md += ["", "## 样例预测 (前 12 条)", "",
           "| 片段 | 真实标签 | 预测 top-3 (概率) |", "| --- | --- | --- |"]
    for i in range(min(12, len(names))):
        gt = [EN_LIST[j] for j in np.flatnonzero(truth[i])]
        top = np.argsort(probs[i])[::-1][:3]
        pred = "%s(%.3f)" % (EN_LIST[top[0]], probs[i][top[0]])
        for j in top[1:]:
            pred += " / %s(%.3f)" % (EN_LIST[j], probs[i][j])
        md.append("| %s.wav | %s | %s |" % (names[i], ", ".join(gt) or "-", pred))
    md.append("")

    with open(os.path.join(args.save_dir, "test_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    summary = {sp: {k: v for k, v in r.items()
                    if k not in ("probs", "truth", "per_class_lwlrap",
                                 "per_class_ap", "ap_class_idx")}
               for sp, r in results.items()}
    summary["per_class_ap"] = {EN_LIST[int(idx[i])]: float(aps[i]) for i in range(len(aps))}
    json.dump(summary, open(os.path.join(args.save_dir, "test_report.json"), "w"),
              ensure_ascii=False, indent=1)
    print("报告已写出:", os.path.join(args.save_dir, "test_report.md"))


if __name__ == "__main__":
    main()
