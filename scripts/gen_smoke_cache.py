# -*- coding: utf-8 -*-
"""为「训练冒烟测试」生成极小合成特征缓存（仅验证训练循环可跑通，非真实数据）

真实训练需要 scripts.prepare_fsd50k 生成的 FSD50K 特征缓存（见 README）。
本脚本只造几条 [1024,128] fp16 特征 + [527] uint8 多热标签，用于证明：
  模型离线加载 -> 前向 -> 反向 -> 验证 lwlrap -> 保存 best_model_ast.pt 全链路可用。

用法（项目根目录下执行）:
    python -m scripts.gen_smoke_cache
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "_smoke_cache")
os.makedirs(OUT, exist_ok=True)

N_TR, N_VA = 8, 4
C = 527
T, M = 1024, 128


def make(n):
    feats = np.random.randn(n, T, M).astype(np.float32) * 2.0 + 1.0
    labels = np.zeros((n, C), dtype=np.uint8)
    # 每条随机点亮 2~4 个类，保证 lwlrap 有正样本可算
    for i in range(n):
        idx = np.random.choice(C, size=np.random.randint(2, 5), replace=False)
        labels[i, idx] = 1
    return feats, labels


for split, n in (("train", N_TR), ("val", N_VA)):
    feats, labels = make(n)
    np.save(os.path.join(OUT, "feats_%s.npy" % split), feats.astype(np.float16))
    np.save(os.path.join(OUT, "labels_%s.npy" % split), labels)
    print("写入 %-6s feats=%s labels=%s" % (split, feats.shape, labels.shape))

print("合成缓存目录:", OUT)
