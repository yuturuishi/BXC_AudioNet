# -*- coding: utf-8 -*-
"""FSD50K 多标签特征缓存读取 + AST 打标推理 (供训练/测试/导出共用)

特征缓存由 scripts.prepare_fsd50k 生成:
    <cache_dir>/feats_<split>.npy   fp16 [N, 1024, 128]  AST 归一化后的 log-mel
    <cache_dir>/labels_<split>.npy  uint8 [N, 527]       多热标签
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from audionet.labels.audioset_classes import EN_LIST

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
NUM_CLASSES = len(EN_LIST)          # AudioSet 527 类


class MultiLabelCache(Dataset):
    """读取 prepare_fsd50k 生成的特征缓存 (fp16 memmap) 与多热标签"""

    def __init__(self, cache_dir, split, limit=0):
        self.feats = np.load(os.path.join(cache_dir, "feats_%s.npy" % split), mmap_mode="r")
        self.labels = np.load(os.path.join(cache_dir, "labels_%s.npy" % split))
        self.limit = limit if limit > 0 else len(self.labels)

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        x = self.feats[idx].astype(np.float32)
        y = self.labels[idx].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def predict(model, loader, device):
    """返回 [N, 527] sigmoid 概率"""
    model.eval()
    outs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(input_values=x).logits
        outs.append(torch.sigmoid(logits.float()).cpu().numpy())
    if outs:
        return np.concatenate(outs, axis=0)
    return np.zeros((0, NUM_CLASSES), dtype=np.float32)