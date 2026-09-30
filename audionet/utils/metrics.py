# -*- coding: utf-8 -*-
"""音频打标 (多标签) 评测指标: lwlrap 与 mAP

lwlrap (label-weighted label-ranking average precision) 是 AudioSet/FSD50K 打标任务
的标准指标, 按每个类别的真实样本数加权平均各类的 per-class lwlrap。
本实现遵循 AudioSet 官方 lwlrap 参考实现 (T. Bertin-Mahieux / Google)。

用法:
    from audionet.utils.metrics import lwlrap, mean_ap
    score, per_class = lwlrap(truth, scores)     # truth/scores 均为 [N, C]
    map_score, ap = mean_ap(truth, scores)
"""
import numpy as np


def _per_sample_positive_class_precisions(scores, truth):
    """单条样本: 返回 (正类索引, 各正类的 precision@rank)"""
    num_classes = scores.shape[0]
    pos = np.flatnonzero(truth > 0)
    ranked = np.argsort(scores)[::-1]              # 按分数降序的类别序列
    rank_of = np.zeros(num_classes, dtype=np.int64)
    rank_of[ranked] = np.arange(num_classes)
    hits_in_rank_order = truth[ranked] > 0         # 注意: 累积命中必须按排名顺序累加
    cum_hits = np.cumsum(hits_in_rank_order)
    prec = cum_hits[rank_of[pos]] / (1.0 + rank_of[pos].astype(np.float64))
    return pos, prec


def lwlrap(truth, scores):
    """返回 (lwlrap, per_class_lwlrap)。truth/scores 形状 [N, C]"""
    truth = np.asarray(truth) > 0
    scores = np.asarray(scores)
    n, c = scores.shape
    acc = np.zeros((n, c), dtype=np.float64)
    for i in range(n):
        if not truth[i].any():
            continue
        pos, prec = _per_sample_positive_class_precisions(scores[i], truth[i])
        acc[i, pos] = prec
    labels_per_class = truth.sum(axis=0).astype(np.float64)
    per_class = acc.sum(axis=0) / np.maximum(1.0, labels_per_class)
    weights = labels_per_class / max(labels_per_class.sum(), 1.0)
    return float(np.dot(per_class, weights)), per_class


def average_precision(truth_col, score_col):
    """单类别 average precision (PR 曲线下面积, 阶梯法)"""
    truth_col = np.asarray(truth_col) > 0
    n_pos = int(truth_col.sum())
    if n_pos == 0:
        return None
    order = np.argsort(score_col)[::-1]
    t = truth_col[order]
    cum_tp = np.cumsum(t)
    precision = cum_tp / (np.arange(len(t)) + 1.0)
    recall = cum_tp / n_pos
    # 只在正样本处取样, 避免 PR 曲线的锯齿高估
    idx = np.flatnonzero(t)
    precision = precision[idx]
    recall = recall[idx]
    prev = 0.0
    ap = 0.0
    for p, r in zip(precision, recall):
        ap += (r - prev) * p
        prev = r
    return float(ap)


def mean_ap(truth, scores):
    """返回 (mAP, 每类 AP 数组, 有正样本的类别索引)。mAP 只对出现过的类别求平均"""
    truth = np.asarray(truth) > 0
    scores = np.asarray(scores)
    aps, idx = [], []
    for j in range(scores.shape[1]):
        ap = average_precision(truth[:, j], scores[:, j])
        if ap is not None:
            aps.append(ap)
            idx.append(j)
    aps = np.asarray(aps)
    return float(aps.mean()) if len(aps) else 0.0, aps, np.asarray(idx, dtype=np.int64)


def macro_f1_at_topk(truth, scores, k=3):
    """预测平均取 top-k 时的 micro-F1, 便于直观理解实际打标质量"""
    truth = np.asarray(truth) > 0
    pred = np.zeros_like(truth, dtype=bool)
    k = min(k, scores.shape[1])
    topk = np.argpartition(-np.asarray(scores), k - 1, axis=1)[:, :k]
    np.put_along_axis(pred, topk, True, axis=1)
    tp = (pred & truth).sum()
    fp = (pred & ~truth).sum()
    fn = (~pred & truth).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return float(f1), float(prec), float(rec)