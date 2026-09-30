# -*- coding: utf-8 -*-
"""FSD50K -> AST 527 类多标签特征缓存

流程:
    1. 读 FSD50K.ground_truth/dev.csv (含 split 列: train/val) 与 eval.csv
    2. FSD50K 的 200 个类 (AudioSet MID) 映射到 AST(AudioSet) 527 类索引
       - 类别清单读 FSD50K.ground_truth/vocabulary.csv (index,label,mid)
       - 标签以 mids 字段为准 (labels 字段是逗号拼接的类名, 以 MID 取标签更直接)
       - 类名归一化后与 AST 类名匹配, 个别写法差异走 OVERRIDES 人工修正表
       - 未匹配的 MID 是 AudioSet 本体里的中间节点(如 Human voice), 其片段会被丢弃
    3. 音频 -> 16kHz 单声道 -> tile 补齐到 10s -> kaldi fbank(128 mel) -> 1024x128
       归一化 (mean=-4.2677393, std=4.5689974, /2) 后以 fp16 落盘
       (逐条在 CPU 上调用 kaldi fbank, 与 HF ASTFeatureExtractor 输出完全一致;
        批量传入会被当成多通道, 因此不能整批调用)
    4. 多标签 target 存成 [N, 527] uint8 多热向量

用法:
    python -m scripts.prepare_fsd50k                       # 默认规模
    python -m scripts.prepare_fsd50k --train_limit 12000 --val_limit 0 --eval_limit 0
    python -m scripts.prepare_fsd50k --train_limit 0        # 0 表示全量

产物 (默认 D:/datasets/FSD50K/ast_cache):
    feats_train.npy / labels_train.npy / names_train.json
    feats_val.npy   / labels_val.npy   / names_val.json
    feats_eval.npy  / labels_eval.npy  / names_eval.json
    classes.json     (527 类名, 与 logits 索引对应)
    mapping_report.json (类名映射明细, 便于核对)
"""
import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi

from audionet.labels.audioset_classes import EN_LIST

SR = 16000
CLIP_SECONDS = 10
N_SAMPLES = SR * CLIP_SECONDS
MAX_LEN = 1024                      # AST 时间帧数
MEAN, STD = -4.2677393, 4.5689974   # AST 归一化常数 (来自 preprocessor_config.json)

# FSD50K 类名与 AST 类名存在个别写法差异, 归一化后仍匹配不上的在这里人工修正
# 格式: FSD50K 类名 -> AST 527 类名 (同一 AudioSet MID 在两边写法不同)
OVERRIDES = {
    "Crash_cymbal": "Cymbal",
}


def norm(s):
    """类名归一化: 下划线->空格, &->and, 再去掉连接词 and 与标点

    FSD50K 用 "_and_" 连接同义词 (Male_speech_and_man_speaking),
    AST 用 ", " 连接 (Male speech, man speaking), 归一化后两者一致。
    """
    s = s.replace("_", " ").replace("&", " and ").replace("/", " ")
    s = s.lower().replace(",", " ").replace("-", " ").replace("'", "")
    return " ".join(t for t in s.split() if t != "and")


def build_mid_map(gt_dir):
    """FSD50K 200 类 (AudioSet MID) -> AST 527 类索引

    类别清单来自 FSD50K.ground_truth/vocabulary.csv (index,label,mid)。
    以 MID 建索引: dev.csv/eval.csv 的 labels 字段用逗号拼接, 而部分类名本身
    含逗号 (如 "Male speech, man speaking"), 按 MID 取标签可完全避开该歧义。
    """
    ast_norm = {norm(n): i for i, n in enumerate(EN_LIST)}
    mid2label = {}
    with open(os.path.join(gt_dir, "vocabulary.csv"), encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                mid2label[row[2].strip()] = row[1].strip()

    mid2ast, label2ast, detail, unmatched = {}, {}, {}, []
    for mid, label in sorted(mid2label.items()):
        hit = ast_norm.get(norm(OVERRIDES.get(label, label)))
        mid2ast[mid] = hit
        label2ast[label] = hit
        detail[mid] = {"label": label, "ast_idx": hit,
                       "ast_name": EN_LIST[hit] if hit is not None else ""}
        if hit is None:
            unmatched.append(mid)
    return mid2ast, label2ast, detail, unmatched


def read_gt(dirs):
    """读 dev.csv / eval.csv -> [(fname, [label,...], [mid,...], split)]"""
    rows = []
    for fn, split_default in [("dev.csv", "train"), ("eval.csv", "eval")]:
        path = next((os.path.join(d, fn) for d in dirs
                     if os.path.exists(os.path.join(d, fn))), None)
        if path is None:
            print("缺少:", fn)
            continue
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                fname = r["fname"].strip()
                # FSD50K 的类名是下划线写法(不含逗号), 逗号只作分隔符
                labels = [x.strip() for x in (r.get("labels") or "").split(",") if x.strip()]
                mids = [x.strip() for x in (r.get("mids") or "").split(",") if x.strip()]
                split = (r.get("split") or split_default).strip()
                rows.append((fname, labels, mids, split))
    return rows


def row_targets(row, mid2ast, label2ast):
    """一行 ground truth -> AST 类别索引集合 (优先 mids, labels 兜底)"""
    _, labels, mids, _ = row
    idx = {mid2ast[m] for m in mids if mid2ast.get(m) is not None}
    if not idx:
        idx = {label2ast[lb] for lb in labels if label2ast.get(lb) is not None}
    return idx


def stratified_take(data, limit, seed):
    """按 AST 类别分层抽稀, 保证每个出现过的类至少保留 1 条, 总量不超过 limit

    一条片段常带多个标签, 因此按「稀有类优先」贪心挑选: 先补样本数少的类,
    避免长尾类别被抽空。data: [(row, target_set)]
    """
    if limit <= 0 or limit >= len(data):
        return data
    by_class = {}
    for i, (_, t) in enumerate(data):
        for j in t:
            by_class.setdefault(j, []).append(i)
    rng = random.Random(seed)
    ratio = limit / len(data)
    need = {j: max(1, int(round(len(v) * ratio))) for j, v in by_class.items()}
    got = {j: 0 for j in by_class}
    chosen, picked = set(), []
    for j in sorted(by_class, key=lambda k: len(by_class[k])):
        idxs = by_class[j][:]
        rng.shuffle(idxs)
        for i in idxs:
            if got[j] >= need[j] or len(picked) >= limit:
                break
            if i in chosen:
                continue
            chosen.add(i)
            picked.append(i)
            for k in data[i][1]:
                got[k] += 1
        if len(picked) >= limit:
            break
    out = [data[i] for i in sorted(picked)]
    print("  分层抽稀 %d -> %d 条 (稀有类优先, 覆盖类别 %d)" % (
        len(data), len(out), sum(1 for j in got if got[j] > 0)))
    return out


def load_clip(path):
    """wav -> 16kHz 单声道 float32 tensor"""
    try:
        wav, sr = torchaudio.load(path)
    except Exception:
        wav, sr = torchaudio.load(path, backend="soundfile")
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav[0]
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    n = wav.numel()
    if n < N_SAMPLES:                     # AST 官方做法: 重复填充而非补零
        rep = int(np.ceil(N_SAMPLES / max(n, 1)))
        wav = wav.repeat(rep)[:N_SAMPLES]
    else:
        wav = wav[:N_SAMPLES]
    return wav


def build_split(data, feats_path, labels_path, names_path, args):
    """一个 split 的特征 + 多标签落盘 (data: [(row, target_set)])"""
    if os.path.exists(feats_path) and os.path.exists(labels_path):
        print("  已存在, 跳过:", os.path.basename(feats_path))
        return
    n = len(data)
    feats = np.lib.format.open_memmap(feats_path, mode="w+",
                                      dtype=np.float16, shape=(n, MAX_LEN, 128))
    mh = np.zeros((n, len(EN_LIST)), dtype=np.uint8)
    names = []
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=args.workers)

    for s in range(0, n, args.batch):
        blk = data[s:s + args.batch]
        dirs = [os.path.join(args.data_dir,
                             "FSD50K.eval_audio" if row[3] == "eval" else "FSD50K.dev_audio",
                             row[0] + ".wav") for row, _ in blk]
        wavs = list(pool.map(load_clip, dirs))
        # 逐条调 kaldi fbank (与 HF ASTFeatureExtractor 完全一致, 偏差 0);
        # 批量传入会被当成多通道, 因此这里逐条调用
        fbs = []
        for w in wavs:
            fb = ta_kaldi.fbank(w.unsqueeze(0), sample_frequency=SR,
                                window_type="hanning", num_mel_bins=128)
            if fb.shape[0] < MAX_LEN:
                fb = torch.nn.functional.pad(fb, (0, 0, 0, MAX_LEN - fb.shape[0]))
            else:
                fb = fb[:MAX_LEN]
            fbs.append(((fb - MEAN) / (STD * 2)).to(torch.float16).numpy())
        feats[s:s + len(blk)] = np.stack(fbs)

        for k, (row, target) in enumerate(blk):
            for j in target:
                mh[s + k, j] = 1
            names.append(row[0])

        if (s // args.batch) % 10 == 0 or s + args.batch >= n:
            done = min(s + len(blk), n)
            speed = done / max(time.time() - t0, 1e-6)
            print("  %d/%d  %.1f 条/s  预计剩余 %.1f 分钟" % (
                done, n, speed, (n - done) / speed / 60), flush=True)
    feats.flush()
    del feats
    np.save(labels_path, mh)
    json.dump(names, open(names_path, "w"))
    print("  完成 %s: %d 条, 耗时 %.1f 分钟" % (
        os.path.basename(feats_path)[6:-4], n, (time.time() - t0) / 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="D:/datasets/FSD50K")
    ap.add_argument("--cache_dir", default="")
    ap.add_argument("--train_limit", type=int, default=14000,
                    help="dev train 抽样条数, 0=全量")
    ap.add_argument("--val_limit", type=int, default=0, help="0=全量 dev val")
    ap.add_argument("--eval_limit", type=int, default=0, help="0=全量 eval")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cache_dir = args.cache_dir or os.path.join(args.data_dir, "ast_cache")
    os.makedirs(cache_dir, exist_ok=True)
    gt_dir = os.path.join(args.data_dir, "FSD50K.ground_truth")
    meta_dir = os.path.join(args.data_dir, "FSD50K.metadata")

    # 类别清单固定在 ground_truth/vocabulary.csv
    src_dir = next((d for d in (gt_dir, meta_dir)
                    if os.path.isfile(os.path.join(d, "vocabulary.csv"))), gt_dir)
    mid2ast, label2ast, detail, unmatched = build_mid_map(src_dir)
    covered = [m for m, v in mid2ast.items() if v is not None]
    print("FSD50K MID 映射: %d/%d 成功 -> AST 527 类" % (len(covered), len(mid2ast)))
    if unmatched:
        print("  未匹配 MID(%d): %s" % (
            len(unmatched), [detail[m]["label"] for m in unmatched]))
    json.dump({"detail": detail, "unmatched": unmatched},
              open(os.path.join(cache_dir, "mapping_report.json"), "w"),
              ensure_ascii=False, indent=1)
    json.dump(EN_LIST, open(os.path.join(cache_dir, "classes.json"), "w"),
              ensure_ascii=False)

    rows = read_gt([gt_dir, meta_dir])
    # 丢掉映射后完全没有标签的片段 (纯负样本会干扰 BCE 训练)
    n0 = len(rows)
    data = [(r, t) for r, t in
            ((r, row_targets(r, mid2ast, label2ast)) for r in rows) if t]
    print("有效标注片段: %d/%d" % (len(data), n0))
    train = [d for d in data if d[0][3] == "train"]
    val = [d for d in data if d[0][3] == "val"]
    test = [d for d in data if d[0][3] not in ("train", "val")]
    print("原始划分: train=%d val=%d eval=%d" % (len(train), len(val), len(test)))

    rng = random.Random(args.seed)
    if args.eval_limit > 0 and args.eval_limit < len(test):
        test = sorted(rng.sample(test, args.eval_limit), key=lambda d: d[0][0])
    train = stratified_take(train, args.train_limit, args.seed)
    if args.val_limit > 0 and args.val_limit < len(val):
        val = sorted(rng.sample(val, args.val_limit), key=lambda d: d[0][0])

    print("缓存目录:", cache_dir)
    for tag, split_data in [("train", train), ("val", val), ("eval", test)]:
        print("[%s] %d 条" % (tag, len(split_data)))
        build_split(split_data,
                    os.path.join(cache_dir, "feats_%s.npy" % tag),
                    os.path.join(cache_dir, "labels_%s.npy" % tag),
                    os.path.join(cache_dir, "names_%s.json" % tag),
                    args)
        y = np.load(os.path.join(cache_dir, "labels_%s.npy" % tag))
        print("  多标签统计: 平均 %.2f 类/条, 空标签 %d 条, 覆盖类别 %d/%d" % (
            y.sum() / max(len(y), 1), int((y.sum(1) == 0).sum()),
            int((y.sum(0) > 0).sum()), len(EN_LIST)))


if __name__ == "__main__":
    main()