# -*- coding: utf-8 -*-
"""AudioSet 527 类音频打标训练: AST (AudioSet 预训练) 在 FSD50K 上部分微调

任务: 开集多标签打标 (一条音频可同时属于多个类), 不是单标签分类
- 输出维度直接复用 AudioSet 预训练的 527 类分类头, 不做类别数迁移
- 损失用 BCEWithLogitsLoss, 指标用 lwlrap / mAP (AudioSet 官方评测协议)

用法 (项目根目录下执行):
    python train.py                       # 默认配置
    python train.py --epochs 12 --batch_size 32 --limit_train 2000   # 快速冒烟

要点:
- 特征来自 scripts.prepare_fsd50k 预生成的 fp16 缓存 (D:/datasets/FSD50K/ast_cache)
- 部分微调: 冻结 patch embedding 与前若干层 Transformer, 只训练后几层 + 分类头
- AMP fp16 + gradient checkpointing, 16GB 显存可跑 batch 32
- 每轮在 dev val 上算 lwlrap, 以 lwlrap 最优保存 <项目根>/checkpoints/audioset_ast/best_model_ast.pt
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ASTForAudioClassification, get_cosine_schedule_with_warmup

from audionet.data.fsd50k_multilabel import (MODEL_ID, NUM_CLASSES, MultiLabelCache,
                                             predict)
from audionet.labels.audioset_classes import EN_LIST
from audionet.utils.metrics import lwlrap

ROOT = os.path.dirname(os.path.abspath(__file__))   # 项目根目录, 产物路径以其为基准


def build_model(train_layers, device):
    """加载 527 类 AudioSet 预训练权重, 只解冻后 train_layers 层 + 最后的 layernorm + 分类头"""
    model = ASTForAudioClassification.from_pretrained(
        MODEL_ID, local_files_only=True)   # 离线环境: 直接用本地 HF 缓存权重
    assert model.config.num_labels == NUM_CLASSES, "类别数不匹配: %d" % model.config.num_labels
    model.gradient_checkpointing_enable()   # 激活重算换显存

    trunk = model.audio_spectrogram_transformer
    for p in model.parameters():
        p.requires_grad = False
    layers = trunk.encoder.layer
    for layer in layers[max(0, len(layers) - train_layers):]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in trunk.layernorm.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    print("总参数 %.1fM | 可训练 %.1fM (后 %d/%d 层 + 分类头)" % (
        total / 1e6, sum(p.numel() for p in trainable) / 1e6,
        train_layers, len(layers)))
    return model.to(device), trainable


@torch.no_grad()
def evaluate_lwlrap(model, loader, truth, device):
    return lwlrap(truth, predict(model, loader, device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="D:/datasets/FSD50K/ast_cache")
    ap.add_argument("--save_dir", default=os.path.join(ROOT, "checkpoints", "audioset_ast"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5, help="主干学习率")
    ap.add_argument("--head_lr", type=float, default=1e-4, help="分类头学习率")
    ap.add_argument("--warmup", type=int, default=1, help="warmup 轮数")
    ap.add_argument("--train_layers", type=int, default=4, help="解冻的后 N 层 Transformer")
    ap.add_argument("--limit_train", type=int, default=0, help="只用前 N 条训练 (冒烟测试)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    tr_ds = MultiLabelCache(args.cache_dir, "train", args.limit_train)
    va_ds = MultiLabelCache(args.cache_dir, "val")
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=64, num_workers=2, pin_memory=True)
    va_truth = np.load(os.path.join(args.cache_dir, "labels_val.npy"))
    print("train=%d val=%d classes=%d" % (len(tr_ds), len(va_ds), NUM_CLASSES))

    model, trainable = build_model(args.train_layers, device)

    backbone = [p for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith("classifier")]
    head = [p for n, p in model.named_parameters()
            if p.requires_grad and n.startswith("classifier")]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.lr},
        {"params": head, "lr": args.head_lr},
    ], weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    total_steps = len(tr_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * len(tr_loader),
        num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best = 0.0
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        pbar = tqdm(tr_loader, desc="E%d train" % epoch, ncols=90)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(input_values=x).logits
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += loss.item() * y.size(0)
            seen += y.size(0)
            pbar.set_postfix(loss="%.4f" % (run_loss / max(seen, 1)))

        val_lwlrap, _ = evaluate_lwlrap(model, va_loader, va_truth, device)
        train_loss = run_loss / max(seen, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_lwlrap": val_lwlrap})
        if val_lwlrap > best:
            best = val_lwlrap
            torch.save({"state_dict": model.state_dict(), "val_lwlrap": val_lwlrap,
                        "epoch": epoch, "classes": EN_LIST,
                        "train_layers": args.train_layers},
                       os.path.join(args.save_dir, "best_model_ast.pt"))
        print("Epoch %d/%d | train_loss %.4f | val_lwlrap %.4f | best %.4f" % (
            epoch, args.epochs, train_loss, val_lwlrap, best))

    json.dump(history, open(os.path.join(args.save_dir, "train_history.json"), "w"), indent=1)
    print("训练完成 | 总耗时 %s | 最佳验证 lwlrap %.4f" % (
        time.strftime("%H:%M:%S", time.gmtime(time.time() - t0)), best))


if __name__ == "__main__":
    main()