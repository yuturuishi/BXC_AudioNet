# BXC_AudioNet

> 基于 AST（Audio Spectrogram Transformer）在 FSD50K 上部分微调训练的 **AudioSet 527 类音频打标**系统（开集多标签，预训练权重 `MIT/ast-finetuned-audioset-10-10-0.4593`）。

| | |
|---|---|
| 官网 | https://www.yuturuishi.com |
| 微信 | yuturuishi |
| Gitee | https://gitee.com/yuturuishi/BXC_AudioNet |
| GitHub | https://github.com/yuturuishi/BXC_AudioNet |

---

## 目录结构

```
BXC_AudioNet/
├── train.py                训练入口（按 val lwlrap 保存最优权重）
├── tests.py                测试入口（FSD50K 上算 lwlrap / mAP / top-k F1）
├── requirements.txt        依赖清单
├── audionet/               库代码
│   ├── data/fsd50k_multilabel.py   数据集（特征缓存 Dataset）+ 批量推理
│   ├── utils/metrics.py            指标：lwlrap / mAP / top-k micro-F1
│   └── labels/audioset_classes.py  527 类英文名 + 中文译名
├── scripts/                工具脚本，统一 python -m scripts.<脚本名> 运行
│   ├── download_fsd50k.py          下载 + 解压 FSD50K（含分卷 zip 解析）
│   ├── prepare_fsd50k.py           FSD50K → AST 特征缓存（fp16）+ 多热标签
│   ├── export_onnx_audioset_ast.py .pt → .onnx（含特征提取器配置与标签表落地）
│   ├── quantize_audioset_ast.py    ONNX INT8 动态量化（仅 MatMul/Gemm）
│   └── gen_smoke_cache.py          生成合成特征缓存，供训练冒烟自测
├── api/audioset_ast_api.py 打标 API（端口 8072，CPU 推理）
├── test_audio/             试听样例
├── checkpoints/            权重、ONNX、标签表、报告（本地生成，不纳入版本管理）
├── venv/                   虚拟环境（不纳入版本管理）
├── LICENSE / .gitignore
└── README.md
```

根目录只保留两个入口：`train.py`（训练）、`tests.py`（测试），其余脚本统一放在 `scripts/`。

> `.gitignore` 已屏蔽全部模型文件与产物（`checkpoints/`、`datasets/`、`ast_cache/`、`_smoke_cache/`、
> `_smoke_ckpt/`、`venv/`，以及 `*.pt`、`*.pth`、`*.ckpt`、`*.bin`、`*.h5`、`*.safetensors`、
> `*.onnx`、`*.engine`、`*.plan`、`*.trt` 等），模型文件不会进入版本库。

---

## 环境安装

* **Python 版本**：3.10 ~ 3.12（本项目已在 Windows Python 3.12.3 上验证；torch 无 3.13 轮子）
* 国内镜像：`https://pypi.tuna.tsinghua.edu.cn/simple`

```bash
# 1) 创建并激活虚拟环境（Linux 用 source venv/bin/activate）
python -m venv venv
venv\Scripts\activate

# 2) 升级 pip + 安装「除 torch 外」的全部依赖（走国内镜像）
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) 单独强制安装 CUDA 版 PyTorch（RTX 30 / 40 系用 cu121；务必带 +cu121 强制替换 CPU 版）
python -m pip install --force-reinstall --no-deps torch==2.2.0+cu121 torchaudio==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ **PyTorch 必须装 CUDA 版（cu121）**：国内镜像默认发的是 CPU 版 torch（约 198MB，版本号也是 2.2.0，
> pip 会误判「已满足」而不换轮子），用它会退回 CPU 推理、训练极慢。torch / torchaudio 两个包必须走
> PyTorch 官方 CUDA 源、用带 `+cu121` 的精确版本号强制安装。
> 训练与评测也支持纯 CPU 跑（自动降级，只是慢）；API 部署默认就是 CPU 推理，无需 GPU。

```bash
# 安装后验证（确认是 CUDA 版）
venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出类似：2.2.0+cu121 True
```

---

## 数据准备

**FSD50K**（ModelScope 镜像 `OmniData/FSD50K`，CC BY 4.0，约 24.7GB）官方划分：dev train 训练 / dev val 验证 / eval 测试。

```bash
# 1) 下载 + 解压（多线程分块 + 断点续传，自动二次解压并清理压缩包）
python -m scripts.download_fsd50k --out_dir D:/datasets/FSD50K --threads 12

# 2) 生成 AST 特征缓存（训练与评测不读音频，只读该缓存）
python -m scripts.prepare_fsd50k --train_limit 12000 --val_limit 0 --eval_limit 0
```

| 项 | 说明 |
|---|---|
| 特征 | 16kHz 单声道 → 波形 tile 补齐 10 秒 → kaldi fbank（hanning，128 mel）→ 998 帧 pad 到 1024 → 归一化，fp16 落盘（每条约 0.25MB） |
| 标签 | `[N, 527]` uint8 多热向量 |
| 缓存规模 | train 按「稀有类优先」分层抽稀 36080 → 11772 条，val 4158 条，eval 10022 条 |
| 类别映射 | FSD50K 的 200 类 = AudioSet 527 类的子集，以 **MID** 为映射键（避开 `labels` 字段的逗号歧义），映射成功 **193/200 类**；未映射的 7 个是 AudioSet 中间节点（如 Human voice），其片段丢弃。可用 50260/51197 条（丢 1.8%），平均 2.73 标签/条，覆盖 **192/527** 类 |
| 产物路径 | `D:/datasets/FSD50K/ast_cache/`（`feats_*.npy` / `labels_*.npy` / `names_*.json` / `classes.json` / `mapping_report.json`） |

* 官方包内层是 **7-Zip 分卷 zip**（`z01..z05 + zip`，标准 `zipfile` 读不了），`download_fsd50k.py` 自行解析中央目录后流式解压。
* `--train_limit` / `--val_limit` / `--eval_limit` 传 0 表示全量；换路径时用 `--cache_dir` 指定（各脚本均支持）。

---

## 模型训练

部分微调：冻结 patch embedding 与前 8 层 Transformer，只训练后 4 层 + layernorm + 分类头（总参数 86.6M，可训练 28.8M）。

```bash
# 默认输出 <项目根>/checkpoints/audioset_ast/
python train.py --epochs 10 --batch_size 32 --train_layers 4
```

* 优化：AdamW（主干 1e-5 / 分类头 1e-4）+ cosine 退火 + 1 epoch warmup
* 显存：AMP fp16 + gradient checkpointing，RTX 3080Ti Laptop 约 5.6GB 可跑 batch 32
* 损失：`BCEWithLogitsLoss`；指标：lwlrap / mAP（AudioSet 官方评测协议）

训练产物：

```
checkpoints/audioset_ast/
├── best_model_ast.pt      # val lwlrap 最优权重
└── train_history.json     # 逐轮 train_loss / val_lwlrap
```

### 环境与训练冒烟测试（可选，不下载数据集）

```bash
python -m scripts.gen_smoke_cache                    # 生成 _smoke_cache（几条合成特征）
python train.py --epochs 1 --limit_train 8 --num_workers 0 \
    --cache_dir _smoke_cache --save_dir _smoke_ckpt
# 期望：正常打印 Epoch 1/1 | train_loss ... | val_lwlrap ... 并产出 _smoke_ckpt/best_model_ast.pt
```

> `_smoke_cache` / `_smoke_ckpt` 仅用于自测，非真实数据，可随时删除（已列入 `.gitignore`）。

### 导出推理模型

```bash
# 训练权重 .pt -> ONNX（自动校验 torch / onnxruntime 一致性，最大偏差 < 1e-3）
python -m scripts.export_onnx_audioset_ast
```

导出同时把特征提取器配置 `preprocessor/` 与标签表 `labels.json` 落地到 `checkpoints/audioset_ast/`，供 API 离线加载。

```bash
# 可选：INT8 动态量化（仅 MatMul/Gemm，Conv 保持 fp32 以免 CPU EP 报 NOT_IMPLEMENTED）
python -m scripts.quantize_audioset_ast
# fp32 346.8MB -> int8 91.0MB，输出 audioset_ast_int8.onnx
```

---

## 测试

```bash
python tests.py                 # 默认跑 eval 集
python tests.py --split both    # 同时看 dev val
```

产出 `checkpoints/audioset_ast/test_report.md`（人读）与 `test_report.json`（机器可读，含单类 AP）。

### 测试结果

| 数据集 | 样本数 | lwlrap | mAP | top-3 micro-F1 |
|---|---|---|---|---|
| dev val | 4158 | 0.7986 | 0.6118 | 0.5988 |
| **eval** | **10022** | **0.7380** | 0.6096 | 0.5536 |

> eval 比 val 低约 6 个点是预期现象：FSD50K 官方把 eval 作为独立测试集，分布与 dev 有差异（官方论文同样存在此 gap），并非过拟合。

**端到端验证**：ONNX 与 torch 输出最大偏差 1.14e-05；API 实测 7 条真实音频语义全部命中（狗叫→狗 0.993、公鸡→鸡 0.987、雷暴→雷 0.983、婴儿哭→哭泣 0.880、键盘→打字 0.840、警笛→警笛 0.727），含 mp3 链路。

---

## API 部署

拷贝以下文件到服务器，保持相对目录结构不变：

```
api/audioset_ast_api.py                             API 服务入口
audionet/labels/audioset_classes.py                 527 类中英文标签表
checkpoints/audioset_ast/audioset_ast.onnx          模型（约 346MB）
checkpoints/audioset_ast/labels.json                527 类英文标签（对应 logits 索引）
checkpoints/audioset_ast/preprocessor/              特征提取器配置（离线加载）
```

服务器依赖（仅部署 API 所需，Python 3.10~3.12；**torch 不需要**）：

```bash
pip install Flask==3.0.3 transformers==4.39.3 onnxruntime==1.19.0 soundfile==0.13.1 scipy==1.12.0 numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

启动（在拷贝目录根下执行）：`python -m api.audioset_ast_api`，默认端口 8072，CPU 单条约 0.8 秒。

**接口**：`POST /audio/tag`，multipart 表单 —— `file`（wav/mp3/flac/ogg）+ `topk`（默认 8）+ `threshold`（默认 0.1）

```json
{
  "code": 1000, "msg": "success",
  "result": {
    "count": 3, "duration": 10.0,
    "tags": [
      {"rank": 1, "label_en": "Speech", "label_cn": "语音",   "prob": 0.9421},
      {"rank": 2, "label_en": "Music",  "label_cn": "音乐",   "prob": 0.7130},
      {"rank": 3, "label_en": "Dog",    "label_cn": "狗叫声", "prob": 0.2286}
    ]
  }
}
```

```html
<input type="file" id="audioFile" accept="audio/*">
<div id="result"></div>
<script>
document.getElementById('audioFile').onchange = async () => {
  const fd = new FormData();
  fd.append('file', document.getElementById('audioFile').files[0]);
  fd.append('topk', '8');
  const j = await (await fetch('http://<服务器地址>:8072/audio/tag', {method: 'POST', body: fd})).json();
  document.getElementById('result').innerHTML = j.code === 1000
    ? j.result.tags.map(t => t.label_cn + '（' + t.label_en + '）' + (t.prob * 100).toFixed(1) + '%').join('<br>')
    : '识别失败: ' + j.msg;
};
</script>
```

---

## 训练结果

10 epoch（RTX 3080Ti Laptop，显存约 5.6GB，总耗时约 1 小时 10 分）：

| epoch | 1 | 2 | 3 | 5 | 7 | 10 |
|---|---|---|---|---|---|---|
| train_loss | 0.0135 | 0.0087 | 0.0074 | 0.0064 | 0.0060 | 0.0058 |
| val_lwlrap | 0.7389 | 0.7771 | 0.7886 | 0.7949 | 0.7980 | **0.7986** |

val_lwlrap 单调上升、未见过拟合拐点，最佳权重取第 10 轮（0.7986）。

**单类 AP 差异极大**：长时、音色鲜明的类很好（雷暴 0.970 / 雷 0.967 / 冲马桶 0.952 / 猫呼噜 0.948 / 乐器 0.921 / 掌声 0.918）；瞬态短促类基本不可用（滴答 0.055 / 轻敲 0.068 / 闷响 0.113 / 木头 0.128）。完整单类 AP 表见 `checkpoints/audioset_ast/test_report.md`。

---

## 能力边界

* **真实监督只覆盖 192/527 类**，其余类是预训练原始能力，精度未验证，不可当作已训练类使用。
* **瞬态短促类不要用**（滴答/轻敲/闷响/木头），长时声音才可靠；选类前先查测试报告里的单类 AP 表。
* **无时间定位、无声源方向、不做音频分离、不识别说话人**，只回答「有没有这类声音」。
* 长音频需自行滑窗（建议步长 1~5 秒）后合并；审核场景建议按类标定阈值并采用「连续 N 窗命中」抑制误报。

---

## 数据与产物位置

| 用途 | 路径 |
|---|---|
| 数据集 | `D:/datasets/FSD50K/`（dev_audio / eval_audio / ground_truth / metadata） |
| 特征缓存 | `D:/datasets/FSD50K/ast_cache/` |
| 权重与产物 | `checkpoints/audioset_ast/`（权重 / ONNX / labels.json / preprocessor/ / 训练与测试报告） |

---

## 预处理说明

| 环节 | 处理 |
|---|---|
| 音频 | 16kHz 单声道；不足 10 秒按 AST 官方做法重复填充（非补零），超过则截断 |
| 特征 | kaldi fbank（hanning 窗，128 mel）→ 时间维 pad 到 1024 帧 |
| 归一化 | `(x + 4.2677393) / (4.5689974 × 2)` |
| 输出 | 527 类 sigmoid 概率（多标签共存，非 softmax 单选） |

---

## 开源协议

本项目以 **MIT 协议** 开源，详见仓库根目录 `LICENSE` 文件（Copyright 2022 Vanishi）。
