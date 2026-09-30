# -*- coding: utf-8 -*-
"""FSD50K 数据集下载与解压（ModelScope 镜像, 多线程分块 + 断点续传）

背景:
    FSD50K 官方发布在 Zenodo, 但当前网络对 zenodo.org 整域返回 403,
    因此改用可达的 ModelScope 镜像 OmniData/FSD50K (CC BY 4.0)。
    该仓库把完整的 FSD50K 打包成单个 raw/FSD50K.tar.gz (约 24.7 GB)。

用法:
    python -m scripts.download_fsd50k                 # 下载 + 解压
    python -m scripts.download_fsd50k --out_dir D:/datasets/FSD50K
    python -m scripts.download_fsd50k --threads 12 --keep_archive

产物目录结构 (解压后):
    D:/datasets/FSD50K/FSD50K.dev_audio/*.wav
    D:/datasets/FSD50K/FSD50K.eval_audio/*.wav
    D:/datasets/FSD50K/FSD50K.ground_truth/*.csv
    D:/datasets/FSD50K/FSD50K.metadata/*.csv

解压分两步: tar.gz 里装的是官方发布的多个 zip(dev/eval 为 7-Zip 分卷 zip),
因此 expand_release_zips() 会再做一次解压, 并清理中间压缩包。
"""
import argparse
import json
import os
import subprocess
import threading
import time

import requests

DATASET = "OmniData/FSD50K"
REMOTE_PATH = "raw/FSD50K.tar.gz"
CHUNK = 8 * 1024 * 1024  # 每个分块 8MB
URL = ("https://www.modelscope.cn/api/v1/datasets/%s/repo"
       "?Revision=master&FilePath=%s" % (DATASET, REMOTE_PATH))
HEADERS = {"User-Agent": "Mozilla/5.0"}


def remote_size():
    """HEAD 拿到文件总大小"""
    r = requests.head(URL, timeout=60, headers=HEADERS, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def plan_chunks(total):
    n = (total + CHUNK - 1) // CHUNK
    return [(i * CHUNK, min((i + 1) * CHUNK, total) - 1) for i in range(n)]


def download(out_dir, threads=12):
    """多线程分块下载, 已完成的块记录在 .state.json 里, 支持断点续传"""
    os.makedirs(out_dir, exist_ok=True)
    tar_path = os.path.join(out_dir, "FSD50K.tar.gz")
    state_path = tar_path + ".state.json"

    total = remote_size()
    chunks = plan_chunks(total)
    done = set()
    if os.path.exists(tar_path) and os.path.exists(state_path):
        done = set(json.load(open(state_path))["done"])
    print("总大小 %.2f GB, 分块 %d 个, 已完成 %d 个" % (
        total / 1e9, len(chunks), len(done)))

    if not os.path.exists(tar_path):
        with open(tar_path, "wb") as f:      # 预分配, 供各线程定位写入
            f.truncate(total)

    lock = threading.Lock()
    counter = {"bytes": 0}
    t0 = time.time()

    def save_state():
        with lock:
            json.dump({"done": sorted(done)}, open(state_path, "w"))

    def worker(items):
        for idx, (a, b) in items:
            h = dict(HEADERS)
            h["Range"] = "bytes=%d-%d" % (a, b)
            for attempt in range(5):
                try:
                    r = requests.get(URL, timeout=120, headers=h, stream=True)
                    if r.status_code not in (200, 206):
                        raise RuntimeError("HTTP %d" % r.status_code)
                    with open(tar_path, "r+b") as f:
                        f.seek(a)
                        got = 0
                        for c in r.iter_content(256 * 1024):
                            f.write(c)
                            got += len(c)
                            with lock:
                                counter["bytes"] += len(c)
                    if got != b - a + 1:
                        raise RuntimeError("分块长度不足 %d != %d" % (got, b - a + 1))
                    with lock:
                        done.add(idx)
                    break
                except Exception as e:      # 网络抖动重试
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))

    pending = [(i, c) for i, c in enumerate(chunks) if i not in done]
    groups = [[] for _ in range(threads)]
    for k, item in enumerate(pending):      # 轮转分配, 各线程负载均衡
        groups[k % threads].append(item)

    ths = [threading.Thread(target=worker, args=(g,), daemon=True)
           for g in groups if g]
    for t in ths:
        t.start()

    reporter = threading.Thread(target=_report, args=(total, done, counter, lock, t0),
                                daemon=True)
    reporter.start()
    for t in ths:
        t.join()
    save_state()
    print("\n下载完成: %s (%.2fGB, 耗时 %.1f 分钟)" % (
        tar_path, os.path.getsize(tar_path) / 1e9, (time.time() - t0) / 60))
    return tar_path


def _report(total, done, counter, lock, t0):
    while len(done) < (total + CHUNK - 1) // CHUNK:
        time.sleep(10)
        with lock:
            n = counter["bytes"]
        dt = time.time() - t0
        print("  已下载 %.2fGB  %.1fMB/s  块 %d" % (n / 1e9, n / 1e6 / dt, len(done)),
              flush=True)


def extract(out_dir, tar_path):
    """解压 tar.gz: 优先 bsdtar(tar.exe), 失败回退 Python tarfile

    注意: 该 tar.gz 里装的是 FSD50K 官方发布的多个 zip(含分卷), 解压后还需
    调用 expand_release_zips() 做二次解压。
    """
    stage = find_stage_dir(out_dir)
    if stage or os.path.isdir(os.path.join(out_dir, "FSD50K.dev_audio")):
        print("已解压, 跳过:", stage or out_dir)
        return
    print("开始解压 %s -> %s ..." % (tar_path, out_dir))
    t0 = time.time()
    try:
        subprocess.run(["tar", "-xzf", tar_path, "-C", out_dir], check=True)
        print("解压完成(bsdtar), 耗时 %.1f 分钟" % ((time.time() - t0) / 60))
        return
    except Exception as e:
        print("bsdtar 解压失败(%s), 回退 Python tarfile" % e)
    import tarfile
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)
    print("解压完成(tarfile), 耗时 %.1f 分钟" % ((time.time() - t0) / 60))


def find_stage_dir(out_dir):
    """定位存放 FSD50K 官方 zip 的目录 (可能在 out_dir 或 out_dir/FSD50K)"""
    for d in (out_dir, os.path.join(out_dir, "FSD50K")):
        if os.path.isfile(os.path.join(d, "FSD50K.dev_audio.zip")):
            return d
    return None


def has_wavs(d):
    """目录里是否已有 wav (用于判断某个 split 是否已解压完成)"""
    return os.path.isdir(d) and any(f.endswith(".wav") for f in os.listdir(d))


class VolReader:
    """把一组分卷文件当成一个连续文件来随机读取"""

    def __init__(self, paths):
        self.paths = paths
        self.handles = [open(p, "rb") for p in paths]
        self.starts, acc = [], 0
        for p in paths:
            self.starts.append(acc)
            acc += os.path.getsize(p)
        self.size = acc

    def close(self):
        for h in self.handles:
            h.close()

    def read_at(self, offset, size):
        """读取 [offset, offset+size) —— 自动跨卷, 返回 bytes"""
        buf = bytearray()
        i = len(self.starts) - 1
        while i > 0 and offset < self.starts[i]:
            i -= 1
        while len(buf) < size and i < len(self.handles):
            pos = offset + len(buf) - self.starts[i]
            self.handles[i].seek(pos)
            buf += self.handles[i].read(size - len(buf))
            i += 1
        return bytes(buf)


def parse_central_dir(reader, cd_off, cd_size):
    """解析中央目录 -> [dict(name, method, csize, usize, disk, lho)]"""
    import struct
    cd = reader.read_at(cd_off, cd_size)
    out, off = [], 0
    while off < len(cd) - 4 and cd[off:off + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[off + 10:off + 12])[0]
        csize = struct.unpack("<L", cd[off + 20:off + 24])[0]
        usize = struct.unpack("<L", cd[off + 24:off + 28])[0]
        nlen = struct.unpack("<H", cd[off + 28:off + 30])[0]
        elen = struct.unpack("<H", cd[off + 30:off + 32])[0]
        clen = struct.unpack("<H", cd[off + 32:off + 34])[0]
        disk = struct.unpack("<H", cd[off + 34:off + 36])[0]
        lho = struct.unpack("<L", cd[off + 42:off + 46])[0]
        name = cd[off + 46:off + 46 + nlen].decode("utf-8", "replace")
        # Zip64 扩展字段: 4 字节字段为 0xFFFFFFFF 时真实值在这里
        extra = cd[off + 46 + nlen:off + 46 + nlen + elen]
        if 0xFFFFFFFF in (csize, usize, lho) or disk == 0xFFFF:
            eo = 0
            while eo < len(extra) - 4:
                hid, hsz = struct.unpack("<HH", extra[eo:eo + 4])
                if hid == 0x0001:
                    p = eo + 4
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", extra[p:p + 8])[0]
                        p += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", extra[p:p + 8])[0]
                        p += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", extra[p:p + 8])[0]
                        p += 8
                    if disk == 0xFFFF:
                        disk = struct.unpack("<L", extra[p:p + 4])[0]
                    break
                eo += 4 + hsz
        out.append({"name": name, "method": method, "csize": csize,
                    "usize": usize, "disk": disk, "lho": lho})
        off += 46 + nlen + elen + clen
    return out


def extract_split_zip(vol_paths, dest_dir, strip_prefix=""):
    """解压 7-Zip 分卷 zip (xxx.z01..zNN + xxx.zip)

    分卷 zip 里 local header 偏移是「磁盘相对」的(相对所在分卷起点), 标准 zipfile
    无法直接读, 因此自行解析中央目录 + 按 disk/offset 定位并流式解压。
    strip_prefix: 条目名里要去掉的顶层目录前缀(如 "FSD50K.dev_audio/")
    """
    import struct
    import zlib
    reader = VolReader(vol_paths)
    try:
        # EOCD 位于最后一个分卷末尾; 在尾部 64KB 内回找
        tail_off = max(0, reader.size - 65557)
        tail = reader.read_at(tail_off, reader.size - tail_off)
        e = tail.rfind(b"PK\x05\x06")
        z = tail.rfind(b"PK\x06\x06")
        if z >= 0:      # Zip64: 中央目录位置与大小在 Zip64 EOCD 里
            cd_size, cd_off = struct.unpack("<QQ", tail[z + 40:z + 56])
        else:
            cd_size, cd_off = struct.unpack("<LL", tail[e + 12:e + 20])
        last_start = reader.starts[-1]
        # EOCD 里的 cd_off 是磁盘相对的, 换算成全局偏移
        if cd_off + cd_size <= reader.size - last_start:
            cd_off += last_start
        entries = parse_central_dir(reader, cd_off, cd_size)
        print("  分卷 %s: %d 个条目" % (os.path.basename(vol_paths[-1])[:-4], len(entries)))
        os.makedirs(dest_dir, exist_ok=True)
        n_file = 0
        for it in entries:
            name = it["name"]
            if strip_prefix and name.startswith(strip_prefix):
                name = name[len(strip_prefix):]
            if not name:
                os.makedirs(dest_dir, exist_ok=True)
                continue
            if name.endswith("/"):
                os.makedirs(os.path.join(dest_dir, name), exist_ok=True)
                continue
            abs_off = reader.starts[it["disk"]] + it["lho"]
            hdr = reader.read_at(abs_off, 30)
            assert hdr[:4] == b"PK\x03\x04", "local header 定位失败: %s" % name
            nlen, elen = struct.unpack("<HH", hdr[26:30])
            data_off = abs_off + 30 + nlen + elen
            dst = os.path.join(dest_dir, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fo:
                if it["method"] == 0:               # 存储
                    left = it["csize"]
                    while left > 0:
                        blk = reader.read_at(data_off, min(left, 4 << 20))
                        fo.write(blk)
                        data_off += len(blk)
                        left -= len(blk)
                else:                               # deflate
                    dec = zlib.decompressobj(-15)
                    left = it["csize"]
                    while left > 0:
                        blk = reader.read_at(data_off, min(left, 4 << 20))
                        data_off += len(blk)
                        left -= len(blk)
                        fo.write(dec.decompress(blk))
                    fo.write(dec.flush())
            n_file += 1
            if n_file % 5000 == 0:
                print("    已解压 %d 个文件" % n_file, flush=True)
        print("  解压完成: %d 个文件 -> %s" % (n_file, dest_dir))
    finally:
        reader.close()


def expand_release_zips(out_dir):
    """把官方发布的 zip(含分卷) 二次解压成最终目录结构"""
    if has_wavs(os.path.join(out_dir, "FSD50K.dev_audio")):
        print("已是最终结构, 跳过二次解压")
        return
    stage = find_stage_dir(out_dir)
    if stage is None:
        raise RuntimeError("未找到 FSD50K 官方 zip, 无法二次解压")
    print("二次解压官方 zip <-", stage)

    # 小型单卷 zip 交给 bsdtar
    for fn in ["FSD50K.ground_truth.zip", "FSD50K.metadata.zip"]:
        p = os.path.join(stage, fn)
        if os.path.isfile(p) and not os.path.isdir(os.path.join(out_dir, fn[:-4])):
            subprocess.run(["tar", "-xf", p, "-C", out_dir], check=True)
            print("  解压", fn)

    # 分卷 zip: .z01..zNN + .zip (按序号拼接最后一个卷)
    for base in ["FSD50K.dev_audio", "FSD50K.eval_audio"]:
        if has_wavs(os.path.join(out_dir, base)):
            print("  已存在:", base)
            continue
        vols = sorted(f for f in os.listdir(stage)
                      if f.startswith(base + ".z") and f != base + ".zip")
        vols.append(base + ".zip")
        t0 = time.time()
        extract_split_zip([os.path.join(stage, v) for v in vols],
                          os.path.join(out_dir, base), strip_prefix=base + "/")
        print("  %s 耗时 %.1f 分钟" % (base, (time.time() - t0) / 60))

    # 清理官方压缩包与临时目录
    for f in os.listdir(stage):
        os.remove(os.path.join(stage, f))
    if os.path.abspath(stage) != os.path.abspath(out_dir):
        os.rmdir(stage)
    print("已清理官方压缩包:", stage)


def summarize(out_dir):
    for d in ["FSD50K.dev_audio", "FSD50K.eval_audio",
              "FSD50K.ground_truth", "FSD50K.metadata"]:
        p = os.path.join(out_dir, d)
        if not os.path.isdir(p):
            print("缺失:", p)
            continue
        if d.endswith("_audio"):
            n = sum(1 for f in os.listdir(p) if f.endswith(".wav"))
            print("%-26s %6d 个 wav" % (d, n))
        else:
            print("%-26s %s" % (d, ", ".join(sorted(os.listdir(p))[:8])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="D:/datasets/FSD50K")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--keep_archive", action="store_true",
                    help="解压后保留 tar.gz 压缩包")
    args = ap.parse_args()

    tar_path = os.path.join(args.out_dir, "FSD50K.tar.gz")
    if has_wavs(os.path.join(args.out_dir, "FSD50K.dev_audio")):
        print("数据集已存在:", args.out_dir)
    else:
        # 已经解压出官方 zip 时不再重复下载
        if find_stage_dir(args.out_dir) is None:
            if not os.path.isfile(tar_path):
                tar_path = download(args.out_dir, args.threads)
            extract(args.out_dir, tar_path)
        expand_release_zips(args.out_dir)
        if not args.keep_archive:
            for f in (tar_path, tar_path + ".state.json"):
                if os.path.exists(f):
                    os.remove(f)
                    print("已清理:", f)
    summarize(args.out_dir)


if __name__ == "__main__":
    main()