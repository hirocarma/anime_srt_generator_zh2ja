#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from threading import Lock

import torch
import webrtcvad
from opencc import OpenCC
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# -----------------------
# Configurable defaults
# -----------------------
WENET_MODEL = "wenetspeech"
DEFAULT_NLLB_MODEL = "facebook/nllb-200-3.3B"

TRANS_TMPDIR = str(Path.home() / "tmp" / "anime_trans_tmp")
CACHE_TRANS_DB_PATH = TRANS_TMPDIR + "/" + "translation_cache.db"
CACHE_WENET_DB_PATH = TRANS_TMPDIR + "/" + "wenet_cache.db"

USE_TRANS_CACHE = 1
USE_ASR_CACHE = 1

WENET_DAEMON_HOST = "127.0.0.1"
WENET_DAEMON_PORT = 9000
WENET_DAEMON_SCRIPT = os.path.dirname(__file__) + "/" + "wenet_daemon.py"
WENET_DAEMON_URL = (
    "http://" + WENET_DAEMON_HOST + ":" + str(WENET_DAEMON_PORT) + "/transcribe"
)

USE_RAMDISK = 1
RAMDISK_MOUNTPOINT = "/mnt/ramdisk"
RAMDISK_SIZE = "6G"
WENET_MODEL_SRC = "/home/hiro/.wenet" + "/" + WENET_MODEL
RAMDISK_MODEL_DIR = RAMDISK_MOUNTPOINT + "/" + WENET_MODEL

PRINT_DEBUG = 1
# -----------------------
# Start
# -----------------------
start_time = time.time()
start_time_f = time.strftime("%Y/%m/%d %H:%M:%S")

# -----------------------
# Global translator setup
# -----------------------
print("[Init] Loading global NLLB translator...")

if torch.cuda.is_available():
    _global_device = torch.device("cuda")
    try:
        print(f"[Init] Using GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        print("[Init] Using GPU")
else:
    _global_device = torch.device("cpu")
    print("[Init] Using CPU (Slow).")

_global_tokenizer = AutoTokenizer.from_pretrained(
    DEFAULT_NLLB_MODEL, src_lang="zho_Hans", tgt_lang="jpn_Jpan"
)
_global_model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_NLLB_MODEL)

# try to compile if available (optional)
try:
    _global_model = torch.compile(_global_model)
    print("[INFO] torch.compile applied")
except Exception:
    pass

if _global_device.type == "cuda":
    try:
        _global_model.half()
    except Exception:
        pass

_global_model.to(_global_device)
_global_model.eval()

print("[Init] NLLB translator ready.")

# Load OpenCC(global)
CC_T2S = OpenCC("t2s")


# --------------------------------------------------
# Logger
# --------------------------------------------------


class ElapsedTimeFormatter(logging.Formatter):
    def format(self, record):
        elapsed_sec = time.time() - start_time
        minutes = int(elapsed_sec // 60)
        seconds = elapsed_sec % 60
        record.min_sec_elapsed = f"({minutes:02}:{seconds:02.0f})"
        return super().format(record)


def setup_logger(log_file="anime_srt.log"):
    logger = logging.getLogger("EnhancedTimerLogger")
    logger.setLevel(logging.INFO)

    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    file_formatter = ElapsedTimeFormatter(
        "%(asctime)s - %(filename)s - [elapsed:%(relativeCreated)dms] %(min_sec_elapsed)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    screen_formatter = file_formatter

    fh = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(screen_formatter)
    logger.addHandler(sh)

    return logger


# -----------------------
# Utilities
# -----------------------


def run_cmd(cmd, capture_output=False, check=True):
    if capture_output:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            close_fds=True,
        )
        if check and p.returncode != 0:
            raise subprocess.CalledProcessError(
                p.returncode, cmd, output=p.stdout, stderr=p.stderr
            )
        return p.stdout.strip(), p.stderr.strip()
    else:
        subprocess.run(cmd, check=check, close_fds=True)


def choose_beam_for_text(zh_text):
    def beam_for_one(s: str):
        L = max(1, len(s))
        if L <= 6:
            return 2
        if L <= 30:
            return 4
        return 6

    if isinstance(zh_text, list):
        return max(beam_for_one(s) for s in zh_text)
    return beam_for_one(zh_text)


def sec_to_srt(ts_seconds: int) -> str:
    s = int(ts_seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02}:{m:02}:{sec:02},000"


CONVERSION_MAP = {
    "修道士": "師匠",
    "マスター": "師匠",
    "太空系": "空間系",
    "小さな白": "小白",
    "山心": "山新",
    "接收器": "レシーバー",
    "登陸": "ログイン",
    "実行者": "執行者",
}


def replace_strings_in_array(input_array: list[str]) -> list[str]:
    output_array = []

    for original_string in input_array:
        current_string = original_string

        for old_word, new_word in CONVERSION_MAP.items():
            current_string = current_string.replace(old_word, new_word)
        output_array.append(current_string)

    return output_array


def split_two_lines(text: str, max_chars: int = 28) -> str:
    text = text.strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    m = re.split(r"([。！？])", text, maxsplit=1)
    if len(m) >= 3:
        first = (m[0] + m[1]).strip()
        rest = "".join(m[2:]).strip()
        if len(first) <= max_chars:
            return (first + "\n" + rest) if rest else first
    mid = len(text) // 2
    for i in range(0, 20):
        for j in (mid + i, mid - i):
            if j <= 0 or j >= len(text):
                continue
            if text[j] in "，,。！？ 、 ":
                return text[:j].strip() + "\n" + text[j:].strip()
    return text[:mid].strip() + "\n" + text[mid:].strip()


# --------------------------------------------------
# Utility: Chinese
# --------------------------------------------------
PUNC_MAP = {"，": ",", "。": ".", "！": "!", "？": "?", "：": ":", "；": ";"}


def clean_chinese_punctuation(txt: str) -> str:
    if not txt:
        return ""
    t = CC_T2S.convert(txt)
    for k, v in PUNC_MAP.items():
        t = t.replace(k, v)
    t = t.replace("｡", ".")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace(". ", ".").replace("! ", "!").replace("? ", "?")
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    return t.strip()


SHORT_ZH_MAP = {
    "嗯": "うん",
    "啊": "ああ",
    "哦": "おお",
    "欸": "ええと",
    "喂": "もしもし",
}


def fix_short_zh(zh):
    if isinstance(zh, list):
        return [SHORT_ZH_MAP.get(item, item) for item in zh]
    return SHORT_ZH_MAP.get(zh, zh)


# --------------------------------------------------
# Ramdisk
# --------------------------------------------------


def mount_ramdisk(mount_point="/mnt/ramdisk", size="6G"):
    try:
        result = subprocess.run(
            ["mount"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for line in result.stdout.splitlines():
            if mount_point in line:
                return True  # Already mounted
    except Exception:
        return False

    try:
        if not os.path.exists(mount_point):
            os.makedirs(mount_point)
    except Exception:
        return False

    try:
        cmd = [
            "sudo",
            "mount",
            "-t",
            "tmpfs",
            "-o",
            f"size={size}",
            "tmpfs",
            mount_point,
        ]
        # subprocess.run(cmd, check=True)
        run_cmd(cmd, check=True)
        return True
    except Exception:
        return False


def ensure_wenet_model_in_ramdisk(src_model_dir: str, ramdisk_dir):

    src = Path(src_model_dir)
    dst = Path(ramdisk_dir)

    if not src.exists():
        raise FileNotFoundError(f"Source model not found: {src}")

    if dst.exists():
        print(f"[INFO] RAM disk model already exists: {dst}")
        return dst

    print(f"[info] Copying WeNet model to RAM disk:\n  {src} → {dst}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(src, dst)

    print(f"[INFO] Model copied to RAM disk: {dst}")
    return dst


# --------------------------------------------------
# Audio duration
# --------------------------------------------------


def extract_wav(input_mp4: Path, out_wav: Path):
    print(f"[INFO] Extracting audio: {input_mp4} -> {out_wav}")
    if os.path.isfile(out_wav):
        print("[INFO] Use an existing wav file.")
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_mp4),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "highpass=f=150, lowpass=f=7500, dynaudnorm, loudnorm=I=-20:TP=-2:LRA=7, silenceremove=1:0:-50dB",
        str(out_wav),
    ]
    run_cmd(cmd, capture_output=True)


def split_wav(wav_path: Path, out_dir: Path):

    # --- base limits ---
    MIN_LEN = 0.02
    PAD_SEC = 0.2  # context padding

    # --- hysteresis ---
    MIN_SPEECH_FRAMES = 2
    MIN_SILENCE_FRAMES = 6

    print(
        "[INFO][split_wav] Using WebRTC VAD with hysteresis + adaptive MAX_LEN + padding"
    )

    with wave.open(str(wav_path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        total_frames_wav = wf.getnframes()
        frames = wf.readframes(total_frames_wav)

    wav_duration = total_frames_wav / sample_rate

    if sample_width != 2:
        raise ValueError("WAV must be 16-bit PCM for VAD.")
    if num_channels != 1:
        raise ValueError("WAV must be mono for VAD.")
    if sample_rate not in (8000, 16000, 32000, 48000):
        raise ValueError("WAV sample_rate must be 8000/16000/32000/48000.")

    vad = webrtcvad.Vad(3)

    # -----------------------------
    # Frame split
    # -----------------------------
    frame_dur = 30  # ms
    frame_sec = frame_dur / 1000.0
    bytes_per_frame = int(sample_rate * frame_sec * 2)

    speech_flags = []

    for i in range(0, len(frames), bytes_per_frame):
        frame = frames[i : i + bytes_per_frame]
        if len(frame) < bytes_per_frame:
            break
        speech_flags.append(1 if vad.is_speech(frame, sample_rate) else 0)

    # -------------------------------------------------
    # Adaptive MAX_LEN
    # -------------------------------------------------
    total_frames = len(speech_flags)
    speech_frames = sum(speech_flags)
    speech_ratio = speech_frames / max(1, total_frames)

    if speech_ratio > 0.75:
        MAX_LEN = 5.0
    elif speech_ratio > 0.45:
        MAX_LEN = 7.0
    else:
        MAX_LEN = 9.0

    if PRINT_DEBUG:
        print(f"[VAD] speech_ratio={speech_ratio:.2f} → MAX_LEN={MAX_LEN:.1f}s")

    # -------------------------------------------------
    # Hysteresis smoothing
    # -------------------------------------------------
    segments = []
    in_speech = False
    speech_start = 0.0
    speech_cnt = 0
    silence_cnt = 0

    for idx, flag in enumerate(speech_flags):
        t = idx * frame_sec

        if flag:
            speech_cnt += 1
            silence_cnt = 0
        else:
            silence_cnt += 1
            speech_cnt = 0

        if not in_speech:
            if speech_cnt >= MIN_SPEECH_FRAMES:
                in_speech = True
                speech_start = t - (MIN_SPEECH_FRAMES - 1) * frame_sec
        else:
            if silence_cnt >= MIN_SILENCE_FRAMES:
                speech_end = t - (MIN_SILENCE_FRAMES - 1) * frame_sec
                if speech_end - speech_start >= MIN_LEN:
                    segments.append((speech_start, speech_end))
                in_speech = False

    if in_speech:
        end_t = total_frames * frame_sec
        if end_t - speech_start >= MIN_LEN:
            segments.append((speech_start, end_t))

    # -------------------------------------------------
    # Split by adaptive MAX_LEN
    # -------------------------------------------------
    final_segments = []

    for st, ed in segments:
        dur = ed - st
        if dur <= MAX_LEN:
            final_segments.append((st, ed))
            continue

        n = int(dur / MAX_LEN) + 1
        for i in range(n):
            s = st + i * MAX_LEN
            e = min(ed, s + MAX_LEN)
            if e - s >= MIN_LEN:
                final_segments.append((s, e))

    # -------------------------------------------------
    # Context padding (±PAD_SEC)
    # -------------------------------------------------
    padded_segments = []

    for st, ed in final_segments:
        pst = max(0.0, st - PAD_SEC)
        ped = min(wav_duration, ed + PAD_SEC)

        if ped - pst >= MIN_LEN:
            padded_segments.append((pst, ped))

    padded_segments.sort(key=lambda x: x[0])

    # -------------------------------------------------
    # Export wav
    # -------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_info_list = []
    for i, (st, ed) in enumerate(padded_segments):
        if ed <= st:
            continue

        out = out_dir / f"{i:04d}.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{st:.3f}",
            "-to",
            f"{ed:.3f}",
            "-i",
            str(wav_path),
            "-c",
            "copy",
            str(out),
        ]

        try:
            run_cmd(cmd, capture_output=True)
            seg_info_list.append((st, ed, out))
            if PRINT_DEBUG:
                print(f"  → segment {i:04d}: {st:.2f} - {ed-st:.2f}s")
        except Exception as e:
            print(f"[WARN] ffmpeg failed for segment {i}: {e}")

    print(f"[INFO] Created {len(seg_info_list)} segments at: {out_dir}")
    return seg_info_list


# -------------------------------------------------------
# Wenet daemon
# -------------------------------------------------------


def _check_daemon_alive(host=WENET_DAEMON_HOST, port=WENET_DAEMON_PORT, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def ensure_wenet_daemon_running(wenet_model_name):
    if _check_daemon_alive():
        print("[INFO] WeNet daemon is already running.")
        return True

    print("[INFO] WeNet daemon not running — starting...")

    daemon_path = Path(WENET_DAEMON_SCRIPT)
    if not daemon_path.exists():
        raise FileNotFoundError(f"daemon script not found: {daemon_path}")

    cmd = [
        "python3",
        str(daemon_path),
        "--model",
        wenet_model_name,
    ]

    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )

    time.sleep(5)

    if _check_daemon_alive():
        print("[INFO] WeNet daemon started successfully.")
        return True

    print("[ERROR] WeNet daemon failed to start.")
    return False


# -------------------------------------------------------
# SQLite-based persistent translation cache
# -------------------------------------------------------
_cache_lock = Lock()
_conn = sqlite3.connect(CACHE_TRANS_DB_PATH, check_same_thread=False)
_cursor = _conn.cursor()

# Create table if not exists
_cursor.execute(
    """
CREATE TABLE IF NOT EXISTS trans_cache (
    zh TEXT PRIMARY KEY,
    ja TEXT
)
"""
)
_conn.commit()


def cache_get(zh: str):
    """Return cached translation or None."""
    with _cache_lock:
        row = _cursor.execute(
            "SELECT ja FROM trans_cache WHERE zh = ?", (zh,)
        ).fetchone()
        if row:
            return row[0]
    return None


def cache_set(zh: str, ja: str):
    """Save translation into SQLite."""
    with _cache_lock:
        _cursor.execute(
            "INSERT OR REPLACE INTO trans_cache (zh, ja) VALUES (?, ?)",
            (zh, ja),
        )
        _conn.commit()


# -----------------------
# ASR Cache class (improved keying: content SHA1)
# -----------------------
class ASRCache:
    def __init__(self, db_path=CACHE_WENET_DB_PATH):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        # ensure dir exists
        dbdir = os.path.dirname(os.path.abspath(self.db_path))
        if dbdir and not os.path.exists(dbdir):
            try:
                os.makedirs(dbdir, exist_ok=True)
            except Exception:
                pass
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS asr_cache (
                    key TEXT PRIMARY KEY,
                    text TEXT NOT NULL
                )
            """
            )
            conn.commit()

    def _sha1_of_file(self, wav_path):
        """Compute SHA1 hash of file contents (streamed)."""
        try:
            h = hashlib.sha1()
            with open(wav_path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    h.update(chunk)
            return h.hexdigest()
        except FileNotFoundError:
            return None
        except Exception:
            # fallback to stat-based key if read fails (very unlikely)
            try:
                stat = os.stat(wav_path)
                raw = f"{wav_path}:{stat.st_size}:{stat.st_mtime}"
                return hashlib.sha1(raw.encode("utf-8")).hexdigest()
            except Exception:
                return None

    def make_key(self, wav_path):
        """Make a robust key based on file contents (SHA1)."""
        # accept Path or str
        if isinstance(wav_path, Path):
            p = str(wav_path)
        else:
            p = wav_path
        key = self._sha1_of_file(p)
        return key

    def get(self, wav_path):
        key = self.make_key(wav_path)
        if key is None:
            return None
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT text FROM asr_cache WHERE key = ?", (key,))
            row = cur.fetchone()
            if row:
                if PRINT_DEBUG:
                    print(f"[ASR Cache] HIT for {key}")
                return row[0]
        # miss
        return None

    def set(self, wav_path, text):
        key = self.make_key(wav_path)
        if key is None:
            return
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "REPLACE INTO asr_cache (key, text) VALUES (?, ?)", (key, text)
            )
            conn.commit()
        if PRINT_DEBUG:
            print(f"[ASR Cache] STORED for {key}")


# ------------------------------------------------------------------
# Translation
# ------------------------------------------------------------------
def _translate_batch_core(texts, beam=5):
    if not texts:
        return []

    enc = _global_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    enc = {k: v.to(_global_device) for k, v in enc.items()}

    with torch.no_grad():
        gen_out = _global_model.generate(
            **enc,
            forced_bos_token_id=_global_tokenizer.convert_tokens_to_ids("jpn_Jpan"),
            max_new_tokens=128,
            num_beams=beam,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
        )

    results = _global_tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    results = [r.strip() for r in results]

    results_r = replace_strings_in_array(results)

    if PRINT_DEBUG:
        print("translate results:")
        for i, r in enumerate(results_r):
            print(f"  [{i}] {r}")

    return results_r


def translate_batch_to_ja(texts, beam=4):
    if not texts:
        return []

    results = [None] * len(texts)
    missing_indices = []
    missing_texts = []

    if USE_TRANS_CACHE:
        for i, t in enumerate(texts):
            ja = cache_get(t)
            if ja:
                results[i] = ja
            else:
                missing_indices.append(i)
                missing_texts.append(t)
    else:
        if PRINT_DEBUG:
            print("[TRANS] Cache disabled → force new translation")
        for i, t in enumerate(texts):
            missing_indices.append(i)
            missing_texts.append(t)

    if USE_TRANS_CACHE and not missing_texts:
        if PRINT_DEBUG:
            print("[TRANS] Cache HIT.")
            print("translate results(cache):")
            for i, r in enumerate(results):
                print(f"  [{i}] {r}")
        return results

    if PRINT_DEBUG and USE_TRANS_CACHE:
        print("[TRANS] Cache not HIT → new translation")

    # --- translate missing ones ---
    new_translations = _translate_batch_core(missing_texts, beam=beam)

    for idx, zh in zip(missing_indices, missing_texts):
        ja = new_translations.pop(0)
        results[idx] = ja
        if USE_TRANS_CACHE:
            cache_set(zh, ja)

    return results


# ------------------------------------------------------------------
# WeNet daemon wrapper
# ------------------------------------------------------------------


class WenetDaemonError(Exception):
    pass


def wenet_daemon_transcribe(wav_path: Path, timeout: int = 300) -> str:
    if not wav_path.exists():
        raise WenetDaemonError(f"wav not found: {wav_path}")

    data = json.dumps(
        {
            "path": str(wav_path.resolve()),
            "timeout": timeout,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        WENET_DAEMON_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout + 5) as res:
            body = res.read().decode("utf-8")
    except Exception as e:
        raise WenetDaemonError(f"HTTP error: {e}")

    try:
        obj = json.loads(body)
    except Exception as e:
        raise WenetDaemonError(f"Invalid JSON from daemon: {e}, body={body}")

    if not obj.get("ok"):
        raise WenetDaemonError(f"Daemon error rc={obj.get('rc')}: {obj.get('stderr')}")

    return obj.get("text", "").strip()


@lru_cache(maxsize=4096)
def wenet_daemon_transcribe_cached(path_str: str) -> str:
    return wenet_daemon_transcribe(Path(path_str))


def run_wenet_asr_with_cache_daemon(seg_path: Path) -> str:
    return wenet_daemon_transcribe_cached(str(seg_path.resolve()))

# instantiate ASR cache (file in cwd by default)
asr_cache = ASRCache()


def run_wenet_asr_with_cache(wav_path: str):
    cached = asr_cache.get(wav_path)
    if cached is not None:
        return cached

    text = run_wenet_asr_with_cache_daemon(wav_path)

    asr_cache.set(wav_path, text)

    return text


# ------------------------------------------------------------------
# build SRT
# ------------------------------------------------------------------
def build_srt_blocks(filtered, text_lines_list, args):
    blocks = []

    for idx, rec in enumerate(filtered):
        start = sec_to_srt(int(rec["start"]))
        end = sec_to_srt(int(rec["end"]))

        lines = text_lines_list[idx]
        if not lines:
            continue

        body = "\n".join(
            split_two_lines(line, max_chars=args.max_chars) for line in lines
        )

        block = f"{idx+1}\n{start} --> {end}\n{body}\n\n"
        blocks.append(block)

    return blocks


# -----------------------
# Main pipeline
# -----------------------
def prepare_input(args, logger):
    inp = Path(args.input)
    if not inp.exists():
        print("Input file not found:", inp)
        sys.exit(1)

    tmp = Path(args.tmpdir)
    tmp.mkdir(parents=True, exist_ok=True)

    logger.info(f"{args.input} start at {start_time_f}")
    logger.info("Start pipeline")

    return inp, tmp


def prepare_wav(inp: Path, tmp: Path):
    wav = tmp / (inp.stem + ".wav")
    extract_wav(inp, wav)
    return wav


def prepare_segments(wav: Path, tmp: Path, stem: str):
    seg_dir = tmp / "segments" / stem
    if seg_dir.exists():
        for f in seg_dir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass

    return split_wav(wav, seg_dir)


def setup_wenet_environment():
    if USE_RAMDISK:
        if not mount_ramdisk(RAMDISK_MOUNTPOINT, RAMDISK_SIZE):
            raise RuntimeError("Ramdisk mount failed")

        wenet_model_name = ensure_wenet_model_in_ramdisk(
            WENET_MODEL_SRC, RAMDISK_MODEL_DIR
        )
        print("[INFO] Using ramdisk")
    else:
        wenet_model_name = WENET_MODEL

    if not ensure_wenet_daemon_running(wenet_model_name):
        raise RuntimeError("WeNet daemon could not be started")
    print("[INFO] Using WeNet daemon...")

    return wenet_model_name


def run_asr_on_segments(segs, wenet_model_name, args, logger):
    logger.info("Start running WeNet ASR on segments...")

    asr_results = [None] * len(segs)
    workers = max(1, min(args.workers, max(1, os.cpu_count() - 1)))
    print(f"  Using {workers} parallel workers for WeNet CLI")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}

        for i, (st, ed, seg_path) in enumerate(segs):
            if USE_ASR_CACHE:
                fut = ex.submit(run_wenet_asr_with_cache, seg_path)
            else:
                fut = ex.submit(wenet_daemon_transcribe, seg_path)

            futures[fut] = (i, st, ed)

        for fut in tqdm(as_completed(futures), total=len(segs), desc="ASR segments"):
            i, st, ed = futures[fut]
            try:
                text = fut.result() or ""
            except Exception as e:
                print(f"Error on segment {i}: {e}")
                text = ""

            text = clean_chinese_punctuation(text)

            asr_results[i] = {
                "start": st,
                "end": ed,
                "zh": text,
            }

    filtered = [r for r in asr_results if r and r["zh"].strip()]
    if not filtered:
        print("No recognized speech found.")
        sys.exit(0)

    return filtered


def translate_zh_to_ja(filtered, args, logger):
    zh_lines = [r["zh"] for r in filtered]
    logger.info(f"Start translating {len(zh_lines)} Chinese segments -> Japanese")

    ja_lines = []
    BATCH = args.batch
    logger.info(f"batch size: {BATCH}")

    for i in tqdm(range(0, len(zh_lines), BATCH), desc="Translating"):
        batch = zh_lines[i : i + BATCH]
        batch_fix = fix_short_zh(batch)
        beam_value = choose_beam_for_text(batch_fix)

        ja_batch = translate_batch_to_ja(
            batch_fix,
            beam=beam_value,
        )
        ja_lines.extend(ja_batch)

    return ja_lines


def write_srt_files(filtered, ja_lines, args, inp: Path, logger):
    logger.info("Building SRT blocks...")

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = inp.stem + ".srt"
        if os.path.isfile(out_path):
            now = datetime.datetime.now()
            dt = now.strftime("_%y-%m-%d_%H-%M-%S")
            out_path = inp.stem + dt + ".srt"

    # --------------------------------------------------
    # Chinese + Japanese SRT
    # --------------------------------------------------
    cj_lines = []
    for idx, rec in enumerate(filtered):
        zh = rec["zh"]
        ja = ja_lines[idx] if idx < len(ja_lines) else ""
        cj_lines.append([zh, ja])

    srt_blocks = build_srt_blocks(filtered, cj_lines, args)

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(srt_blocks)

    print(f"[INFO] Chinese+Japanese SRT written to: {out_path}")

    # --------------------------------------------------
    # Chinese only SRT
    # --------------------------------------------------
    zh_srt_path = Path(str(out_path).replace(".srt", "_zh.srt"))
    zh_lines = [[rec["zh"]] for rec in filtered]

    zh_blocks = build_srt_blocks(filtered, zh_lines, args)

    with open(zh_srt_path, "w", encoding="utf-8") as f:
        f.writelines(zh_blocks)

    print(f"[INFO] Chinese SRT written to: {zh_srt_path}")

    # --------------------------------------------------
    # Japanese only SRT
    # --------------------------------------------------
    ja_srt_path = Path(str(out_path).replace(".srt", "_ja.srt"))
    ja_only_lines = [[ja_lines[idx]] for idx in range(len(filtered))]

    ja_blocks = build_srt_blocks(filtered, ja_only_lines, args)

    with open(ja_srt_path, "w", encoding="utf-8") as f:
        f.writelines(ja_blocks)

    print(f"[INFO] Japanese SRT written to: {ja_srt_path}")


def pipeline(args):
    logger = setup_logger()

    inp, tmp = prepare_input(args, logger)
    wav = prepare_wav(inp, tmp)
    segs = prepare_segments(wav, tmp, inp.stem)

    wenet_model_name = setup_wenet_environment()
    filtered = run_asr_on_segments(segs, wenet_model_name, args, logger)
    ja_lines = translate_zh_to_ja(filtered, args, logger)
    write_srt_files(filtered, ja_lines, args, inp, logger)

    logger.info("====== pipeline ended ======")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anime full pipeline Optimized")
    parser.add_argument("input", help="input mp4 file")
    parser.add_argument("output", nargs="?", help="output srt file")
    parser.add_argument("--tmpdir", default=TRANS_TMPDIR)
    parser.add_argument("--workers", type=int, default=4, help="workers for WeNet")
    parser.add_argument("--batch", type=int, default=16, help="translation batch size")
    parser.add_argument("--num-threads", type=int, default=4, help="torch threads number")
    parser.add_argument("--max-chars", type=int, default=28)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    pipeline(args)
