#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil

# import socket
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
import wave
import contextlib

# ---------- Config ----------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9000
DEFAULT_WORKERS = 2
WENET_CMD = "wenet"
WENET_MODEL = "/mnt/ramdisk/wenetspeech/"
# WENET_MODEL = "wenetspeech"
WENET_DEVICE = "cpu"  # cpu,npu,cuda
WENET_BEAM = 40
WARMUP_SECONDS = 0.6
WARMUP_AUDIO = "wenet_warmup.wav"
WARMUP_TIMEOUT = 60
REQUEST_TIMEOUT = 300
ANIME_WORDS_PATH = str(Path.home() / "etc" / "anime_words.txt")
MIN_DUR = 1.0

# ----------------------------

_lock_print = threading.Lock()


def safe_print(*args, **kwargs):
    with _lock_print:
        print(*args, **kwargs)


def get_wav_duration(wav_path: str) -> float:
    try:
        with contextlib.closing(wave.open(wav_path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0


class WenetWorkerPool:
    """
    Manages a ThreadPoolExecutor that runs `wenet` CLI jobs.
    Provides optional warmup for each worker to reduce cold-start penalty.
    """

    def __init__(
        self,
        workers: int = DEFAULT_WORKERS,
        model: str = WENET_MODEL,
        device: str = WENET_DEVICE,
        beam: int = WENET_BEAM,
        warmup_seconds: float = WARMUP_SECONDS,
    ):
        self.workers = max(1, int(workers))
        self.model = model
        self.device = device
        self.beam = beam
        self._executor = ThreadPoolExecutor(max_workers=self.workers)
        self._warmup_seconds = warmup_seconds
        self._warmup_done = False
        self._warmup_lock = threading.Lock()

    def _create_silence_wav(self, out_path: Path, seconds: float):
        """
        Create a short silent WAV using ffmpeg (anullsrc).
        """
        if out_path.exists():
            return
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=channel_layout=mono:sample_rate=16000",
            "-t",
            f"{seconds:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ]
        try:
            subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            safe_print(f"[warmup] created {out_path}")
        except Exception as e:
            safe_print(f"[warmup] ffmpeg failed to create silence: {e}")

    def _run_wenet_once(self, wav_path: str, timeout: int = REQUEST_TIMEOUT):
        p = Path(wav_path)
        if not p.exists():
            return 127, "", f"file not found: {wav_path}"

        dur = get_wav_duration(wav_path)
        if dur < MIN_DUR:
            return 0, "", f"too short ({dur:.3f}s)"

        cmd = [
            WENET_CMD,
            "-m",
            self.model,
            "--device",
            self.device,
            "--beam",
            str(self.beam),
            "--context_path",
            ANIME_WORDS_PATH,
            "--context_score",
            "3.0",
            "--punc",
            "-pm",
            "./wenet_punc_model",
            wav_path,
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )

            stderr_text = proc.stderr or ""
            if "list index out of range" in stderr_text:
                return 0, "", "asr_failed"

            if proc.returncode != 0:
                return 0, "", "asr_failed"

            raw_lines = proc.stdout.splitlines()
            candidates = []
            for ln in raw_lines:
                line = ln.strip()
                if not line:
                    continue
                if "torch_npu" in line or "Ascend NPU" in line:
                    continue
                if "Module" in line and "not found" in line:
                    continue
                if line.startswith("File:") or line.lower().startswith("processing"):
                    continue
                if line.startswith("Read"):
                    continue
                candidates.append(line)

            if not candidates:
                return 0, "", "asr_failed"

            return 0, candidates[0], stderr_text

        except subprocess.TimeoutExpired:
            return 124, "", f"timeout after {timeout}s"
        except Exception as e:
            if "list index out of range" in str(e):
                return 0, "", "asr_failed"
            return 1, "", f"exception: {e}"

    def warmup(self):
        """
        Warm up workers once: create a short silent wav and run a single
        wenet call to encourage model files to be loaded into OS cache.
        This is performed only once (thread-safe).
        """
        with self._warmup_lock:
            if self._warmup_done:
                return
            safe_print(
                "[warmup] Generating short silence WAV and running warmup calls..."
            )
            tmp = Path.cwd() / WARMUP_AUDIO
            try:
                self._create_silence_wav(tmp, self._warmup_seconds)
            except Exception as e:
                safe_print("[warmup] failed to create silence:", e)
                tmp = None

            # submit warmup tasks equal to worker count to populate parallel caches
            futures = []
            for _ in range(self.workers):
                futures.append(
                    self._executor.submit(
                        self._run_wenet_once, str(tmp) if tmp else "", WARMUP_TIMEOUT
                    )
                )
            for f in futures:
                try:
                    rc, out, err = f.result(timeout=WARMUP_TIMEOUT + 5)
                    safe_print(
                        f"[warmup] rc={rc}, out_len={len(out) if out else 0}, err_len={len(err) if err else 0}"
                    )
                except Exception as e:
                    safe_print("[warmup] warmup task exception:", e)

            # remove temp warmup wav
            try:
                if tmp and tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

            self._warmup_done = True
            safe_print("[warmup] done")

    def submit(self, wav_path: str, timeout: Optional[int] = None):
        """
        Submit a transcription job to the pool.
        Returns concurrent.futures.Future whose result() returns dict.
        """
        t = timeout if timeout is not None else REQUEST_TIMEOUT

        def job():
            rc, out, err = self._run_wenet_once(wav_path, timeout=t)
            # sanitize/trim stdout
            text = out.strip() if out else ""
            return {"rc": rc, "text": text, "stderr": err}

        return self._executor.submit(job)

    def shutdown(self, wait: bool = True):
        self._executor.shutdown(wait=wait)


# ---------- HTTP handler ----------
class WenetHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "WenetDaemon/0.1"

    def _send_json(self, obj: dict, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/transcribe":
            self._send_json({"ok": False, "error": "unknown endpoint"}, status=404)
            return

        # read body
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_json({"ok": False, "error": "empty body"}, status=400)
            return
        raw = self.rfile.read(length)
        try:
            j = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send_json({"ok": False, "error": f"invalid json: {e}"}, status=400)
            return

        wav_path = j.get("path")
        if not wav_path:
            self._send_json({"ok": False, "error": "missing 'path' field"}, status=400)
            return

        # security: ensure path is absolute and exists
        wav_p = Path(wav_path)
        if not wav_p.is_absolute():
            self._send_json({"ok": False, "error": "path must be absolute"}, status=400)
            return
        if not wav_p.exists():
            self._send_json({"ok": False, "error": "file not found"}, status=404)
            return

        # optional params
        timeout = int(j.get("timeout", REQUEST_TIMEOUT))

        # submit job
        self.server.pool.warmup()  # ensure warmup called at least once
        fut = self.server.pool.submit(str(wav_p), timeout=timeout)

        try:
            result = fut.result(timeout=timeout + 5)
        except Exception as e:
            self._send_json({"ok": False, "error": f"internal error: {e}"}, status=500)
            return

        # return result
        ok = result["rc"] == 0
        resp = {
            "ok": ok,
            "text": result["text"],
            "rc": result["rc"],
            "stderr": result["stderr"],
        }
        self._send_json(resp, status=200 if ok else 500)

    def log_message(self, format, *args):
        # silence default logging; print concise info
        safe_print(f"[http] {self.client_address[0]} - {format % args}")


# ---------- Server runner ----------
class WenetHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, pool: WenetWorkerPool):
        super().__init__(server_address, RequestHandlerClass)
        self.pool = pool


def main():
    ap = argparse.ArgumentParser(description="WeNet CLI daemon")
    ap.add_argument(
        "--host", default=DEFAULT_HOST, help="listen host (default 127.0.0.1)"
    )
    ap.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="listen port (default 9000)"
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="concurrent workers (default 2)",
    )
    ap.add_argument(
        "--model", default=WENET_MODEL, help="WeNet model name or local dir"
    )
    ap.add_argument("--device", default=WENET_DEVICE, help="WeNet device: cpu/npu/cuda")
    ap.add_argument("--beam", type=int, default=WENET_BEAM, help="WeNet beam size")
    ap.add_argument(
        "--warmup", action="store_true", help="perform warmup calls on start"
    )
    args = ap.parse_args()

    # Check presence of wenet in PATH
    if shutil.which(WENET_CMD) is None:
        safe_print("ERROR: 'wenet' not found in PATH. Please install or add to PATH.")
        return

    pool = WenetWorkerPool(
        workers=args.workers, model=args.model, device=args.device, beam=args.beam
    )
    server = WenetHTTPServer((args.host, args.port), WenetHTTPRequestHandler, pool)

    safe_print(
        f"[server] Listening on http://{args.host}:{args.port}  workers={args.workers}  model={args.model}"
    )

    try:
        if args.warmup:
            safe_print("[server] performing warmup...")
            pool.warmup()
        server.serve_forever()
    except KeyboardInterrupt:
        safe_print("[server] interrupted, shutting down...")
    finally:
        try:
            server.shutdown()
        except Exception:
            pass
        pool.shutdown()
        safe_print("[server] stopped")


if __name__ == "__main__":
    main()
