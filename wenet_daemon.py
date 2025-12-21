#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
import wave
import contextlib
from dataclasses import dataclass
from argparse import Namespace
from typing import Optional

# ---------- Config ----------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9000
WENET_CMD = "wenet"

REQUEST_TIMEOUT = 300
ANIME_WORDS_PATH = str(Path.home() / "etc" / "anime_words.txt")
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


@dataclass
class ASRConfig:
    workers: int = 2
    model: str = "wenetspeech"
    device: str = "cpu"
    beam: int = 40
    min_dur: float = 1.0
    request_timeout: int = 300

    @classmethod
    def from_args(cls, args: Namespace) -> "ASRConfig":
        return cls(
            workers=args.workers if args.workers is not None else cls.workers,
            model=args.model if args.model is not None else cls.model,
            device=args.device if args.device is not None else cls.device,
            beam=args.beam if args.beam is not None else cls.beam,
        )


@dataclass
class ASRResult:
    rc: int
    text: str
    stderr: str


class ASRBackend:
    def transcribe(self, wav_path: Path, timeout: int) -> ASRResult:
        raise NotImplementedError


class WenetBackend(ASRBackend):
    def __init__(self, config: ASRConfig):
        self.config = config

    def transcribe(self, wav_path: Path, timeout: int) -> ASRResult:
        if not wav_path.exists():
            return ASRResult(127, "", f"file not found: {wav_path}")

        dur = get_wav_duration(str(wav_path))
        if dur < self.config.min_dur:
            return ASRResult(0, "", f"too short ({dur:.3f}s)")

        cmd = [
            WENET_CMD,
            "-m",
            self.config.model,
            "--device",
            self.config.device,
            "--beam",
            str(self.config.beam),
            "--context_path",
            ANIME_WORDS_PATH,
            "--context_score",
            "3.0",
            "--punc",
            "-pm",
            "./wenet_punc_model",
            str(wav_path),
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
            if proc.returncode != 0:
                return ASRResult(0, "", "asr_failed")

            candidates = self._filter_stdout(proc.stdout)
            if not candidates:
                return ASRResult(0, "", "asr_failed")

            return ASRResult(0, candidates[0], stderr_text)

        except subprocess.TimeoutExpired:
            return ASRResult(124, "", f"timeout after {timeout}s")
        except Exception as e:
            return ASRResult(1, "", f"exception: {e}")

    def _filter_stdout(self, stdout: str) -> list[str]:
        lines = []
        for ln in stdout.splitlines():
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
            lines.append(line)
        return lines


class WenetWorkerPool:
    def __init__(
        self,
        config: ASRConfig,
        backend: ASRBackend,
    ):
        self.config = config
        self.backend = backend

        # self.workers = self.config.workers
        self._executor = ThreadPoolExecutor(max_workers=self.config.workers)

    def submit(self, wav_path: str, timeout: Optional[int] = None):
        t = timeout if timeout is not None else self.config.request_timeout

        def job():
            r = self.backend.transcribe(Path(wav_path), t)
            return {
                "rc": r.rc,
                "text": r.text,
                "stderr": r.stderr,
            }

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
        help="concurrent workers (default 2)",
    )
    ap.add_argument("--model", help="WeNet model name or local dir")
    ap.add_argument("--device", help="WeNet device: cpu/npu/cuda")
    ap.add_argument("--beam", type=int, help="WeNet beam size")

    args = ap.parse_args()

    # Check presence of wenet in PATH
    if shutil.which(WENET_CMD) is None:
        safe_print("ERROR: 'wenet' not found in PATH. Please install or add to PATH.")
        return

    args = ap.parse_args()
    config = ASRConfig.from_args(args)

    backend = WenetBackend(config)
    pool = WenetWorkerPool(config, backend)

    server = WenetHTTPServer((args.host, args.port), WenetHTTPRequestHandler, pool)

    safe_print(
        f"[server] Listening on http://{args.host}:{args.port}"
        + f" workers={config.workers} model={config.model}"
    )

    try:
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
