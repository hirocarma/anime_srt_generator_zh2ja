#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto


# ---------- Config ----------
@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 9000
    request_timeout: int = 300

    def validate(self):
        if not (0 < self.port < 65536):
            raise ValueError(f"invalid port: {self.port}")
        if self.request_timeout < 0:
            raise ValueError("request_timeout must be >= 0")


@dataclass
class ASRConfig:
    wenet_cmd: str = "wenet"
    workers: int = 2
    model: str = "wenetspeech"
    device: str = "cpu"
    beam: int = 40
    context_path: str = str(Path.home() / "etc" / "anime_words.txt")
    min_dur: float = 1.0

    def validate(self):
        if shutil.which(self.wenet_cmd) is None:
            raise ValueError(f"{self.wenet_cmd} not found in PATH")

        if self.workers < 1:
            raise ValueError("ASRConfig.workers must be >= 1")

        if not self.model:
            raise ValueError("ASRConfig.model must not be empty")

        if self.device not in ("cpu", "cuda", "npu"):
            raise ValueError(f"invalid device: {self.device}")

        if self.beam < 1:
            raise ValueError("ASRConfig.beam must be >= 1")

        if not os.path.isfile(self.context_path):
            raise ValueError(f"ASRConfig.context_path {self.context_path} not found")

        if self.min_dur <= 0:
            raise ValueError("ASRConfig.min_dur must be > 0")


@dataclass
class LoggingConfig:
    log_file: Path = Path("logs/wenet_daemon.log")
    level: int = logging.INFO

    def validate(self):
        if self.log_file.parent:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    server: ServerConfig
    asr: ASRConfig
    logging: LoggingConfig

    @classmethod
    def from_args(cls, args):
        return cls(
            server=ServerConfig(
                host=args.host or ServerConfig.host,
                port=args.port or ServerConfig.port,
                request_timeout=(
                    args.request_timeout
                    if args.request_timeout is not None
                    else ServerConfig.request_timeout
                ),
            ),
            asr=ASRConfig(
                workers=args.workers or ASRConfig.workers,
                model=args.model or ASRConfig.model,
                device=args.device or ASRConfig.device,
                context_path=args.context_path or ASRConfig.context_path,
                beam=args.beam or ASRConfig.beam,
            ),
            logging=LoggingConfig(
                log_file=(
                    Path(args.log_file) if args.log_file else LoggingConfig.log_file
                ),
                level=logging.DEBUG if args.debug else LoggingConfig.level,
            ),
        )

    def validate(self):
        self.server.validate()
        self.asr.validate()
        self.logging.validate()


class ASRCode(Enum):
    SUCCESS = auto()
    NO_RESULT = auto()
    BACKEND_ERROR = auto()
    TIMEOUT = auto()
    EXCEPTION = auto()

    @property
    def is_fatal(self) -> bool:
        return self in {
            ASRCode.BACKEND_ERROR,
            ASRCode.TIMEOUT,
            ASRCode.EXCEPTION,
        }

    @property
    def is_ok(self) -> bool:
        return self in {
            ASRCode.SUCCESS,
            ASRCode.NO_RESULT,
        }


@dataclass
class ASRResult:
    code: ASRCode
    text: str
    message: str = ""
    stderr: str = ""


# ----------------------------


def setup_logging(
    log_file: Path,
    level: int = logging.INFO,
):
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(threadName)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console (stdout)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)

    # file (rotate)
    fh = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)


def get_wav_duration(wav_path: str) -> float:
    try:
        with contextlib.closing(wave.open(wav_path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0


class ASRBackend:
    def transcribe(self, wav_path: Path, timeout: int) -> ASRResult:
        raise NotImplementedError


class WenetBackend(ASRBackend):
    def __init__(self, config: ASRConfig):
        self.config = config

    def _build_cmd(self, wav_path: Path) -> list[str]:
        return [
            self.config.wenet_cmd,
            "-m",
            self.config.model,
            "--device",
            self.config.device,
            "--beam",
            str(self.config.beam),
            "--context_path",
            self.config.context_path,
            "--context_score",
            "3.0",
            "--punc",
            "-pm",
            "./wenet_punc_model",
            str(wav_path),
        ]

    def transcribe(self, wav_path: Path, timeout: int) -> ASRResult:
        dur = get_wav_duration(str(wav_path))
        if dur == 0.0:
            return ASRResult(
                code=ASRCode.BACKEND_ERROR,
                text="",
                message="failed to read wav header",
            )

        if dur < self.config.min_dur:
            return ASRResult(
                code=ASRCode.NO_RESULT,
                text="",
                message=f"too short ({dur:.3f}s)",
                stderr="",
            )

        cmd = self._build_cmd(wav_path)

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
                return ASRResult(
                    code=ASRCode.BACKEND_ERROR,
                    text="",
                    message="wenet returned non-zero",
                    stderr=stderr_text,
                )

            candidates = self._filter_stdout(proc.stdout)
            if not candidates:
                return ASRResult(
                    code=ASRCode.NO_RESULT,
                    text="",
                    message="asr no candidates",
                    stderr=stderr_text,
                )

            return ASRResult(
                code=ASRCode.SUCCESS,
                text=candidates[0],
                message="OK candidates",
                stderr=stderr_text,
            )

        except subprocess.TimeoutExpired as e:
            return ASRResult(
                code=ASRCode.TIMEOUT,
                text="",
                message=f"timeout after {timeout}s",
                stderr=e.stderr or "",
            )
        except Exception as e:
            return ASRResult(
                code=ASRCode.EXCEPTION,
                text="",
                message=f"exception: {e}",
                stderr="",
            )

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

        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(self.config.workers))
        )

    def submit(self, wav_path: str, timeout: int):
        def job():
            return self.backend.transcribe(Path(wav_path), timeout)

        return self._executor.submit(job)

    def shutdown(self, wait: bool = True):
        self._executor.shutdown(wait=wait)


# ---------- HTTP handler ----------
class WenetHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "WenetDaemon/0.2"
    logger = logging.getLogger("http")

    # ---------- logging ----------
    def log_message(self, format, *args):
        self.logger.info(
            "%s - %s",
            self.client_address[0],
            format % args,
        )

    # ---------- response ----------
    def _send_json(self, obj: dict, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---------- helpers ----------
    def _ensure_endpoint(self) -> bool:
        parsed = urlparse(self.path)
        if parsed.path != "/transcribe":
            self._send_json(
                {"ok": False, "error": "unknown endpoint"},
                status=404,
            )
            return False
        return True

    def _read_json_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_json(
                {"ok": False, "error": "empty body"},
                status=400,
            )
            return None
        try:
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send_json(
                {"ok": False, "error": f"invalid json: {e}"},
                status=400,
            )
            return None

    def _validate_wav_path(self, j: dict) -> Path | None:
        wav_path = j.get("path")
        if not wav_path:
            self._send_json(
                {"ok": False, "error": "missing 'path' field"},
                status=400,
            )
            return None

        wav_p = Path(wav_path)
        if not wav_p.is_absolute():
            self._send_json(
                {"ok": False, "error": "path must be absolute"},
                status=400,
            )
            return None
        if not wav_p.exists():
            self._send_json(
                {"ok": False, "error": "file not found"},
                status=404,
            )
            return None

        return wav_p

    def _run_asr(self, wav_p: Path, timeout: int):
        fut = self.server.pool.submit(str(wav_p), timeout=timeout)
        try:
            return fut.result(timeout=timeout + 30)
        except Exception as e:
            self._send_json(
                {"ok": False, "error": f"internal error: {e}"},
                status=500,
            )
            return None

    def _send_asr_result(self, result):
        if result.code.is_fatal:
            self._send_json(
                {
                    "ok": False,
                    "code": result.code.name,
                    "error": result.message,
                },
                status=500,
            )
            return

        resp = {
            "ok": result.code.is_ok,
            "code": result.code.name,
            "text": result.text,
            "message": result.message,
            "stderr": result.stderr,
        }
        self._send_json(resp, status=200)

    # ---------- main entry ----------
    def do_POST(self):
        if not self._ensure_endpoint():
            return

        j = self._read_json_body()
        if j is None:
            return

        wav_p = self._validate_wav_path(j)
        if wav_p is None:
            return

        cfg = self.server.config
        timeout = int(j.get("timeout", cfg.request_timeout))

        result = self._run_asr(wav_p, timeout)
        if result is None:
            return

        self._send_asr_result(result)


# ---------- Server runner ----------
class WenetHTTPServer(HTTPServer):
    def __init__(
        self,
        config: ServerConfig,
        RequestHandlerClass,
        pool: WenetWorkerPool,
    ):
        server_address = (config.host, config.port)
        super().__init__(server_address, RequestHandlerClass)
        self.config = config
        self.pool = pool


def main():

    def build_server(config: AppConfig) -> WenetHTTPServer:
        backend = WenetBackend(config.asr)
        pool = WenetWorkerPool(config.asr, backend)
        return WenetHTTPServer(config.server, WenetHTTPRequestHandler, pool)

    ap = argparse.ArgumentParser(description="WeNet CLI daemon")
    ap.add_argument("--host", help="listen host")
    ap.add_argument("--port", type=int, help="listen port")
    ap.add_argument(
        "--request_timeout", type=int, help="HTTP request / ASR execution timeout"
    )
    ap.add_argument("--workers", type=int, help="concurrent workers")
    ap.add_argument("--model", help="WeNet model name or local dir")
    ap.add_argument("--device", help="WeNet device: cpu/npu/cuda")
    ap.add_argument("--beam", type=int, help="WeNet beam size")
    ap.add_argument("--context_path", help="WeNet context_path")
    ap.add_argument("--log-file", help="log file path")
    ap.add_argument("--debug", action="store_true", help="enable debug logging")

    args = ap.parse_args()
    config = AppConfig.from_args(args)
    config.validate()

    setup_logging(
        log_file=config.logging.log_file,
        level=config.logging.level,
    )

    logger = logging.getLogger(__name__)

    server = build_server(config)

    logger.info(
        "Listening on http://%s:%d workers=%d model=%s",
        config.server.host,
        config.server.port,
        config.asr.workers,
        config.asr.model,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("interrupted, shutting down...")
    finally:
        try:
            server.shutdown()
        except Exception:
            pass
        pool.shutdown()
        logger.info("stopped")


if __name__ == "__main__":
    main()
