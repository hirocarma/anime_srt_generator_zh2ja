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
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from http.server import BaseHTTPRequestHandler, HTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlparse


# ---------- Config ----------
@dataclass
class ServerConfig:
    """
    Configuration for the HTTP server.

    Attributes:
        host (str): Host address to bind the server.
        port (int): TCP port number to listen on.
        request_timeout (int): Default timeout (in seconds) for
            HTTP requests and ASR processing.
    """

    host: str = "127.0.0.1"
    port: int = 9000
    request_timeout: int = 300

    def validate(self) -> None:
        """
        Validate server configuration values.

        Raises:
            ValueError: If the port number or timeout value is invalid.
        """

        if not 0 < self.port < 65536:
            raise ValueError(f"Invalid port: {self.port}")
        if self.request_timeout < 0:
            raise ValueError("request_timeout must be >= 0")


@dataclass
class ASRConfig:
    """
    Configuration for the ASR backend.

    Attributes:
        wenet_cmd (str): WeNet CLI command or executable path.
        workers (int): Number of concurrent ASR worker threads.
        model (str): WeNet model name or local directory.
        device (str): Device for inference ("cpu", "cuda", or "npu").
        beam (int): Beam size for beam search.
        context_path (str): Path to context word list file.
        min_dur (float): Minimum audio duration (seconds) required
            to run ASR.
    """

    wenet_cmd: str = "wenet"
    workers: int = 2
    model: str = "wenetspeech"
    device: str = "cpu"
    beam: int = 40
    context_path: str = str(Path.home() / "etc" / "anime_words.txt")
    min_dur: float = 1.0

    def validate(self) -> None:
        """
        Validate ASR backend configuration.

        Raises:
            ValueError: If required commands, files, or parameter values
                are invalid or missing.
        """

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
    """
    Logging configuration.

    Attributes:
        log_file (Path): Path to the log file.
        level (int): Logging level (e.g., logging.INFO).
    """

    log_file: Path = Path("logs/wenet_daemon.log")
    level: int = logging.INFO

    def validate(self) -> None:
        """
        Ensure that the log directory exists.
        """

        if self.log_file.parent:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """
    Root application configuration container.

    Attributes:
        server (ServerConfig): HTTP server configuration.
        asr (ASRConfig): ASR backend configuration.
        logging (LoggingConfig): Logging configuration.
    """

    server: ServerConfig
    asr: ASRConfig
    logging: LoggingConfig

    @classmethod
    def from_args(cls, args):
        """
        Create AppConfig from command-line arguments.

        Args:
            args: Parsed arguments from argparse.

        Returns:
            AppConfig: Constructed application configuration.
        """

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

    def validate(self) -> None:
        """
        Validate all nested configuration objects.
        """

        self.server.validate()
        self.asr.validate()
        self.logging.validate()


class ASRCode(Enum):
    """
    Enumeration of ASR processing result codes.

    SUCCESS:
        Recognition completed successfully.
    NO_RESULT:
        Audio was processed but no valid recognition result was found.
    BACKEND_ERROR:
        ASR backend failed or exited abnormally.
    TIMEOUT:
        ASR processing timed out.
    EXCEPTION:
        An unexpected exception occurred.
    """

    SUCCESS = auto()
    NO_RESULT = auto()
    BACKEND_ERROR = auto()
    TIMEOUT = auto()
    EXCEPTION = auto()

    @property
    def is_fatal(self) -> bool:
        """
        Check whether this result code represents a fatal error.

        Returns:
            bool: True if the error is fatal.
        """

        return self in {
            ASRCode.BACKEND_ERROR,
            ASRCode.TIMEOUT,
            ASRCode.EXCEPTION,
        }

    @property
    def is_ok(self) -> bool:
        """
        Check whether this result code is considered a successful outcome.

        Returns:
            bool: True if the result is successful or non-fatal.
        """

        return self in {
            ASRCode.SUCCESS,
            ASRCode.NO_RESULT,
        }


@dataclass
class ASRResult:
    """
    Container for ASR processing results.

    Attributes:
        code (ASRCode): Result status code.
        text (str): Recognized text.
        message (str): Human-readable message or explanation.
        stderr (str): Captured standard error output from backend.
    """

    code: ASRCode
    text: str
    message: str = ""
    stderr: str = ""


# ----------------------------


def setup_logging(
    log_file: Path,
    level: int = logging.INFO,
) -> None:
    """
    Initialize application-wide logging configuration.

    This function configures both console (stdout) and rotating file
    logging with a unified format. Existing root logger handlers
    are cleared and replaced.

    Args:
        log_file (Path):
            Path to the log file. Parent directories are created
            automatically if they do not exist.
        level (int):
            Logging level (e.g., logging.INFO, logging.DEBUG).

    Side Effects:
        - Creates the log directory if necessary.
        - Reconfigures the root logger.
        - Attaches a StreamHandler (stdout) and a RotatingFileHandler.

    Notes:
        - The file logger rotates when the file size exceeds 10 MB.
        - Up to 5 backup log files are retained.
        - UTF-8 encoding is used for file logging.
    """

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


class ASRBackend:
    """
    Base class for ASR backends.

    Provides shared audio utility methods and defines
    the transcribe interface.
    """

    @staticmethod
    def get_wav_duration(wav_path: Path) -> float:
        """
        Get the duration of a WAV file in seconds.

        Args:
            wav_path (Path): Path to the WAV file.

        Returns:
            float: Duration in seconds, or 0.0 if reading fails.
        """

        try:
            with contextlib.closing(wave.open(str(wav_path), "rb")) as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        except (wave.Error, OSError):
            return 0.0

    def transcribe(self, wav_path: Path, timeout: int) -> ASRResult:
        """
        Transcribe an audio file into text.

        Args:
            wav_path (Path): Path to the audio file.
            timeout (int): Timeout in seconds.

        Returns:
            ASRResult: Transcription result.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """

        raise NotImplementedError


class WenetBackend(ASRBackend):
    """
    Initialize the WeNet backend.

    Args:
        config (ASRConfig): ASR configuration.
    """

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
        """
        Run WeNet ASR on a WAV file.

        Args:
            wav_path (Path): Path to the audio file.
            timeout (int): Maximum execution time in seconds.

        Returns:
            ASRResult: Transcription result.
        """

        result: ASRResult

        dur = self.get_wav_duration(wav_path)
        if dur == 0.0:
            result = ASRResult(
                code=ASRCode.BACKEND_ERROR,
                text="",
                message="failed to read wav header",
            )

        elif dur < self.config.min_dur:
            result = ASRResult(
                code=ASRCode.NO_RESULT,
                text="",
                message=f"too short ({dur:.3f}s)",
                stderr="",
            )

        else:
            cmd = self._build_cmd(wav_path)

            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                    check=False,  # explicitly handle returncode manually
                )

                stderr_text = proc.stderr or ""
                if proc.returncode != 0:
                    result = ASRResult(
                        code=ASRCode.BACKEND_ERROR,
                        text="",
                        message="wenet returned non-zero",
                        stderr=stderr_text,
                    )
                else:
                    candidates = self._filter_stdout(proc.stdout)
                    if not candidates:
                        result = ASRResult(
                            code=ASRCode.NO_RESULT,
                            text="",
                            message="asr no candidates",
                            stderr=stderr_text,
                        )
                    else:
                        result = ASRResult(
                            code=ASRCode.SUCCESS,
                            text=candidates[0],
                            message="OK candidates",
                            stderr=stderr_text,
                        )

            except subprocess.TimeoutExpired as e:
                result = ASRResult(
                    code=ASRCode.TIMEOUT,
                    text="",
                    message=f"timeout after {timeout}s",
                    stderr=e.stderr or "",
                )
            except OSError as e:
                result = ASRResult(
                    code=ASRCode.EXCEPTION,
                    text="",
                    message=f"exception: {e}",
                    stderr="",
                )

        return result

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
    """
    Thread pool manager for executing ASR jobs asynchronously.
    """

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

    def submit(self, wav_path: Path, timeout: int):
        """
        Submit an ASR job to the worker pool.

        Args:
            wav_path (Path): Path to the audio file.
            timeout (int): ASR timeout in seconds.

        Returns:
            Future: A future returning an ASRResult.
        """

        def job():
            return self.backend.transcribe(wav_path, timeout)

        return self._executor.submit(job)

    def shutdown(self, wait: bool = True) -> None:
        """
        Shut down the worker pool.

        Args:
            wait (bool): Whether to wait for running jobs to finish.
        """

        self._executor.shutdown(wait=wait)


def asr_result_to_http_response(result: ASRResult) -> tuple[int, dict]:
    """
    Convert an ASRResult into an HTTP response payload.

    Args:
        result (ASRResult): ASR processing result.

    Returns:
        tuple[int, dict]: HTTP status code and JSON response body.
    """
    if result.code.is_fatal:
        return 500, {
            "ok": False,
            "code": result.code.name,
            "error": result.message,
        }

    return 200, {
        "ok": result.code.is_ok,
        "code": result.code.name,
        "text": result.text,
        "message": result.message,
        "stderr": result.stderr,
    }


# ---------- HTTP handler ----------
class WenetHTTPRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler providing the /transcribe ASR endpoint.
    """

    server_version = "WenetDaemon/0.2"
    logger = logging.getLogger("http")
    FUTURE_TIMEOUT_MARGIN = 30

    # ---------- logging ----------
    def log_message(self, format, *args) -> None:  # pylint: disable=redefined-builtin
        """
        Override BaseHTTPRequestHandler.log_message.

        Note:
        The parameter name `format` is required to match the base class
        signature and intentionally shadows the built-in `format()`.
        """

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
        except (UnicodeDecodeError, json.JSONDecodeError, OSError) as e:
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

        fut = self.server.pool.submit(wav_p, timeout=timeout)
        try:
            return fut.result(timeout=timeout + self.FUTURE_TIMEOUT_MARGIN)

        except TimeoutError:
            self._send_json(
                {"ok": False, "error": "ASR execution timeout"},
                status=504,
            )
            return None

        except CancelledError:
            self._send_json(
                {"ok": False, "error": "ASR job cancelled"},
                status=503,
            )
        return None

    def _send_asr_result(self, result: ASRResult) -> None:
        status, body = asr_result_to_http_response(result)
        self._send_json(body, status=status)

    # ---------- main entry ----------
    # pylint: disable=invalid-name
    def do_POST(self):
        """
        Handle POST requests for the /transcribe endpoint.
        """

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
    """
    HTTPServer extension that holds configuration and ASR worker pool.
    """

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


def build_server(config: AppConfig) -> WenetHTTPServer:
    """
    Build and initialize the HTTP server and ASR components.

    Args:
        config (AppConfig): Application configuration.

    Returns:
        WenetHTTPServer: Initialized server instance.
    """

    backend = WenetBackend(config.asr)
    pool = WenetWorkerPool(config.asr, backend)
    return WenetHTTPServer(config.server, WenetHTTPRequestHandler, pool)


def main():
    """
    Application entry point.

    Responsibilities:
        - Parse command-line arguments
        - Build and validate configuration
        - Initialize logging
        - Start and gracefully shut down the HTTP server
    """

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
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        server.pool.shutdown()
        logger.info("stopped")


if __name__ == "__main__":
    main()
