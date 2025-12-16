# Anime SRT Generator (Chinese ASR → Japanese Translation)

An end-to-end **subtitle generation pipeline** for anime and similar video content.

This tool:
- Extracts audio from an MP4 file
- Automatically segments speech using **WebRTC VAD with hysteresis**
- Performs **Chinese ASR** using **WeNet** (with daemon mode)
- Translates Chinese subtitles to Japanese using **NLLB-200**
- Outputs **SRT subtitle files**:
  - Chinese + Japanese
  - Chinese only
  - Japanese only
- Uses **SQLite-based persistent caches** for both ASR and translation to dramatically speed up repeated runs

---

## Features

### 🎧 Audio Processing
- FFmpeg-based audio extraction (mono, 16kHz)
- Automatic loudness normalization and silence removal
- Robust segmentation with:
  - WebRTC VAD
  - Hysteresis smoothing
  - Adaptive maximum segment length
  - Context padding

### 🧠 Speech Recognition (ASR)
- Powered by **WeNet**
- WeNet run **Daemon mode** for high performance
- Parallel ASR with configurable worker count
- Persistent **SQLite ASR cache** keyed by audio content (SHA1)

### 🌏 Translation
- Chinese → Japanese translation via **facebook/nllb-200**
- GPU acceleration supported (FP16 if available)
- Adaptive beam size based on sentence length
- Batch translation for efficiency
- Persistent **SQLite translation cache**

### 📝 Subtitle Output
- SRT format compliant
- Automatic line splitting for readability
- Optional Chinese terminology normalization
- Outputs:
  - `output.srt` (Chinese + Japanese)
  - `output_zh.srt` (Chinese only)
  - `output_ja.srt` (Japanese only)

### ⚡ Performance Optimizations
- Optional **RAM disk** for WeNet model loading
- Multi-threaded ASR
- Batch translation
- Caching at every expensive stage

---

## Requirements

### System
- Linux (recommended)
- Python **3.9+**
- FFmpeg
- sudo access (only if RAM disk is enabled)

### Python Dependencies

```bash
pip install torch transformers tqdm opencc-python-reimplemented webrtcvad wenet
```

> **Note:**  
> - CUDA is strongly recommended for NLLB translation.  
> - CPU-only translation is supported but slow.


## WeNet Setup

### Install WeNet
Follow the official instructions:
- https://github.com/wenet-e2e/wenet

Make sure the `wenet` CLI is available in your `$PATH`.

### WeNet Daemon

The pipeline can automatically start a daemon defined in:


The daemon provides:
- Lower startup overhead
- Better throughput for long videos
- Stable performance with parallel ASR jobs

---

## Usage

### Basic Command

```bash
python3 anime_srt_generator_zh2ja.py input.mp4
```

### This generates:
- input.srt
- input_zh.srt
- input_ja.srt

### Specify Output File

``` bash
python3 anime_srt_generator_zh2ja.py input.mp4 output.srt
```

### Common Options

``` bash
--tmpdir        Temporary working directory (default: ~/tmp/anime_trans_tmp)
--workers       Parallel ASR workers (default: 4)
--batch         Translation batch size (default: 16)
--num-threads   Torch CPU threads (default: 4)
--max-chars     Max characters per subtitle line (default: 28)
```

### Example:

``` bash
python3 anime_srt_generator_zh2ja.py episode01.mp4 \
  --workers 8 \
  --batch 32 \
  --num-threads 8
```

### Output Files
- File	Description
- *.srt	Chinese + Japanese subtitles
- *_zh.srt	Chinese-only subtitles
- *_ja.srt	Japanese-only subtitles

###  Output Format Details

- Time format: `HH:MM:SS,000`
- Line wrapping is automatically applied using `--max-chars`
- Sentence-aware splitting prefers punctuation boundaries (`。！？`)
- Fallback midpoint splitting is used when punctuation is unavailable

### Example block:

``` bash
12
00:03:41,000 --> 00:03:44,000
他真的不知道自己在做什么。
彼は自分が何をしているのか
本当に分かっていなかった。

```

---
## Caching Behavior 

### Translation Cache

- SQLite-based persistent cache

- Key: Chinese text

- Value: Japanese translation

### ASR Cache

- SQLite-based persistent cache

- Key: SHA1 hash of WAV segment contents

- Value: Recognized Chinese text

Cache files are stored under:

``` bash
~/tmp/anime_trans_tmp/
```

Caches persist across runs and significantly reduce processing time.


---
## Configuration Highlights

Important configuration options at the top of the script:

``` python
WENET_MODEL = "wenetspeech"
DEFAULT_NLLB_MODEL = "facebook/nllb-200-3.3B"

USE_TRANS_CACHE = 1
USE_ASR_CACHE = 1

USE_RAMDISK = 1
```

You may disable any optimization depending on your environment.

---
### Notes & Tips

- The first execution will be slow due to model loading and cache warm-up.

- Subsequent executions on the same or similar content are much faster.

- Using a RAM disk greatly reduces WeNet model loading latency.

- Designed primarily for Chinese audio → Japanese subtitles.

- Long videos benefit most from daemon mode and caching.

---
## License

This project is provided as-is for research and personal use.

Please comply with the licenses of:

- WeNet

- NLLB-200

- FFmpeg

- WebRTC VAD

---
## Acknowledgements

- WeNet ASR Team

- Meta AI (NLLB-200)

- Hugging Face Transformers

- WebRTC VAD

---
## Disclaimer

This tool is intended for personal, research, or educational use only.

Ensure that you have the legal right to process, transcribe, and translate the input media.
