"""
Audio extraction and speech transcription using local faster-whisper.
"""

import os
import subprocess
import shutil
from typing import List, Dict, Any, Tuple

# Fix compatibility between PyAV >= 19.0.0 and faster-whisper <= 1.2.1
# Auto-patch faster_whisper/audio.py if metadata_errors="ignore" is present
try:
    import faster_whisper.audio as _fwa
    _p = _fwa.__file__
    with open(_p, "r", encoding="utf-8") as _f:
        _src = _f.read()
    if ', metadata_errors="ignore"' in _src:
        _src = _src.replace(', metadata_errors="ignore"', '')
        with open(_p, "w", encoding="utf-8") as _f:
            _f.write(_src)
except Exception:
    pass

try:
    import av
    _orig_av_open = av.open

    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        kwargs.pop("metadata_encoding", None)
        return _orig_av_open(*args, **kwargs)

    av.open = _safe_av_open
except ImportError:
    pass

from faster_whisper import WhisperModel


def get_ffmpeg_bin() -> str:
    """Find ffmpeg binary in local bin, PATH, or nixpkgs."""
    local_bin = os.path.abspath("bin/ffmpeg")
    if os.path.exists(local_bin) and os.access(local_bin, os.X_OK):
        return local_bin
    system_bin = shutil.which("ffmpeg")
    if system_bin:
        return system_bin
    return "ffmpeg"


def get_ffprobe_bin() -> str:
    local_bin = os.path.abspath("bin/ffprobe")
    if os.path.exists(local_bin) and os.access(local_bin, os.X_OK):
        return local_bin
    system_bin = shutil.which("ffprobe")
    if system_bin:
        return system_bin
    return "ffprobe"


def extract_audio(video_path: str, output_wav: str) -> str:
    """Extract 16kHz mono audio from video file for Whisper processing."""
    os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
    ffmpeg_bin = get_ffmpeg_bin()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel", "error",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav
    ]
    subprocess.run(cmd, check=True)
    return output_wav


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    ffprobe_bin = get_ffprobe_bin()
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 0.0


class Transcriber:
    def __init__(self, config: dict):
        self.cfg = config.get("whisper", {})
        self.model_size = self.cfg.get("model_size", "base")
        self.requested_device = str(self.cfg.get("device", "auto")).lower()
        self.requested_compute = str(self.cfg.get("compute_type", "auto")).lower()
        self.batch_size = int(self.cfg.get("batch_size", 8))
        self.language = self.cfg.get("language", None)

        self.model, self._batched = self._init_model()
        mode = f"{self.device}/{self.compute_type}" + (" + батчі" if self._batched else "")
        print(f"[✓] Whisper модель готова: '{self.model_size}' ({mode})")

    # ------------------------------------------------------------------
    # Бекенд: CUDA з безпечним відкатом на CPU
    # ------------------------------------------------------------------

    @staticmethod
    def _preload_cuda_libs() -> bool:
        """
        Підтягує cuBLAS/cuDNN з pip-паків (nvidia-cublas-cu12, nvidia-cudnn-cu12)
        через ctypes, щоб ctranslate2 знайшов їх за soname. True — хоч щось завантажилось.
        """
        import ctypes
        loaded = 0
        for mod_name in ("nvidia.cublas", "nvidia.cudnn"):
            try:
                mod = __import__(mod_name, fromlist=["__path__"])
                libdir = os.path.join(list(mod.__path__)[0], "lib")
                for so in sorted(os.listdir(libdir)):
                    if ".so" in so:
                        try:
                            ctypes.CDLL(os.path.join(libdir, so), mode=ctypes.RTLD_GLOBAL)
                            loaded += 1
                        except OSError:
                            pass
            except (ImportError, OSError):
                pass
        return loaded > 0

    def _init_model(self):
        """
        Повертає (model, batched_pipeline | None).
        Порядок: CUDA (float16, батчі) → прова-інференс 1с тиші → будь-яка невдача → CPU (int8).
        """
        from faster_whisper import WhisperModel

        def _probe(model):
            import numpy as np
            next(iter(model.transcribe(np.zeros(16000, dtype=np.float32), language="ru")), None)

        # --- Явний CPU
        if self.requested_device == "cpu":
            compute = self.requested_compute if self.requested_compute != "auto" else "int8"
            print(f"[*] Ініціалізація Whisper моделі: '{self.model_size}' (device=cpu, compute={compute})...")
            self.compute_type = compute
            return WhisperModel(self.model_size, device="cpu", compute_type=compute), None

        # --- CUDA (auto або cuda)
        try:
            if not self._preload_cuda_libs() and self.requested_device == "cuda":
                print("[!] CUDA-бібліотеки (cublas/cudnn) не знайдені")
            compute = self.requested_compute if self.requested_compute not in ("auto", "") else "float16"
            print(f"[*] Спроба Whisper на CUDA: '{self.model_size}' (compute={compute})...")
            model = WhisperModel(self.model_size, device="cuda", compute_type=compute)
            _probe(model)
            self.compute_type = compute
            print("[⚡] Whisper працює на GPU")

            batched = None
            try:
                from faster_whisper import BatchedInferencePipeline
                batched = BatchedInferencePipeline(model=model)
                print(f"[⚡] Батч-режим увімкнено (batch_size={self.batch_size})")
            except Exception as e:
                print(f"[!] Батч-режим недоступний ({e}) — послідовний на GPU")
            return model, batched
        except Exception as e:
            print(f"[!] CUDA недоступна ({type(e).__name__}: {str(e)[:120]}) — Whisper на CPU")

        compute = self.requested_compute if self.requested_compute != "auto" else "int8"
        print(f"[*] Ініціалізація Whisper моделі: '{self.model_size}' (device=cpu, compute={compute})...")
        self.compute_type = compute
        return WhisperModel(self.model_size, device="cpu", compute_type=compute), None

    @property
    def device(self) -> str:
        return getattr(self.model, "device", "cpu")

    def transcribe(self, audio_wav: str) -> Tuple[List[Dict[str, Any]], float, str]:
        """
        Transcribes the audio file with word timestamps and VAD filter.
        Returns (segments, total_duration, detected_language).
        """
        vad_params = dict(
            min_silence_duration_ms=500,
            speech_pad_ms=300
        )

        gen = None
        if self._batched is not None:
            try:
                gen, info = self._batched.transcribe(
                    audio_wav,
                    vad_filter=True,
                    vad_parameters=vad_params,
                    word_timestamps=True,
                    language=self.language,
                    batch_size=self.batch_size,
                )
            except Exception as e:
                print(f"[!] Батч-транскрибація впала ({e}) — перемикання на послідовну")
                self._batched = None
        if gen is None:
            gen, info = self.model.transcribe(
                audio_wav,
                vad_filter=True,
                vad_parameters=vad_params,
                word_timestamps=True,
                language=self.language
            )

        segments = []
        last_report_pct = 0.0
        for s in gen:
            words = []
            if s.words:
                for w in s.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability
                    })
            segments.append({
                "id": s.id,
                "start": s.start,
                "end": s.end,
                "text": s.text.strip(),
                "words": words
            })
            # Прогрес у логи (TUI показує останній рядок) — щоб не було «мовчазних» хвилин
            if info.duration > 0:
                pct = min(99.0, s.end / info.duration * 100.0)
                if pct - last_report_pct >= 5.0:
                    last_report_pct = pct
                    print(f"[*] Транскрибація: {pct:.0f}% ({s.end / 60:.0f}/{info.duration / 60:.0f} хв аудіо)")

        return segments, info.duration, info.language
