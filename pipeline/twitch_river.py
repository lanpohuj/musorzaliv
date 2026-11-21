"""River inference using timestamped frames and locally processed speech."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import math
import os
from pathlib import Path
import subprocess

import httpx
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ValidationError

from .twitch_editor import CameraReview, EditingModel, SCAN_FPS, remote_call
from .twitch_media import EditError, executable, probe

RIVER_URL = "https://api.river.ai/v1"
RIVER_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
RIVER_RATES = {RIVER_MODEL: (0.07, 0.37), "Qwen/Qwen3.8-27B-FP8": (0.10, 2.00)}
RIVER_DENSE_FPS = 3
BOARD_COLUMNS = 3
BOARD_ROWS = 4


def model_name() -> str:
    return os.environ.get("TWITCH_RIVER_MODEL", "").strip() or RIVER_MODEL


def _decode(video: Path, *args: str, seek: float | None = None) -> bytes:
    try:
        position = ["-ss", str(seek)] if seek is not None else []
        result = subprocess.run(
            [executable("ffmpeg"), "-nostdin", "-v", "error", "-threads", "2", *position, "-i", str(video), *args, "-"],
            capture_output=True, timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EditError("Не удалось извлечь кадры и звуковые метки для River.") from error
    if result.returncode:
        raise EditError("FFmpeg не смог прочитать фрагмент для River.")
    return result.stdout


def timestamped_frames(video: Path, fps: float, width: int, height: int) -> list[dict]:
    raw = _decode(video, "-an", "-vf", f"fps={fps}:start_time=0,scale={width}:{height},setsar=1",
                  "-pix_fmt", "rgb24", "-threads", "2", "-f", "rawvideo")
    frame_bytes = width * height * 3
    count = len(raw) // frame_bytes
    if not count:
        raise EditError("Во фрагменте для River нет читаемых кадров.")
    stamp_height = 27
    row_height = height + stamp_height
    board_count = BOARD_COLUMNS * BOARD_ROWS
    font = ImageFont.load_default(size=20)
    content = []
    for first in range(0, count, board_count):
        stop = min(count, first + board_count)
        rows = math.ceil((stop - first) / BOARD_COLUMNS)
        board = Image.new("RGB", (width * BOARD_COLUMNS, row_height * rows), "#11151e")
        draw = ImageDraw.Draw(board)
        for i in range(first, stop):
            row, col = divmod(i - first, BOARD_COLUMNS)
            x, y = col * width, row * row_height
            frame = Image.frombytes("RGB", (width, height), raw[i * frame_bytes:(i + 1) * frame_bytes])
            board.paste(frame, (x, y + stamp_height))
            draw.text((x + 7, y + 2), f"{i / fps:.2f} s", fill="white", font=font)
        output = BytesIO()
        board.save(output, format="JPEG", quality=82)
        data = base64.b64encode(output.getvalue()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}})
    return content


def camera_frames(video: Path, duration: float) -> list[dict]:
    """Keep larger individual frames and coordinate rulers outside the video."""
    count = min(9, max(3, math.ceil(duration / 4) + 1))
    content = []
    for index in range(count):
        seconds = index * max(0, duration - .12) / (count - 1)
        raw = _decode(video, "-an", "-vf", "scale=960:540", "-frames:v", "1",
                      "-pix_fmt", "rgb24", "-f", "rawvideo", seek=seconds)
        if not raw:
            continue
        frame = Image.frombytes("RGB", (960, 540), raw)
        canvas = Image.new("RGB", (1002, 572), "#151722")
        canvas.paste(frame, (42, 32))
        draw = ImageDraw.Draw(canvas)
        for tick in range(0, 1001, 100):
            x = min(42 + round(960 * tick / 1000), 973)
            y = min(32 + round(540 * tick / 1000), 557)
            draw.text((x, 12), str(tick), fill="white")
            draw.text((3, y), str(tick), fill="white")
        output = BytesIO()
        canvas.save(output, format="JPEG", quality=87)
        content.append({"type": "text", "text": f"Frame at {seconds:.2f} s"})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                        base64.b64encode(output.getvalue()).decode("ascii")}})
    if not content:
        raise EditError("Не удалось получить кадры для проверки вебки.")
    return content


def audio_peaks(video: Path) -> list[float]:
    import numpy as np

    raw = _decode(video, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le")
    samples = np.frombuffer(raw, dtype="<i2")
    per_block = 2000
    blocks = len(samples) // per_block
    if blocks < 4:
        return []
    signal = samples[:blocks * per_block].reshape(blocks, per_block).astype(np.float32)
    rms = np.sqrt(np.mean(signal * signal, axis=1))

    med = max(250.0, float(np.median(rms)))
    mad = float(np.median(np.abs(rms - med)))
    normalizer = max(100.0, mad * 1.4826)
    z = (rms - med) / normalizer

    k = min(blocks, 6)
    kernel = np.ones(k) / k
    rolling_rms = np.convolve(rms, kernel, mode="same")
    rolling_z = (rolling_rms - med) / normalizer
    p85 = float(np.percentile(rms, 85))

    activation = 0.55 * z + 0.45 * rolling_z

    candidates = []
    for i in range(2, blocks):
        prior = max(med, float(np.median(rms[max(0, i - 6):i])))
        is_spike = z[i] >= 2.0
        is_sustained = rolling_z[i] >= 1.5 and rms[i] >= p85
        is_jump = rms[i] >= prior * 1.65 and z[i] >= 1.3
        if is_spike or is_sustained or is_jump:
            candidates.append(i)

    candidates.sort(key=lambda i: -activation[i])
    peaks = []
    for i in candidates:
        if all(abs(i - old) >= 4 for old in peaks):
            peaks.append(i)
            if len(peaks) == 20:
                break
    return [round(i * 0.25, 2) for i in sorted(peaks)]


def strict_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema


class RiverEditor(EditingModel):
    provider = "River"
    requires_scan_transcript = True

    def __init__(self):
        key = os.environ.get("RIVER_API_KEY", "").strip()
        if not key:
            raise EditError("Задайте RIVER_API_KEY, чтобы запустить анализ Twitch через River.")
        self.model = model_name()
        self.client = httpx.Client(base_url=RIVER_URL + "/", timeout=180,
                                   headers={"Authorization": f"Bearer {key}"})

    def close(self):
        self.client.close()

    def query(self, prompt: str, schema: type[BaseModel], video: Path | None = None,
              *, fps: float = SCAN_FPS) -> tuple[BaseModel, dict]:
        content = [{"type": "text", "text": prompt}]
        if video and schema is CameraReview:
            content[0]["text"] += "\nThe rulers label 0–1000 in the video image, excluding margins."
            content.extend(camera_frames(video, probe(video).duration))
        elif video:
            info = probe(video)
            long_edge = 384 if fps == SCAN_FPS else 448
            ratio = long_edge / max(info.width, info.height)
            width = max(2, round(info.width * ratio / 2) * 2)
            height = max(2, round(info.height * ratio / 2) * 2)
            samples_per_second = min(fps, RIVER_DENSE_FPS)
            peaks = audio_peaks(video) if info.audio else []
            content[0]["text"] += (
                "\nThe available media are ordered contact sheets. Read each row left to right. "
                "The timestamp above each tile is seconds since the window began. Camera boxes "
                "are relative to one individual tile's video image, excluding the timestamp strip. "
                "Only local ASR provides spoken words and timings. Do not invent dialogue, speaker "
                "identity, or sound effects from still frames. Unknown speaker is valid. "
                f"Mixed audio level jumps, useful only as cues (seconds): {peaks}."
            )
            content.extend(timestamped_frames(video, samples_per_second, width, height))
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
            "max_completion_tokens": 8192,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "strict": True, "schema": strict_schema(schema),
            }},
        }
        if self.model == RIVER_MODEL:
            payload["reasoning_effort"] = "medium" if video else "none"
        elif self.model == "Qwen/Qwen3.8-27B-FP8":
            payload["reasoning_effort"] = "low" if video else "none"

        def request():
            response = self.client.post("chat/completions", json=payload)
            if response.is_error:
                # Access the status through a generic API exception without rendering response bodies.
                response.raise_for_status()
            return response

        usage = {"input": 0, "output": 0}
        for attempt in range(2):
            response = remote_call(self.provider, request)
            try:
                data = response.json()
                tokens = data.get("usage") or {}
                usage["input"] += int(tokens.get("prompt_tokens") or 0)
                usage["output"] += int(tokens.get("completion_tokens") or 0)
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("truncated")
                return schema.model_validate_json(choice["message"]["content"]), usage
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError, ValidationError):
                if attempt == 0:
                    content[0]["text"] += "\nReturn the complete JSON object, with no prose outside it."
                    continue
                request_id = response.headers.get("x-river-request-id")
                detail = f" Идентификатор River: {request_id}." if request_id else ""
                raise EditError(f"River вернул неполный ответ. Повторите этап.{detail}") from None
        raise AssertionError("unreachable")
