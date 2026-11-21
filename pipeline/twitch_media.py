"""Source media, a frame-aligned edit timeline, and verified Twitch exports."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

from .subtitles import format_ass_time

FPS = 30
RATE = 48000
SAMPLES_PER_FRAME = RATE // FPS
LAYOUT_VERSION = 5
CAPTION_BAND = 24
CAPTION_FONT = Path(__file__).resolve().parents[1] / "assets/fonts/DelaGothicOne-Regular.ttf"
LOCAL_CAPTION_FONT = CAPTION_FONT.with_name("ObelixPro.ttf")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD = 20 * 1024**3
MAX_AD_UPLOAD = 500 * 1024**2


class EditError(Exception):
    """An actionable error that can be displayed in the local UI."""


@dataclass(frozen=True)
class Media:
    duration: float
    width: int
    height: int
    audio: bool
    video_codec: str = ""
    audio_codec: str = ""
    audio_duration: float | None = None


def executable(name: str) -> str:
    bundled = Path(__file__).resolve().parents[1] / "bin" / name
    if bundled.is_file() and not shutil.which(name):
        return str(bundled)
    return shutil.which(name) or name


def run_media(args: list[str], *, timeout: float = 240, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except FileNotFoundError as exc:
        raise EditError(f"Установите {Path(args[0]).name} и повторите запуск.") from exc
    except subprocess.TimeoutExpired as exc:
        raise EditError(f"{Path(args[0]).name} не завершился за отведённое время.") from exc
    if result.returncode:
        detail = "\n".join(result.stderr.strip().splitlines()[-3:])[-500:]
        raise EditError(f"{Path(args[0]).name}: {detail or 'ошибка обработки файла'}")
    return result.stdout


def ffmpeg(*args: str, timeout: float = 240, cwd: Path | None = None) -> str:
    return run_media(
        [executable("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args],
        timeout=timeout, cwd=cwd,
    )


def probe(path: Path, *, require_video: bool = True) -> Media:
    data = json.loads(run_media([
        executable("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ], timeout=30))
    streams = data.get("streams", [])
    video = next((s for s in streams if s["codec_type"] == "video"
                  and not s.get("disposition", {}).get("attached_pic")), {})
    audio = next((s for s in streams if s["codec_type"] == "audio"), {})
    stream = video if require_video else audio
    try:
        duration = float(stream.get("duration", "N/A"))
    except (ValueError, TypeError):
        duration = float(data.get("format", {}).get("duration", 0))
    if not stream or not math.isfinite(duration) or duration <= 0:
        raise EditError("Файл не содержит читаемой дорожки нужного типа.")
    width, height = video.get("width", 0), video.get("height", 0)
    for entry in video.get("side_data_list", []):
        if entry.get("rotation", 0) % 180:
            width, height = height, width
            break
    try:
        audio_duration = float(audio["duration"])
    except (KeyError, ValueError, TypeError):
        audio_duration = None
    return Media(duration, width, height, bool(audio), video.get("codec_name", ""),
                 audio.get("codec_name", ""), audio_duration)


def twitch_id(url: str) -> str:
    try:
        parsed = urlsplit(url.strip())
        match = re.fullmatch(r"/videos/([0-9]{1,20})/?", parsed.path)
        if (parsed.scheme == "https" and parsed.netloc.lower() in {"twitch.tv", "www.twitch.tv"}
                and match):
            return match.group(1)
    except ValueError:
        pass
    raise EditError("Нужна ссылка на запись Twitch вида https://www.twitch.tv/videos/123456.")


def fetch_vod(video_id: str, folder: Path) -> Path:
    source = folder / "source.mp4"
    marker = folder / "download.complete"
    if source.is_file() and marker.is_file():
        return source
    url = f"https://www.twitch.tv/videos/{video_id}"
    command = [
        sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist", "--continue",
        "--check-formats", "--retries", "3", "--fragment-retries", "4",
        "--abort-on-unavailable-fragments", "--no-progress", "--socket-timeout", "25",
        "-f", "b[ext=mp4][height<=1080]/b[height<=1080]/b",
        "--max-filesize", "20G", "--remux-video", "mp4",
        "-o", str(folder / "source.%(ext)s"), "--", url,
    ]
    if shutil.disk_usage(folder).free < 1024**3:
        raise EditError("Для загрузки нужно освободить место на диске.")
    try:
        run_media(command, timeout=8 * 3600)
    except EditError as exc:
        raise EditError(f"Запись Twitch не скачалась. Проверьте доступность VOD и повторите. {exc}") from exc
    if not source.is_file():
        raise EditError("Twitch не вернул видео. Проверьте доступ к записи или загрузите локальный файл.")
    probe(source)
    marker.touch()
    return source


def cloud_window(source: Path, start: float, end: float, path: Path, *, dense: bool = False) -> None:
    height = 720 if dense else 540
    ffmpeg("-ss", str(start), "-i", str(source), "-t", str(end - start),
           "-map", "0:v:0", "-map", "0:a:0?", "-vf", f"scale=-2:'min(ih,{height})',setsar=1",
           "-c:v", "libx264", "-threads", "3", "-preset", "ultrafast", "-crf", "25",
           "-maxrate", "3500k", "-bufsize", "7000k", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-ac", "2", "-b:a", "96k", "-movflags", "+faststart", str(path))


def has_visual_change(source: Path, start: float, end: float) -> bool:
    """Detect frozen, silent candidates without decoding the rest of the VOD."""
    import numpy as np
    raw = subprocess.run([executable("ffmpeg"), "-nostdin", "-v", "error", "-ss", str(start),
                          "-i", str(source), "-t", str(end-start), "-an",
                          "-vf", "fps=2,scale=96:54,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, timeout=60, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 96*54).astype(np.int16)
    return len(frames) > 1 and np.abs(np.diff(frames, axis=0)).mean(axis=1).max() >= .5


def still(source: Path, at: float, target: Path) -> None:
    ffmpeg("-ss", str(at), "-i", str(source), "-frames:v", "1",
           "-vf", "scale='min(1280,iw)':-2,setsar=1", "-update", "1", str(target))
    if at and not target.is_file():
        still(source, 0, target)
    if not target.is_file():
        raise EditError("Исходный видеокадр не читается.")


def wav(source: Path, target: Path, start: float = 0, end: float | None = None) -> None:
    length = ["-t", str(end - start)] if end is not None else []
    ffmpeg("-ss", str(start), "-i", str(source), *length, "-vn",
           "-af", "highpass=f=85,dynaudnorm",
           "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(target))


def frames(seconds: float) -> int:
    return round(seconds * FPS)


def camera_box(box: list[float]) -> list[float]:
    if (len(box) != 4 or any(not math.isfinite(x) for x in box)
            or min(box) < 0 or min(box[2:]) < 0.025
            or box[0] + box[2] > 1.000001 or box[1] + box[3] > 1.000001):
        raise EditError("Выделите прямоугольник вебки внутри кадра.")
    return [round(n, 5) for n in box]


def camera_layout(clip: dict, total: int) -> list[tuple[int, int, list[float] | None]]:
    if clip.get("camera_mode") == "manual":
        return [(0, total, camera_box(clip["camera_box"]))]
    if clip.get("camera_mode") == "off":
        return [(0, total, None)]
    parts = []
    cursor = 0
    for region in sorted(clip.get("cameras", []), key=lambda span: span["start"]):
        lo = min(total, max(cursor, frames(region["start"] - clip["start"])))
        hi = min(total, frames(region["end"] - clip["start"]))
        if hi <= lo:
            continue
        if lo > cursor:
            parts.append((cursor, lo, None))
        parts.append((lo, hi, camera_box(region["box"])))
        cursor = hi
    if cursor < total:
        parts.append((cursor, total, None))
    return parts


def choose_ad_point(clip: dict) -> float | None:
    length = clip["end"] - clip["start"]
    turns = clip.get("turns", [])
    gaps = []
    occupied_until = 0.0
    for turn in turns:
        if turn["start"] - occupied_until >= 0.16:
            gaps.append((occupied_until, turn["start"]))
        occupied_until = max(occupied_until, turn["end"])
    if length - occupied_until >= 0.16:
        gaps.append((occupied_until, length))
    peak = clip["peak"] - clip["start"]
    safe = []
    for start, end in gaps:
        middle = frames((start + end) / 2)
        for lo, hi in ((3.0, peak - 1.2), (peak + 1.2, length - 2.0)):
            first = math.ceil(max(start + .07, lo, 3.0) * FPS)
            last = math.floor(min(end - .07, hi, length - 2.0) * FPS)
            if first <= last:
                safe.append(max(first, min(last, middle)))
    return min(safe, key=lambda point: abs(point / FPS - length * .55)) / FPS if safe else None


def _ass_escape(text: str) -> str:
    return text.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", r"\N")


def write_ass(path: Path, cues: list[dict], split: float | None = None, ad_length: float = 0, *,
              positions: list[tuple[float, float, int]] | None = None) -> None:
    if LOCAL_CAPTION_FONT.is_file():
        font, size, scale_x, scale_y, angle, outline = "Obelix Pro", 64, 100, 89, 0, 5
    else:
        font, size, scale_x, scale_y, angle, outline = "Dela Gothic One", 79, 88, 100, 1.5, 7
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Subtitle,{font},{size},&H001ADCFF,&H001ADCFF,&H00000000,&H99000000,0,0,0,0,{scale_x},{scale_y},0,{angle},1,{outline},2,5,110,110,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for cue in cues:
        text = _ass_escape(cue["text"].upper())
        for begin, finish, y in positions or [(0, float("inf"), 1490)]:
            start, end = max(begin, cue["start"]), min(finish, cue["end"])
            chunks = [(start, end)]
            if split is not None:
                chunks = [(start, min(end, split)), (max(start, split) + ad_length, end + ad_length)]
            for lo, hi in chunks:
                if hi - lo >= 0.025:
                    lines.append(f"Dialogue: 0,{format_ass_time(lo)},{format_ass_time(hi)},"
                                 rf"Subtitle,,0,0,0,,{{\pos(540,{y})}}{text}")
    path.write_text("\n".join(lines), encoding="utf-8")


def voice_track(parts: list[dict], total: float, target: Path) -> None:
    """Build a mono voice bed with exact offsets at 48 kHz."""
    import wave

    count = max(round(total * RATE), frames(total) * SAMPLES_PER_FRAME)
    timeline = bytearray(count * 2)
    previous_end = 0
    for part in parts:
        start = round(part["start"] * RATE)
        stop = min(count, round(part["end"] * RATE))
        if start < previous_end or stop <= start:
            raise EditError("Русские реплики накладываются. Проверьте их границы в предпросмотре.")
        with wave.open(str(part["path"]), "rb") as track:
            if (track.getframerate(), track.getnchannels(), track.getsampwidth()) != (RATE, 1, 2):
                raise EditError("Озвучка имеет неожиданный аудиоформат.")
            audio = track.readframes(track.getnframes())
        audio_samples = len(audio) // 2
        slot_samples = stop - start
        if audio_samples > slot_samples:
            if audio_samples - slot_samples <= 96:
                audio = audio[:slot_samples * 2]
            else:
                raise EditError("Русская реплика не помещается. Сократите текст и повторите предпросмотр.")
        timeline[start * 2:start * 2 + len(audio)] = audio
        previous_end = stop
    with wave.open(str(target), "wb") as output:
        output.setparams((1, 2, RATE, count, "NONE", "not compressed"))
        output.writeframes(timeline)


def build_ducking_volume_expr(intervals: list[tuple[float, float]], base_vol: float = 0.78,
                              duck_vol: float = 0.16, fade: float = 0.10) -> str:
    """Build an FFmpeg volume expression that smoothly ducks audio during active intervals."""
    if not intervals:
        return f"{base_vol:.2f}"

    merged: list[list[float]] = []
    for s, e in sorted(intervals):
        if not merged:
            merged.append([float(s), float(e)])
        elif s <= merged[-1][1] + (fade * 2.0):
            merged[-1][1] = max(merged[-1][1], float(e))
        else:
            merged.append([float(s), float(e)])

    diff = base_vol - duck_vol
    expr = f"{base_vol:.2f}"
    for s, e in reversed(merged):
        s_fade = max(0.0, s - fade)
        e_fade = e + fade
        ramp_down = f"({base_vol:.2f}-{diff:.2f}*(t-{s_fade:.3f})/{fade:.3f})"
        ramp_up = f"({duck_vol:.2f}+{diff:.2f}*(t-{e:.3f})/{fade:.3f})"
        expr = (
            f"if(between(t,{s_fade:.3f},{s:.3f}),{ramp_down},"
            f"if(between(t,{s:.3f},{e:.3f}),{duck_vol:.2f},"
            f"if(between(t,{e:.3f},{e_fade:.3f}),{ramp_up},{expr})))"
        )
    return expr


def _fit_stream(graph: list[str], source: str, output: str, width: int, height: int) -> None:
    small_w, small_h = max(2, width // 8 * 2), max(2, height // 8 * 2)
    graph.extend([
        f"[{source}]split=2[{output}soft][{output}sharp]",
        f"[{output}soft]scale={small_w}:{small_h}:force_original_aspect_ratio=increase,"
        f"crop={small_w}:{small_h},boxblur=10:2,eq=brightness=-0.16:saturation=0.8,"
        f"scale={width}:{height},setsar=1[{output}background]",
        f"[{output}sharp]scale={width}:{height}:force_original_aspect_ratio=decrease:"
        f"force_divisible_by=2,setsar=1[{output}foreground]",
        f"[{output}background][{output}foreground]overlay=(W-w)/2:(H-h)/2[{output}]",
    ])


def _camera_crop(source: Media, box: list[float]) -> tuple[int, int, int, int]:
    width, height = source.width // 2 * 2, source.height // 2 * 2
    x, y, w, h = box
    left, top = math.floor(x * width / 2) * 2, math.floor(y * height / 2) * 2
    right = min(width, math.ceil((x + w) * width / 2) * 2)
    bottom = min(height, math.ceil((y + h) * height / 2) * 2)
    return right - left, bottom - top, left, top


def _gameplay_crop(source: Media, camera: tuple[int, int, int, int],
                   aspect: float) -> tuple[int, int, int, int]:
    width, height = source.width // 2 * 2, source.height // 2 * 2
    cam_w, cam_h, left, top = camera
    right, bottom = left + cam_w, top + cam_h
    side = min(right, width - left)
    candidates = [(max(0, width - 2 * side), height, side, 0),
                  (left, height, 0, 0), (width - right, height, right, 0),
                  (width, top, 0, 0), (width, height - bottom, 0, bottom)]
    # Maximize the source pixels left after fitting the gameplay panel. Break ties
    # in favor of the symmetric crop so center-screen gameplay stays centered.
    return max(candidates, key=lambda region: min(region[0], region[1] * aspect))


def _layout(graph: list[str], clip: dict, total: int, width: int, height: int,
            source: Media) -> list[tuple[float, float, int]]:
    parts = camera_layout(clip, total)
    positions = []
    band = CAPTION_BAND * height // 1920
    base = "[0:v]setpts=PTS-STARTPTS,fps=30:start_time=0,tpad=stop_mode=clone:stop=1"
    if len(parts) > 1:
        graph.append(base + f",split={len(parts)}" + "".join(f"[p{i}]" for i in range(len(parts))))
    else:
        graph.append(base + "[p0]")
    for i, (lo, hi, box) in enumerate(parts):
        section = f"[p{i}]trim=start_frame={lo}:end_frame={hi},setpts=PTS-STARTPTS"
        if box is None:
            graph.append(section + f"[gameinput{i}]")
            _fit_stream(graph, f"gameinput{i}", f"v{i}", width, height)
            positions.append((lo / FPS, hi / FPS, 1490))
            continue
        camera = _camera_crop(source, box)
        # Fit the webcam to the full width. Limit very tall cameras by cropping vertically.
        camera_height = round(width * camera[1] / camera[0] / 2) * 2
        top = max(2, min(camera_height, int(height * .45) // 2 * 2))
        bottom = height - top - band
        gameplay = _gameplay_crop(source, camera, width / bottom)
        if not gameplay[0] or not gameplay[1]:
            raise EditError("Выделите только область вебки, чтобы оставить место для игрового кадра.")
        camera_crop = ":".join(map(str, camera))
        game_crop = ":".join(map(str, gameplay))
        positions.append((lo / FPS, hi / FPS, round((top + band / 2) * 1920 / height)))
        graph.extend([
            section + f",split=2[cs{i}][gs{i}]",
            f"[cs{i}]crop={camera_crop},scale={width}:-2:flags=lanczos,crop={width}:{top},setsar=1[cam{i}]",
            f"[gs{i}]crop={game_crop},split=2[clean{i}][soft{i}]",
            f"[clean{i}]scale={width}:{bottom}:flags=lanczos:force_original_aspect_ratio=increase:"
            f"force_divisible_by=2,crop={width}:{bottom},setsar=1[game{i}]",
            f"[soft{i}]scale=54:32,boxblur=7:1,eq=brightness=-0.08,"
            f"scale={width}:{band},setsar=1[band{i}]",
            f"[cam{i}][band{i}][game{i}]vstack=inputs=3[v{i}]",
        ])
    if len(parts) == 1:
        graph.append("[v0]null[base]")
    else:
        graph.append("".join(f"[v{i}]" for i in range(len(parts))) +
                     f"concat=n={len(parts)}:v=1:a=0[base]")
    return positions


def render(source: Path, clip: dict, source_has_audio: bool, work: Path, target: Path,
           *, voice: Path | None = None, cues: list[dict] | None = None, ad: Path | None = None,
           ad_info: Media | None = None, preview: bool = False) -> float:
    work.mkdir(parents=True, exist_ok=True)
    size = (540, 960) if preview else (1080, 1920)
    width, height = size
    total = frames(clip["end"] - clip["start"])
    seconds = total / FPS
    inputs = ["-ss", str(clip["start"]), "-threads", "2", "-i", str(source)]
    if voice:
        inputs.extend(["-i", str(voice)])
    if ad:
        inputs.extend(["-threads", "2", "-i", str(ad)])
    ad_index = 2 if voice else 1
    split = frames(clip["ad_at"]) if ad else None
    ad_frames = max(1, frames(ad_info.duration)) if ad and ad_info else 0
    if ad and (split is None or not 0 < split < total or not ad_frames):
        raise EditError("Укажите точку рекламной паузы внутри клипа.")
    graph: list[str] = []
    positions = _layout(graph, clip, total, width, height, probe(source))
    sound = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    span = total * SAMPLES_PER_FRAME
    source_audio = "[0:a]asetpts=PTS-STARTPTS," if source_has_audio else "anullsrc=r=48000:cl=stereo,"
    graph.append(f"{source_audio}{sound},apad,atrim=end_sample={span}[original]")
    if voice:
        turns = clip.get("turns") or []
        turn_intervals = [(t["start"], t["end"]) for t in turns if t.get("russian")]
        duck_expr = build_ducking_volume_expr(turn_intervals, base_vol=0.78, duck_vol=0.16, fade=0.10)
        graph.extend([
            f"[1:a]{sound},asplit=2[key][dub]",
            f"[original]volume=eval=frame:volume='{duck_expr}'[quiet]",
            "[quiet][key]sidechaincompress=threshold=0.015:ratio=16:attack=15:release=320[duck]",
            "[duck][dub]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.93:level=0:latency=1[mix]",
        ])
    else:
        graph.append("[original]anull[mix]")
    v, a = "base", "mix"
    if split is not None:
        at = split * SAMPLES_PER_FRAME
        max_w, max_h = int(width * 0.78) // 2 * 2, int(height * 0.62) // 2 * 2
        graph.extend([
            "[base]split=3[before][freeze][after]",
            f"[before]trim=end_frame={split},setpts=PTS-STARTPTS[v0]",
            f"[freeze]trim=start_frame={split-1}:end_frame={split},setpts=PTS-STARTPTS,"
            f"tpad=stop_mode=clone:stop={ad_frames-1}[hold]",
            f"[{ad_index}:v]setpts=PTS-STARTPTS,fps=30:start_time=0,"
            f"scale={max_w}:{max_h}:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1,"
            f"tpad=stop_mode=clone:stop=1,trim=end_frame={ad_frames}[advert]",
            "[hold][advert]overlay=(W-w)/2:(H-h)/2[v1]",
            f"[after]trim=start_frame={split}:end_frame={total},setpts=PTS-STARTPTS[v2]",
            "[mix]asplit=2[pre][post]",
            f"[pre]atrim=end_sample={at},asetpts=PTS-STARTPTS[a0]",
            f"[post]atrim=start_sample={at}:end_sample={span},asetpts=PTS-STARTPTS[a2]",
            f"[{ad_index}:a]asetpts=PTS-STARTPTS,loudnorm=I=-16:TP=-1.5:LRA=9,"
            f"{sound},apad,atrim=end_sample={ad_frames*SAMPLES_PER_FRAME}[a1]",
            "[v0][v1][v2]concat=n=3:v=1:a=0[edit]",
            "[a0][a1][a2]concat=n=3:v=0:a=1[aedit]",
        ])
        v, a = "edit", "aedit"
    if cues:
        fonts = work / "caption-fonts"
        fonts.mkdir(exist_ok=True)
        shutil.copyfile(CAPTION_FONT, fonts / CAPTION_FONT.name)
        if LOCAL_CAPTION_FONT.is_file():
            shutil.copyfile(LOCAL_CAPTION_FONT, fonts / LOCAL_CAPTION_FONT.name)
        write_ass(work / "captions.ass", cues, split / FPS if split is not None else None, ad_frames / FPS,
                  positions=positions)
        graph.append(f"[{v}]ass=captions.ass:fontsdir=caption-fonts[out]")
        v = "out"
    graph.append(f"[{a}]apad,atrim=end_sample={(total+ad_frames)*SAMPLES_PER_FRAME}[outaudio]")
    duration = seconds + ad_frames / FPS
    tmp = target.with_name(target.stem + ".partial.mp4")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg(*inputs, "-filter_complex_threads", "2", "-filter_complex", ";".join(graph),
               "-map", f"[{v}]", "-map", "[outaudio]", "-r", str(FPS),
               "-frames:v", str(total + ad_frames), "-c:v", "libx264", "-threads", "3",
               "-preset", "veryfast" if preview else "fast", "-crf", "24" if preview else "17", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k", "-ar", str(RATE), "-ac", "2",
               "-movflags", "+faststart", str(tmp), timeout=max(420, duration * 4), cwd=work)
        verify_export(tmp, size, duration)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return duration


def verify_export(path: Path, size: tuple[int, int], duration: float) -> None:
    rendered = probe(path)
    if ((rendered.width, rendered.height) != size or rendered.video_codec != "h264"
            or rendered.audio_codec != "aac" or abs(rendered.duration - duration) > 0.075
            or rendered.audio_duration is None or abs(rendered.audio_duration - duration) > 0.075):
        raise EditError("Часть видео или звука не прошла проверку длительности и формата.")
    ffmpeg("-xerror", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-",
           timeout=max(180, duration * 2))
