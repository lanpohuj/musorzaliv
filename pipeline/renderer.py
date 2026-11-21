"""
Video rendering engine: NVENC 9:16 layout, burned-in subtitles, mid-roll promo.

Невидима унікалізація (обходить перцептивні хеші, для ока відео ідентичне):
  - мікрокроп N пікселів з країв (рандом 2-5px на кліп) + рескейл
  - часовий шум силою 1-2 (noise=alls=N:allf=t) — змінює кожен піксель
  - мікрозсув відтінку (hue ±0.8°) та гамми (±1%)
  - випадковий speed у діапазоні [1.03..1.05] на кліп + джиттер старту
  - нечутний ультразвуковий тон 16-19kHz @ ~-48dB (ламає аудіо-відбиток)
  - випадкові GOP / CQ / бітрейт (±12%) / обрізка хвоста 0-0.12с

Beautify (видимі покращення):
  - плавний розмитий фон вищої роздільної здатності (270x480 -> lanczos)
  - легкий unsharp на геймплеї, делікатна вінєтка
  - аудіо-прибирка: highpass + acompressor перед loudnorm
"""

import os
import random
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from pipeline.transcriber import get_ffmpeg_bin


def escape_ffmpeg_path(path: str) -> str:
    """Escape path for use inside FFmpeg filtergraph strings."""
    clean = os.path.abspath(path).replace("\\", "/")
    return clean.replace(":", "\\:").replace("'", "\\'")


def _parse_bitrate_mbps(br: str) -> float:
    """'6M' / '6000k' -> 6.0 (Мбіт/с)."""
    br = str(br).strip().upper()
    mult = 1.0
    if br.endswith("M"):
        mult, br = 1.0, br[:-1]
    elif br.endswith("K"):
        mult, br = 0.001, br[:-1]
    try:
        return float(br) * mult
    except ValueError:
        return 6.0


def _format_bitrate(mbps: float) -> str:
    return f"{int(round(mbps * 1000))}k"


class Renderer:
    def __init__(self, config: dict):
        self.cfg = config
        self.video_cfg = config.get("video", {})
        self.banner_cfg = config.get("banner", {})
        self.sub_cfg = config.get("subtitles", {})
        self.promo_cfg = config.get("promo_midroll", {})
        self.uni_cfg = self.video_cfg.get("uniquize", {})
        self.beau_cfg = self.video_cfg.get("beautify", {})
        self.audio_cfg = config.get("audio", {})

        self.width = self.video_cfg.get("target_width", 1080)
        self.height = self.video_cfg.get("target_height", 1920)
        self.crop_mode = self.video_cfg.get("crop_mode", "center_crop")
        self.zoom = float(self.video_cfg.get("zoom", 1.08)) # 105-110% zoom for pixel hash & focus
        self.speed = float(self.video_cfg.get("speed", 1.04))
        self.use_nvenc = bool(self.video_cfg.get("use_nvenc", True))
        self.nvenc_preset = self.video_cfg.get("nvenc_preset", "p4")
        self.bitrate = self.video_cfg.get("video_bitrate", "6M")

        # Color grade
        cg = self.video_cfg.get("color_grade", {})
        self.cg_enabled = cg.get("enabled", True)
        self.contrast = cg.get("contrast", 1.03)
        self.brightness = cg.get("brightness", 0.01)
        self.saturation = cg.get("saturation", 1.05)

        # Унікалізація
        self.uni_enabled = bool(self.uni_cfg.get("enabled", True))
        self.tone_enabled = bool(self.uni_cfg.get("audio_tone", True))

        # Beautify
        self.beau_enabled = bool(self.beau_cfg.get("enabled", True))

        # Hook zoom: punch-in effect on first N seconds
        self.hook_zoom_cfg = self.video_cfg.get("hook_zoom", {})
        self.hook_zoom_enabled = bool(self.hook_zoom_cfg.get("enabled", True))
        self.hook_zoom_scale = float(self.hook_zoom_cfg.get("scale", 1.12))
        self.hook_zoom_duration = float(self.hook_zoom_cfg.get("duration", 1.5))

        self.rand = random.Random()

        self.ffmpeg_bin = get_ffmpeg_bin()

    # ------------------------------------------------------------------
    # Параметри унікалізації (рандомляться один раз на кліп)
    # ------------------------------------------------------------------

    def _uniq(self, clip: Dict[str, Any]) -> Dict[str, Any]:
        if "_uniq" in clip:
            return clip["_uniq"]
        u: Dict[str, Any] = {}
        if self.uni_enabled:
            c_lo, c_hi = self.uni_cfg.get("crop_px", [2, 5])
            u["crop_px"] = self.rand.randint(int(min(c_lo, c_hi)), int(max(c_lo, c_hi)))
            n_lo, n_hi = self.uni_cfg.get("noise_strength", [1, 2])
            u["noise"] = self.rand.randint(int(min(n_lo, n_hi)), int(max(n_lo, n_hi)))
            h_lo, h_hi = self.uni_cfg.get("hue_shift", [-0.8, 0.8])
            u["hue"] = round(self.rand.uniform(h_lo, h_hi), 2)
            g_lo, g_hi = self.uni_cfg.get("gamma", [0.99, 1.01])
            u["gamma"] = round(self.rand.uniform(g_lo, g_hi), 3)
            s_lo, s_hi = self.uni_cfg.get("speed_range", [1.03, 1.05])
            lo, hi = min(s_lo, s_hi), max(s_lo, s_hi)
            lo, hi = max(lo, 1.0), hi
            u["speed"] = round(self.rand.uniform(lo, hi), 4)
            if self.tone_enabled:
                f_lo, f_hi = self.uni_cfg.get("tone_freq", [16000, 19000])
                u["tone_freq"] = int(self.rand.uniform(min(f_lo, f_hi), max(f_lo, f_hi)))
                a_lo, a_hi = self.uni_cfg.get("tone_amp", [0.002, 0.005])
                u["tone_amp"] = round(self.rand.uniform(min(a_lo, a_hi), max(a_lo, a_hi)), 5)
            g_lo, g_hi = self.uni_cfg.get("gop_range", [96, 240])
            u["gop"] = self.rand.randint(int(min(g_lo, g_hi)), int(max(g_lo, g_hi)))
            u["cq"] = self.rand.randint(21, 23)
            bj = float(self.uni_cfg.get("bitrate_jitter", 0.12))
            u["bitrate_mult"] = round(self.rand.uniform(1.0 - bj, 1.0 + bj), 3)
            tt = float(self.uni_cfg.get("tail_trim_max", 0.12))
            u["tail_trim"] = round(self.rand.uniform(0.0, max(0.0, tt)), 3)
        clip["_uniq"] = u
        return u

    def get_clip_speed(self, clip: Dict[str, Any]) -> float:
        """Швидкість цього кліпу (з урахуванням унікалізації). Кешується в кліпі."""
        u = self._uniq(clip)
        if u and "speed" in u:
            return u["speed"]
        return self.speed

    def get_uniq_summary(self, clip: Dict[str, Any]) -> str:
        u = self._uniq(clip)
        if not u:
            return "вимкнено"
        tone = f", tone {u.get('tone_freq', '-')}Hz" if u.get("tone_freq") else ""
        return (f"{u['speed']}x, crop {u['crop_px']}px, noise {u['noise']}, "
                f"hue {u['hue']:+.1f}°, gamma {u['gamma']}{tone}")

    # ------------------------------------------------------------------
    # Промо
    # ------------------------------------------------------------------

    def get_promo_video(self) -> Optional[str]:
        """Auto-detect promo video from config or assets directory."""
        if not self.promo_cfg.get("enabled", False):
            return None

        configured = self.promo_cfg.get("video_path")
        if configured and os.path.exists(configured):
            return configured

        candidates = [
            "assets/baner.mp4", "assets/baner.MP4", "assets/promo.mp4",
            "assets/promo.MP4", "assets/banner.mp4", "assets/banner.MP4"
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

        if os.path.isdir("assets"):
            for f in sorted(os.listdir("assets")):
                if Path(f).suffix.lower() in [".mp4", ".mov", ".mkv", ".webm"]:
                    return os.path.join("assets", f)
        return None

    # ------------------------------------------------------------------
    # Побудова фільтрів
    # ------------------------------------------------------------------

    def _layout_filter(self, in_label: str, uid: str, uniq: Dict[str, Any]) -> str:
        """
        Мікрокроп джерела -> 9:16 композиція (blur bg або center crop)
        -> sharpen fg -> вінєтка. uid уніфікує мітки для multi-input графів.
        """
        W, H = self.width, self.height
        u = uniq
        pre = ""
        if u and u.get("crop_px"):
            cp = u["crop_px"]
            pre = f"crop=iw-{2 * cp}:ih-{2 * cp}:{cp}:{cp},"

        beau = self.beau_cfg if self.beau_enabled else {}
        sharp = float(beau.get("sharpen", 0.35))
        use_vignette = bool(beau.get("vignette", True))

        if self.crop_mode == "center_crop":
            # Dynamic 9:16 Crop with 105-110% zoom centering main action/crosshair
            base_zoom = float(self.video_cfg.get("zoom", self.zoom))
            if u and self.uni_enabled:
                zoom = round(self.rand.uniform(1.05, 1.10), 3)
            else:
                zoom = max(1.05, min(1.10, base_zoom))

            chain = (
                f"[{in_label}]{pre}"
                f"crop=trunc(ih*9/16/{zoom}/2)*2:trunc(ih/{zoom}/2)*2:(iw-ow)/2:(ih-oh)/2,"
                f"scale={W}:{H}:flags=lanczos,setsar=1"
            )
            if sharp > 0:
                chain += f",unsharp=5:5:{sharp}:3:3:0"
            if use_vignette:
                chain += ",vignette=angle=PI/4.8"
            return f"{chain}[v_lay_{uid}]"

        # blur_background: фон вищої роздільної здатності -> сильний блюр -> апскейл
        mid_w = int(beau.get("bg_resolution", 270))
        mid_h = mid_w * 16 // 9
        bg_blur = int(beau.get("bg_blur", 10))
        bg_dark = float(beau.get("bg_darken", 0.06))
        bg_sat = float(beau.get("bg_saturation", 1.12))

        sharpen_fg = f",unsharp=5:5:{sharp}:3:3:0" if sharp > 0 else ""
        vign = ",vignette=angle=PI/4.8" if use_vignette else ""

        return (
            f"[{in_label}]{pre}split=2[fg_{uid}][bg_{uid}];"
            f"[bg_{uid}]scale={mid_w}:{mid_h}:force_original_aspect_ratio=increase,"
            f"crop={mid_w}:{mid_h},boxblur={bg_blur}:2,"
            f"scale={W}:{H}:flags=lanczos,"
            f"eq=brightness=-{bg_dark}:saturation={bg_sat}[bgd_{uid}];"
            f"[fg_{uid}]scale={W}:-2:flags=lanczos{sharpen_fg}[fgs_{uid}];"
            f"[bgd_{uid}][fgs_{uid}]overlay=0:(H-h)/2,setsar=1{vign}[v_lay_{uid}]"
        )

    def _grade_chain(self, uniq: Dict[str, Any]) -> str:
        """Кольорокорекція + мікро hue/gamma унікалізації (без setpts)."""
        parts = []
        if self.cg_enabled:
            parts.append(f"eq=contrast={self.contrast}:brightness={self.brightness}:saturation={self.saturation}")
        if uniq and uniq.get("gamma"):
            parts.append(f"eq=gamma={uniq['gamma']}")
        if uniq and uniq.get("hue"):
            parts.append(f"hue=h={uniq['hue']}")
        return ",".join(parts)

    def _audio_cleanup(self) -> str:
        """Прибирка голосу: зріз гулу + м'яка компресія для «панчі»."""
        if not bool(self.audio_cfg.get("enhance", True)):
            return ""
        hp = int(self.audio_cfg.get("highpass", 70))
        comp = self.audio_cfg.get("compressor", "threshold=-20dB:ratio=2.5:attack=10:release=150:makeup=1.5")
        return f"highpass=f={hp},acompressor={comp},"

    def _audio_final_chain(self, norm_label: str, uniq: Dict[str, Any]) -> str:
        """loudnorm + (опційно) нечутний унікальний тон -> [a_final]."""
        aformat = "aformat=sample_rates=44100:channel_layouts=stereo"
        norm = f"[{norm_label}]loudnorm=I=-14:TP=-1.5:LRA=11,{aformat}"
        if uniq and uniq.get("tone_freq") and uniq.get("tone_amp"):
            f = uniq["tone_freq"]
            amp = uniq["tone_amp"]
            return (
                f"sine=frequency={f}:sample_rate=44100,volume={amp},{aformat}[tone_u];"
                f"{norm}[an_u];"
                f"[an_u][tone_u]amix=inputs=2:duration=first:normalize=0[a_final]"
            )
        return f"{norm}[a_final]"

    # ------------------------------------------------------------------
    # Рендер
    # ------------------------------------------------------------------

    def render_clip(
        self,
        input_video: str,
        clip: Dict[str, Any],
        output_path: str,
        banner_path: Optional[str] = None,
        ass_subtitles_path: Optional[str] = None,
        ass_subtitles_split: Optional[tuple] = None
    ) -> bool:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        start_time = max(0.0, float(clip["start"]))
        duration = float(clip["duration"])

        if self.use_nvenc:
            try:
                success = self._run_ffmpeg(
                    input_video, start_time, duration, output_path,
                    banner_path, ass_subtitles_path, ass_subtitles_split, encoder="nvenc", clip=clip
                )
                if success:
                    return True
            except Exception as e:
                print(f"[!] NVENC помилка ({e}). Перемикання на процесорний рендер (libx264)...")

        return self._run_ffmpeg(
            input_video, start_time, duration, output_path,
            banner_path, ass_subtitles_path, ass_subtitles_split, encoder="cpu", clip=clip
        )

    def _run_ffmpeg(
        self,
        input_video: str,
        start_time: float,
        duration: float,
        output_path: str,
        banner_path: Optional[str],
        ass_subtitles_path: Optional[str],
        ass_subtitles_split: Optional[tuple],
        encoder: str = "nvenc",
        clip: Optional[Dict[str, Any]] = None
    ) -> bool:
        clip = clip or {}
        uniq = self._uniq(clip) if self.uni_enabled else {}
        speed = self.get_clip_speed(clip) if uniq else self.speed
        tail_trim = uniq.get("tail_trim", 0.0)

        promo_video = self.get_promo_video()
        if promo_video:
            return self._run_ffmpeg_midroll(
                input_video, start_time, duration, promo_video, output_path,
                ass_subtitles_split, encoder, uniq, speed, tail_trim
            )

        # Standard processing without promo
        input_args = [
            "-ss", f"{start_time:.3f}",
            "-t", f"{duration:.3f}",
            "-i", input_video
        ]

        v_filters = [self._layout_filter("0:v", "0", uniq)]
        last_v = "v_lay_0"

        chain = [f"setpts=PTS/{speed}"] if abs(speed - 1.0) > 0.001 else []
        grade = self._grade_chain(uniq)
        if grade:
            chain.append(grade)
        if chain:
            v_filters.append(f"[{last_v}]{','.join(chain)}[v_graded]")
            last_v = "v_graded"

        if uniq and uniq.get("noise"):
            v_filters.append(f"[{last_v}]noise=alls={uniq['noise']}:allf=t[v_noised]")
            last_v = "v_noised"

        # Hook zoom: punch-in jump zoom on first N seconds for instant visual movement
        eff_dur = duration / speed
        if self.hook_zoom_enabled and eff_dur > (self.hook_zoom_duration / speed + 1.0):
            hz_scale = self.hook_zoom_scale
            hz_dur = self.hook_zoom_duration / speed
            W, H = self.width, self.height
            v_filters.append(
                f"[{last_v}]split=2[v_hz_in1][v_hz_in2];"
                f"[v_hz_in1]trim=duration={hz_dur:.3f},setpts=PTS-STARTPTS,"
                f"crop=w=trunc(iw/{hz_scale:.4f}/2)*2:h=trunc(ih/{hz_scale:.4f}/2)*2:x=(iw-ow)/2:y=(ih-oh)/2,"
                f"scale={W}:{H}:flags=lanczos,setsar=1[v_hz_zoomed];"
                f"[v_hz_in2]trim=start={hz_dur:.3f},setpts=PTS-STARTPTS[v_hz_rest];"
                f"[v_hz_zoomed][v_hz_rest]concat=n=2:v=1:a=0[v_hookzoom]"
            )
            last_v = "v_hookzoom"

        if ass_subtitles_path and os.path.exists(ass_subtitles_path) and self.sub_cfg.get("enabled", True):
            escaped_ass = escape_ffmpeg_path(ass_subtitles_path)
            v_filters.append(f"[{last_v}]ass='{escaped_ass}'[v_final]")
            last_v = "v_final"
        else:
            v_filters.append(f"[{last_v}]null[v_final]")
            last_v = "v_final"

        cleanup = self._audio_cleanup()
        a_pre = f"[0:a]{cleanup}atempo={speed},loudnorm=I=-14:TP=-1.5:LRA=11[a_pre]"
        a_final = self._audio_final_chain("a_pre", uniq)
        filter_complex = ";".join(v_filters) + ";" + a_pre + ";" + a_final

        expected_out_dur = (duration / speed) + 0.1 - tail_trim
        return self._execute_ffmpeg(
            input_args, filter_complex, last_v, "a_final",
            expected_out_dur, output_path, encoder, uniq
        )

    def _run_ffmpeg_midroll(
        self,
        input_video: str,
        start_time: float,
        duration: float,
        promo_video: str,
        output_path: str,
        ass_subtitles_split: Optional[tuple],
        encoder: str = "nvenc",
        uniq: Optional[Dict[str, Any]] = None,
        speed: float = 1.04,
        tail_trim: float = 0.0
    ) -> bool:
        """
        Part 1 (0 -> mid) -> Promo (1.00x + оригінальний звук) -> Part 2 (mid -> end).
        """
        uniq = uniq or {}
        mid_dur = duration / 2.0
        part2_start = start_time + mid_dur
        part2_dur = duration - mid_dur

        input_args = [
            "-ss", f"{start_time:.3f}", "-t", f"{mid_dur:.3f}", "-i", input_video,
            "-i", promo_video,
            "-ss", f"{part2_start:.3f}", "-t", f"{part2_dur:.3f}", "-i", input_video
        ]

        cleanup = self._audio_cleanup()
        grade = self._grade_chain(uniq)
        noise_f = f",noise=alls={uniq['noise']}:allf=t" if uniq.get("noise") else ""

        def build_gameplay(in_v: str, uid: str, out_v: str) -> str:
            inner = f"setpts=PTS/{speed}"
            if grade:
                inner += f",{grade}"
            return (
                f"{self._layout_filter(in_v, uid, uniq)};"
                f"[v_lay_{uid}]{inner}{noise_f},setsar=1[{out_v}]"
            )

        filter_parts = [
            build_gameplay("0:v", "0", "v0_raw"),
            f"[0:a]{cleanup}atempo={speed},aformat=sample_rates=44100:channel_layouts=stereo[a0]",

            # Mid-roll Banner: 1.00x + оригінальний звук
            f"[1:v]split=2[fg_pr][bg_pr];"
            f"[bg_pr]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=10:2,"
            f"scale={self.width}:{self.height}:flags=lanczos,"
            f"eq=brightness=-0.06:saturation=1.12[bgd_pr];"
            f"[fg_pr]scale={self.width}:-2:flags=lanczos[fgs_pr];"
            f"[bgd_pr][fgs_pr]overlay=0:(H-h)/2,setsar=1[v1]",
            f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo[a1]",

            build_gameplay("2:v", "2", "v2_raw"),
            f"[2:a]{cleanup}atempo={speed},aformat=sample_rates=44100:channel_layouts=stereo[a2]",
        ]

        # Hook zoom на початку кліпу (перша частина gameplay)
        v0_base = "v0_raw"
        eff_p1_dur = mid_dur / speed
        if self.hook_zoom_enabled and eff_p1_dur > (self.hook_zoom_duration / speed + 1.0):
            hz_scale = self.hook_zoom_scale
            hz_dur = self.hook_zoom_duration / speed
            W, H = self.width, self.height
            filter_parts.append(
                f"[v0_raw]split=2[v0_hz1][v0_hz2];"
                f"[v0_hz1]trim=duration={hz_dur:.3f},setpts=PTS-STARTPTS,"
                f"crop=w=trunc(iw/{hz_scale:.4f}/2)*2:h=trunc(ih/{hz_scale:.4f}/2)*2:x=(iw-ow)/2:y=(ih-oh)/2,"
                f"scale={W}:{H}:flags=lanczos,setsar=1[v0_hz_z];"
                f"[v0_hz2]trim=start={hz_dur:.3f},setpts=PTS-STARTPTS[v0_hz_rest];"
                f"[v0_hz_z][v0_hz_rest]concat=n=2:v=1:a=0[v0_hookzoom]"
            )
            v0_base = "v0_hookzoom"

        # Субтитри на частини 1 і 2 (банер — чистий екран)
        v0_target, v2_target = v0_base, "v2_raw"
        if ass_subtitles_split and self.sub_cfg.get("enabled", True):
            p1_ass, p2_ass = ass_subtitles_split
            if p1_ass and os.path.exists(p1_ass):
                filter_parts.append(f"[{v0_base}]ass='{escape_ffmpeg_path(p1_ass)}'[v0]")
            else:
                filter_parts.append(f"[{v0_base}]null[v0]")
            if p2_ass and os.path.exists(p2_ass):
                filter_parts.append(f"[v2_raw]ass='{escape_ffmpeg_path(p2_ass)}'[v2]")
            else:
                filter_parts.append("[v2_raw]null[v2]")
            v0_target, v2_target = "v0", "v2"
        else:
            filter_parts.append(f"[{v0_base}]null[v0]")
            filter_parts.append("[v2_raw]null[v2]")
            v0_target, v2_target = "v0", "v2"

        filter_parts.append(f"[{v0_target}][a0][v1][a1][{v2_target}][a2]concat=n=3:v=1:a=1[v_concat][a_concat]")
        filter_parts.append(self._audio_final_chain("a_concat", uniq))

        filter_complex = ";".join(filter_parts)

        # (gameplay / speed) + тривалість банера + запас
        expected_out_dur = (duration / speed) + 10.0 - tail_trim

        return self._execute_ffmpeg(
            input_args, filter_complex, "v_concat", "a_final",
            expected_out_dur, output_path, encoder, uniq
        )

    def _execute_ffmpeg(
        self,
        input_args: list,
        filter_complex: str,
        v_map: str,
        a_map: str,
        duration: float,
        output_path: str,
        encoder: str,
        uniq: Optional[Dict[str, Any]] = None
    ) -> bool:
        uniq = uniq or {}
        gop = uniq.get("gop", 120)
        gop_args = ["-g", str(gop)] if gop else []

        if encoder == "nvenc":
            cq = uniq.get("cq", 22)
            mbps = _parse_bitrate_mbps(self.bitrate) * uniq.get("bitrate_mult", 1.0)
            b_v = _format_bitrate(mbps)
            maxrate = _format_bitrate(mbps * 1.35)
            bufsize = _format_bitrate(mbps * 2.0)
            video_enc_args = [
                "-c:v", "h264_nvenc",
                "-preset", self.nvenc_preset,
                "-rc", "vbr",
                "-cq", str(cq),
                "-b:v", b_v,
                "-maxrate", maxrate,
                "-bufsize", bufsize,
                "-pix_fmt", "yuv420p"
            ] + gop_args
        else:
            crf = 22 + (1 if uniq.get("cq") == 23 else 0) - (1 if uniq.get("cq") == 21 else 0)
            video_enc_args = [
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", str(max(18, min(26, crf))),
                "-pix_fmt", "yuv420p"
            ] + gop_args

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-loglevel", "warning"
        ] + input_args + [
            "-filter_complex", filter_complex,
            "-map", f"[{v_map}]",
            "-map", f"[{a_map}]"
        ] + video_enc_args + [
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", f"{max(1.0, duration):.3f}",
            "-map_metadata", "-1",
            output_path
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"[!] FFmpeg error: {result.stderr.strip()[-800:]}")
            return False

        return True
