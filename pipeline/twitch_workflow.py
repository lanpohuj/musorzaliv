"""Persisted workflow behind the local Twitch editor."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import logging
import shutil
from tempfile import TemporaryDirectory
from pathlib import Path
import re
from threading import RLock
import traceback
from uuid import uuid4

from . import twitch_editor as editor
from . import twitch_media as media

ROOT = Path(__file__).resolve().parents[1]
JOB_ROOT = ROOT / "temp" / "twitch"
OUTPUT_ROOT = ROOT / "output_clips"
JOB_ID = re.compile(r"[a-f0-9]{12}")
CLIP_ID = re.compile(r"[a-f0-9]{20}")

BUSY = {
    "load": "loading",
    "analyze": "analyzing",
    "script": "translating",
    "preview": "preparing",
    "export": "exporting",
}
NEXT = {
    "load": {"created"},
    "analyze": {"source_ready"},
    "script": {"review", "scripted", "preview"},
    "preview": {"scripted", "preview"},
    "export": {"preview"},
}
LOG = logging.getLogger(__name__)


class Job:
    def __init__(self, folder: Path, output_root: Path):
        self.folder = folder
        self.id = folder.name
        self.record = folder / "job.json"
        self.output = output_root / f"twitch-{self.id}"

    def read(self) -> dict:
        data = editor.read_json(self.record)
        if data is None:
            raise media.EditError("Задача не найдена.")
        return data

    def update(self, **changes) -> dict:
        state = self.read()
        state.update(changes)
        editor.write_json(self.record, state)
        return state

    @property
    def source(self) -> Path:
        return self.folder / self.read()["source"]

    def progress(self, message: str, done: int, total: int) -> None:
        self.update(progress={"message": message, "done": done, "total": total})

    def prepare_source(self) -> None:
        state = self.read()
        self.progress("Загрузка записи Twitch…" if state["vod"] else "Проверка файла…", 0, 1)
        if state["vod"]:
            path = media.fetch_vod(state["vod"], self.folder)
        else:
            if not state.get("source"):
                raise media.EditError("Выберите локальный видеофайл.")
            path = self.source
        info = media.probe(path)
        still = self.folder / "source.jpg"
        media.still(path, min(info.duration / 2, 15), still)
        self.update(source=path.name, source_info=asdict(info), estimate=editor.estimate(info.duration, twitch_vod=bool(state["vod"])),
                    status="source_ready", progress=None)

    def _cloud_cache(self, key: list, result_type, create) -> tuple:
        path = self.folder / f"ai-{editor.stable_id([editor.EDITOR_VERSION, editor.analysis_config()[:2], key])}.json"
        cached = editor.read_json(path)
        if cached is None:
            result, usage = create()
            editor.write_json(path, {"data": result.model_dump(), "usage": usage})
            return result, usage
        return result_type.model_validate(cached["data"]), cached["usage"]

    def analyze(self) -> None:
        source = self.source
        state = self.read()
        source_info = state["source_info"]
        realtime = "live_origin" in state
        length = source_info["duration"]
        contextual_review = not realtime and (bool(state["vod"]) or length >= 20 * 60)
        windows = editor.window_ranges(length)
        model = editor.create_editor()
        speech = editor.LocalSpeech()
        analysis_proxy = self.folder / "analysis.partial.mp4"
        events, refined = [], []
        usage = {"input": 0, "output": 0}

        def collect(tokens):
            for key in usage:
                usage[key] += tokens[key]

        def scan(start, end):
            media.cloud_window(source, start, end, analysis_proxy)
            if getattr(model, "requires_scan_transcript", False):
                transcript = local_transcript(start, end)
                return model.scan(analysis_proxy, end - start, transcript=hints(transcript))
            return model.scan(analysis_proxy, end - start)

        def transcript_file(start, end):
            return self.folder / f"asr-{editor.stable_id([editor.EDITOR_VERSION, start, end])}.json"

        def local_transcript(start, end):
            path = transcript_file(start, end)
            transcript = editor.read_json(path)
            if transcript is None:
                transcript = speech.from_video(analysis_proxy, self.folder, source_info["audio"])
                editor.write_json(path, transcript)
            return transcript

        def hints(transcript):
            return [{key: row[key] for key in ("start", "end", "text")} for row in transcript]

        def refine(seed, start, end):
            media.cloud_window(source, start, end, analysis_proxy, dense=True)
            transcript = local_transcript(start, end)
            relative_seed = dict(seed)
            for key in ("start", "peak", "end"):
                relative_seed[key] = round(seed[key] - start, 3)
            return model.refine(analysis_proxy, relative_seed, hints(transcript), end - start)

        def check_opening(item, minimum_novelty):
            t0, t1 = item["start"], item["end"]
            def judge():
                media.cloud_window(source, t0, t1, analysis_proxy, dense=True)
                dialogue = hints(local_transcript(t0, t1))
                return model.judge_opening(analysis_proxy, dialogue, t1-t0)
            verdict, tokens = self._cloud_cache(["opening-v2", t0, t1], editor.OpeningReview, judge)
            collect(tokens)
            item = editor.apply_opening(item, verdict, minimum_novelty)
            if item is None:
                return
            if not editor.read_json(transcript_file(t0, t1)) and not media.has_visual_change(
                    source, item["start"], t1):
                return
            refined.append(item)

        def review_highlight(seed):
            start, end = editor.opening_context(seed, length)
            tokens = {"input": 0, "output": 0}
            with TemporaryDirectory(prefix="highlight-", dir=self.folder) as directory:
                work = Path(directory)
                proxy = work / "proxy.mp4"

                def transcript(lo, hi):
                    # Most River scans already transcribed a full overlapping window.
                    for a, b in windows:
                        if a > lo:
                            break
                        if lo >= a and hi <= b:
                            rows = editor.read_json(transcript_file(a, b))
                            if rows is not None:
                                overlap = [row for row in rows if a + row["start"] < hi and a + row["end"] > lo]
                                if all(lo <= a + row["start"] and a + row["end"] <= hi for row in overlap):
                                    return [dict(start=round(a + row["start"] - lo, 2),
                                                 end=round(a + row["end"] - lo, 2), text=row["text"])
                                            for row in overlap]
                    return hints(editor.LocalSpeech().from_video(proxy, work, source_info["audio"]))

                def plan():
                    media.cloud_window(source, start, end, proxy, dense=True)
                    hint = {"title": seed["title"], "evidence": seed["evidence"],
                            **{k: round(seed[k] - start, 3) for k in ("start", "peak", "end")}}
                    return model.edit_highlight(proxy, transcript(start, end), end-start, hint)

                cut, spent = self._cloud_cache(["highlight-edit-v1", start, end, seed], editor.HighlightEdit, plan)
                for key in tokens:
                    tokens[key] += spent[key]
                item = editor.checked_highlight(cut, start, end-start, seed)
                if item is None:
                    return None, tokens
                lo, hi = item["start"], item["end"]

                def verify():
                    media.cloud_window(source, lo, hi, proxy, dense=True)
                    return model.verify_highlight(proxy, transcript(lo, hi), hi-lo)

                verdict, spent = self._cloud_cache(["highlight-verdict-v1", lo, hi], editor.OpeningReview, verify)
                for key in tokens:
                    tokens[key] += spent[key]
            if not (verdict.publish and verdict.opening.strip() and verdict.payoff.strip()
                    and 0 <= verdict.hook_at <= 2 and verdict.hook_at < verdict.payoff_at < hi-lo):
                return None, tokens
            item.update(hook_time=round(lo + verdict.hook_at, 3), peak=round(lo + verdict.payoff_at, 3),
                        reason=verdict.opening, evidence=verdict.payoff,
                        score=round(.7 * verdict.novelty + .3 * item["score"], 1))
            if "audience_views" in seed:
                item["audience_views"] = seed["audience_views"]
            return item, tokens

        def rank_reviewed(reviewed_clips):
            unique = editor.deduplicate(reviewed_clips)
            if not unique:
                return []
            result, tokens = self._cloud_cache(["highlight-ranking-v2", unique], editor.HighlightRanking,
                                               lambda: model.rank_highlights(unique))
            collect(tokens)
            return [unique[i] for i in dict.fromkeys(result.indices) if 0 <= i < len(unique)]

        try:
            for n, (start, end) in enumerate(windows, 1):
                self.progress(f"Облачный просмотр VOD · окно {n} из {len(windows)}", n - 1, len(windows))
                result, tokens = self._cloud_cache(["scan", start, end], editor.Scan, lambda: scan(start, end))
                collect(tokens)
                transcript = editor.read_json(transcript_file(start, end)) or []
                for raw in result.events:
                    item = editor.checked_event(raw, start, end-start)
                    if item and item["score"] >= 54:
                        item["dialogue"] = " ".join(r["text"] for r in transcript
                                                    if start+r["end"] > item["start"]
                                                    and start+r["start"] < item["end"])[:700]
                        if realtime:
                            item["signals"] = sorted({kind for t, kind in state.get("live_signals", [])
                                                      if item["start"] <= t <= item["end"]+5})
                        events.append(item)
            pool = editor.deduplicate(events)
            ranked_pool = pool.copy()
            first_round = []
            candidate_count = len(pool)
            if (realtime and pool) or len(pool) > editor.REFINE_LIMIT:
                pool.sort(key=lambda item: item["peak"])
                def compare(group):
                    result, tokens = self._cloud_cache(["compare-v1", group], editor.Nominations,
                                                       lambda: model.nominate(group))
                    return [group[i] for i in dict.fromkeys(result.indices) if 0 <= i < len(group)], tokens
                if realtime and len(pool) <= 60:
                    pool, tokens = compare(pool)
                    collect(tokens)
                    pool = pool[:2]
                while len(pool) > editor.REFINE_LIMIT:
                    groups = [pool[i:i + 60] for i in range(0, len(pool), 60)]
                    self.progress(f"Сравнение {len(pool)} моментов по завязке и развязке", 0, len(groups))
                    with ThreadPoolExecutor(max_workers=3) as comparisons:
                        pool = []
                        for winners, tokens in comparisons.map(compare, groups):
                            pool.extend(winners)
                            collect(tokens)
                    if not first_round:
                        first_round = pool.copy()
            from .twitch_audience import viewer_moments
            audience_file = self.folder / "viewer-moments.json"
            audience = []
            if contextual_review and state["vod"]:
                audience = editor.read_json(audience_file) or viewer_moments(state["vod"], length)
                if audience and not audience_file.exists():
                    editor.write_json(audience_file, audience)
            nominated = audience + pool
            reviewed = len(nominated)
            if contextual_review:
                # Preserve promising episodes from early comparison groups and spread backup checks over the VOD.
                priority = first_round + ranked_pool
                reserve = []
                covered = set()
                checked = list(nominated)
                region_seconds = max(30 * 60, length / editor.VOD_RESERVE_LIMIT)
                for seed in priority:
                    region = int(seed["peak"] // region_seconds)
                    if region in covered or any(abs(seed["peak"] - other["peak"]) < 90 for other in checked):
                        continue
                    reserve.append(seed)
                    checked.append(seed)
                    covered.add(region)
                    if len(reserve) >= editor.VOD_RESERVE_LIMIT:
                        break
                with ThreadPoolExecutor(max_workers=3) as workers:
                    selected = []
                    batch = nominated
                    for stage in ("Проверка хука и развязки", "Проверка резерва"):
                        self.progress(stage, 0, len(batch))
                        checked_clips = []
                        for n, seed_result in enumerate(workers.map(review_highlight, batch), 1):
                            item, tokens = seed_result
                            collect(tokens)
                            if item is not None and len(editor.deduplicate(selected + [item])) > len(selected):
                                checked_clips.append(item)
                            self.progress(f"{stage} · {n} из {len(batch)}", n, len(batch))
                        self.progress("Сравнение готовых фрагментов", 0, 1)
                        selected.extend(rank_reviewed(checked_clips)[:editor.VOD_REVIEW_MAX - len(selected)])
                        if not reserve or len(selected) >= editor.VOD_REVIEW_TARGET:
                            break
                        batch = reserve
                        reserve = []
                        reviewed += len(batch)
            else:
                for n, seed in enumerate(nominated, 1):
                    self.progress(f"Повторный просмотр · {n} из {len(nominated)}", n - 1, len(nominated))
                    start, end = editor.limit_window(seed, length)
                    result, tokens = self._cloud_cache(["refine", seed, start, end], editor.Refinement,
                                                        lambda: refine(seed, start, end))
                    collect(tokens)
                    if result.event is None:
                        continue
                    item = editor.checked_event(result.event, start, end - start)
                    if (item and item["score"] >= 65 and item["end"] - item["start"] <= 49
                            and abs(item["peak"] - seed["peak"]) <= 12):
                        transcript = editor.read_json(transcript_file(start, end)) or []
                        item = editor.snap_speech(item, transcript, start, end)
                        item = editor.include_reaction_tail(item, transcript, start, end)
                        item["cameras"] = editor.camera_spans(result.cameras, start, end - start)
                        check_opening(item, minimum_novelty=78)
        finally:
            model.close()
            analysis_proxy.unlink(missing_ok=True)

        if not contextual_review:
            selected = editor.deduplicate(refined)[:editor.TOP_CLIPS]
        clips = []
        for n, clip in enumerate(selected, 1):
            clip_id = editor.stable_id([clip["start"], clip["peak"], clip["end"]])
            self.progress(f"Подготовка исходных фрагментов · {n} из {len(selected)}", n - 1, len(selected))
            folder = self.folder / clip_id
            folder.mkdir(exist_ok=True)
            start = max(0, clip["start"] - 5)
            end = min(length, clip["end"] + 5)
            self._reference(folder, source, start, end)
            clip.update(id=clip_id, selected=True, camera_mode="auto", camera_box=None,
                        reference_start=start, reference_end=end, turns=[], scripted=False)
            clip["ad_at"] = media.choose_ad_point(clip)
            clips.append(clip)
        self.update(status="review", clips=clips, contextual_review=contextual_review,
                    analysis_provider=editor.analysis_config()[0], candidates=candidate_count, finalists=reviewed, audience_candidates=len(audience),
                    covered_windows=len(windows), analysis_usage=usage, progress=None)

    @staticmethod
    def _reference(folder: Path, source: Path, start: float, end: float) -> None:
        reference = folder / "reference.mp4"
        marker = folder / "reference.json"
        bounds = [start, end]
        if reference.is_file() and editor.read_json(marker) == bounds:
            return
        temp = folder / "reference.partial.mp4"
        try:
            media.cloud_window(source, start, end, temp, dense=True)
            temp.replace(reference)
            editor.write_json(marker, bounds)
        finally:
            temp.unlink(missing_ok=True)

    def _script_clip(self, clip: dict, model: editor.EditingModel,
                     speech: editor.LocalSpeech, has_audio: bool) -> bool:
        if clip.get("scripted"):
            return False
        folder = self.folder / clip["id"]
        start, end = clip["start"], clip["end"]
        proxy = folder / "script.partial.mp4"

        def adapt():
            media.cloud_window(self.source, start, end, proxy, dense=True)
            transcript = speech.from_video(proxy, folder, has_audio)
            hints = [{key: row[key] for key in ("start", "end", "text")} for row in transcript]
            result, usage = model.adapt(proxy, hints, end - start)
            if hints and not result.turns and sum(row["end"] - row["start"] for row in hints) > 1.2:
                raise media.EditError(f"{editor.analysis_config()[0]} пропустил речь в выбранном отрывке. Повторите перевод или измените границы.")
            return result, usage

        try:
            result, usage = self._cloud_cache(["script", 2, start, end, clip["id"]], editor.Script, adapt)
        finally:
            proxy.unlink(missing_ok=True)
        turns = editor.checked_turns(result, end - start) if has_audio else []
        if has_audio and result.turns and not turns:
            raise media.EditError("Метки дубляжа вышли за границы клипа. Повторите перевод.")
        clip.update(turns=turns, scripted=True, script_usage=usage)
        clip["ad_at"] = media.choose_ad_point(clip)
        return True

    def script(self) -> None:
        data = self.read()
        clips = data["clips"]
        selected = [clip for clip in clips if clip["selected"]]
        model = editor.create_editor()
        speech = editor.LocalSpeech()
        try:
            for n, clip in enumerate(selected, 1):
                self.progress(f"Адаптация реплик · клип {n} из {len(selected)}", n - 1, len(selected))
                self._script_clip(clip, model, speech, data["source_info"]["audio"])
                self.update(clips=clips)
        finally:
            model.close()
        self.update(status="scripted", progress=None)

    def _ad(self, data: dict, clip: dict) -> tuple[Path | None, media.Media | None]:
        if not data.get("ad") or clip.get("ad_enabled", True) is False:
            return None, None
        if clip.get("ad_at") is None:
            raise media.EditError(f"Выберите точку рекламы для клипа «{clip['title']}» или отключите рекламу в нём.")
        return self.folder / data["ad"]["file"], media.Media(**data["ad"]["info"])

    def _review_camera(self, clip, model, work):
        start, end = clip["start"], clip["end"]
        key = editor.stable_id(["camera-v1", editor.analysis_config()[:2], start, end])
        if clip.get("camera_mode") != "auto" or clip.get("camera_review") == key:
            return
        proxy = work / "camera.partial.mp4"

        def locate():
            media.cloud_window(self.source, start, end, proxy, dense=True)
            return model.locate_camera(proxy, end-start)

        try:
            result, _ = self._cloud_cache(["camera-v1", start, end], editor.CameraReview, locate)
        finally:
            proxy.unlink(missing_ok=True)
        clip.update(cameras=editor.camera_spans(result.cameras, start, end-start), camera_review=key)

    def preview(self) -> None:
        data = self.read()
        clips = data["clips"]
        selected = [clip for clip in clips if clip["selected"]]
        for clip in selected:
            self._ad(data, clip)
        speech = editor.LocalSpeech()
        model = fish = None
        try:
            if any(clip["turns"] for clip in selected):
                model = editor.create_editor()
                fish = editor.FishVoices()
            for n, clip in enumerate(selected, 1):
                self.progress(f"Дубляж и предпросмотр · клип {n} из {len(selected)}", n - 1, len(selected))
                work = self.folder / clip["id"]
                if clip.get("camera_mode") == "auto":
                    if model is None:
                        model = editor.create_editor()
                    self._review_camera(clip, model, work)
                identity = self.preview_identity(data, clip)
                target = work / "preview.mp4"
                if target.is_file() and clip.get("preview_key") == identity:
                    continue
                ad, ad_info = self._ad(data, clip)
                parts, cues, resolved = [], [], []
                for turn in clip["turns"]:
                    role = "guest" if turn["role"] == "guest" else "streamer"
                    if clip.get("ad_at") is not None and ad:
                        point = media.frames(clip["ad_at"]) / media.FPS
                        if turn["start"] < point < turn["end"]:
                            raise media.EditError(f"Реклама в клипе «{clip['title']}» прерывает реплику. "
                                                  "Выберите паузу между репликами.")
                    try:
                        path, spoken, captions = editor.dub_turn(turn, data["voices"][role], work, fish, model, speech)
                    except media.EditError as error:
                        raise media.EditError(f"«{clip['title']}», реплика на {turn['start']:.2f} с. {error}") from error
                    parts.append({"start": spoken["start"], "end": spoken["end"], "path": path})
                    cues.extend(captions)
                    resolved.append(spoken)
                voice = work / "voice.wav" if parts else None
                if voice:
                    media.voice_track(parts, clip["end"] - clip["start"], voice)
                duration = media.render(self.source, clip, data["source_info"]["audio"], work, target,
                                        voice=voice, cues=cues, ad=ad, ad_info=ad_info, preview=True)
                editor.write_json(work / "prepared.json", {"key": identity, "voice": bool(voice),
                                                           "cues": cues, "duration": duration})
                clip.update(preview_key=identity, spoken=resolved, rendered_duration=duration)
                self.update(clips=clips)
        finally:
            if fish:
                fish.close()
            if model:
                model.close()
        self.update(status="preview", progress=None)

    @staticmethod
    def preview_identity(data: dict, clip: dict) -> str:
        fields = ("start", "end", "turns", "camera_mode", "camera_box", "cameras", "ad_at", "ad_enabled")
        return editor.stable_id([editor.EDITOR_VERSION, editor.DUB_VERSION, media.LAYOUT_VERSION,
                                 media.LOCAL_CAPTION_FONT.is_file(), editor.analysis_config()[:2],
                                 data["voices"], data.get("ad"),
                                 {field: clip.get(field) for field in fields}])

    def ready_to_export(self, data: dict) -> bool:
        return all(clip.get("preview_key") == self.preview_identity(data, clip)
                   for clip in data["clips"] if clip["selected"])

    def export(self) -> None:
        data = self.read()
        clips = [clip for clip in data["clips"] if clip["selected"]]
        # Stage and verify every MP4 on the output volume before exposing the job folder.
        font_variant = "-obelix" if media.LOCAL_CAPTION_FONT.is_file() else ""
        staging = self.output.with_name(f".{self.output.name}.layout-{media.LAYOUT_VERSION}{font_variant}.partial")
        staging.mkdir(parents=True, exist_ok=True)
        files = []
        for n, clip in enumerate(clips, 1):
            self.progress(f"Экспорт 1080×1920 · {n} из {len(clips)}", n - 1, len(clips))
            work = self.folder / clip["id"]
            prepared = editor.read_json(work / "prepared.json")
            if not prepared or prepared["key"] != self.preview_identity(data, clip):
                raise media.EditError("Предпросмотр устарел. Соберите его повторно перед экспортом.")
            suffix = re.sub(r"[^\w-]+", "-", clip["title"].lower(), flags=re.UNICODE).strip("-")[:48]
            filename = f"{n:02d}-{suffix or clip['id'][:8]}.mp4"
            files.append(filename)
            ad, ad_info = self._ad(data, clip)
            target = staging / filename
            existing = self.output / filename
            if existing.is_file():
                media.verify_export(existing, (1080, 1920), prepared["duration"])
                continue
            if target.is_file():
                try:
                    media.verify_export(target, (1080, 1920), prepared["duration"])
                    continue
                except media.EditError:
                    target.unlink()
            media.render(self.source, clip, data["source_info"]["audio"], work, target,
                         voice=work / "voice.wav" if prepared["voice"] else None,
                         cues=prepared["cues"], ad=ad, ad_info=ad_info)
        if self.output.is_dir():
            if any(not (self.output / f).is_file() for f in files):
                raise media.EditError("Итоговая папка была изменена. Верните сохранённые файлы или создайте новую задачу.")
            staging.rmdir()
        else:
            staging.rename(self.output)
        self.update(status="done", files=files, output=str(self.output), progress=None)


class Jobs:
    def __init__(self, root: Path = JOB_ROOT, outputs: Path = OUTPUT_ROOT):
        self.root = root
        self.outputs = outputs
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="twitch")
        self.lock = RLock()
        self.uploading: set[str] = set()

    def all(self, *, include_pending: bool = False) -> list[dict]:
        if not self.root.exists():
            return []
        records = []
        for record in self.root.glob("*/job.json"):
            try:
                modified = record.stat().st_mtime
            except FileNotFoundError:
                continue
            state = editor.read_json(record)
            if state and (include_pending or not state.get("live_pending") or state["status"] == "failed"):
                records.append((modified, {field: state.get(field) for field in ("id", "label", "status")}))
        return [entry for _, entry in sorted(records, key=lambda pair: -pair[0])]

    def recover(self) -> None:
        for state in self.all(include_pending=True):
            if state["status"] == "watching":
                self.get(state["id"]).update(status="live_done", error="Наблюдение прервано при закрытии студии.", progress=None)
            elif state["status"] in BUSY.values():
                self.get(state["id"]).update(status="failed", live_pending=False, error="Приложение закрылось во время обработки. "
                                            "Успешные этапы сохранены, нажмите «Повторить этап».", progress=None)

    def get(self, job_id: str) -> Job:
        if not JOB_ID.fullmatch(job_id):
            raise media.EditError("Задача не найдена.")
        job = Job(self.root / job_id, self.outputs)
        job.read()
        return job

    def create(self, url: str, label: str) -> Job:
        vod = media.twitch_id(url) if url else ""
        job_id = uuid4().hex[:12]
        root = self.root / job_id
        root.mkdir(parents=True, mode=0o700)
        job = Job(root, self.outputs)
        editor.write_json(job.record, {"id": job_id, "vod": vod, "label": f"Twitch {vod}" if vod else label,
                                       "status": "created", "clips": [], "progress": None, "error": None})
        return job

    def rescan(self, original: Job) -> dict:
        with self.lock:
            state = original.read()
            if state["status"] not in {"review", "scripted", "preview", "done"} or original.id in self.uploading:
                raise media.EditError("Дождитесь завершения текущего этапа.")
            child = self.create("", state["label"] + " · новый отбор")
            try:
                names = [original.source, original.folder / "source.jpg"]
                if (original.folder / "viewer-moments.json").is_file():
                    names.append(original.folder / "viewer-moments.json")
                names.extend(original.folder.glob("ai-*.json"))
                names.extend(original.folder.glob("asr-*.json"))
                for path in names:
                    (child.folder / path.name).hardlink_to(path)
                child.update(source=original.source.name, source_info=state["source_info"],
                             vod=state["vod"], voices=state.get("voices"), status="source_ready")
                return self.start(child, "analyze")
            except Exception:
                shutil.rmtree(child.folder)
                raise

    def start(self, job: Job, action: str, *, changes: dict | None = None) -> dict:
        with self.lock:
            state = job.read()
            status = state["status"]
            if job.id in self.uploading:
                raise media.EditError("Дождитесь завершения загрузки файла.")
            if action == "retry":
                if status != "failed" or state.get("operation") not in BUSY:
                    raise media.EditError("Нет прерванного этапа.")
                action = state["operation"]
            elif action not in NEXT or status not in NEXT[action]:
                raise media.EditError("Этот этап сейчас недоступен.")
            if action == "export" and not job.ready_to_export(state):
                note = "Предпросмотр устарел. Соберите его повторно перед экспортом."
                if status == "failed":
                    return job.update(status="scripted", error=note, progress=None)
                raise media.EditError(note)
            new_state = job.update(**(changes or {}), status=BUSY[action], operation=action,
                                   previous=status if status != "failed" else state.get("previous"), error=None,
                                   progress={"message": "Этап в очереди…", "done": 0, "total": 1})
            self.worker.submit(self._run, job, action)
            return new_state

    @staticmethod
    def _run(job: Job, action: str) -> None:
        try:
            operations = {"load": job.prepare_source, "analyze": job.analyze, "script": job.script,
                          "preview": job.preview, "export": job.export}
            operations[action]()
        except media.EditError as error:
            job.update(status="failed", error=str(error), progress=None)
        except Exception as error:
            # Trace locations help diagnose local errors without logging credentials or provider bodies.
            LOG.error("%s: %s\n%s", action, type(error).__name__,
                      "".join(traceback.format_tb(error.__traceback__)))
            job.update(status="failed", error=f"Ошибка этапа ({type(error).__name__}). "
                       "Успешные результаты сохранены. Повторите этап.", progress=None)
