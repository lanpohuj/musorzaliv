"""Run the Twitch workflow in a browser on the local machine."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import asdict
from hashlib import sha256
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import webbrowser

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, FiniteFloat
from starlette.middleware.trustedhost import TrustedHostMiddleware
from typing import Literal

from . import twitch_editor as editor
from . import twitch_media as media
from .twitch_workflow import BUSY, CLIP_ID, Jobs, ROOT

STATIC = ROOT / "web_twitch"
FILENAME = {
    "reference": ("reference.mp4", "video/mp4"),
    "preview": ("preview.mp4", "video/mp4"),
}


class NewJob(BaseModel):
    url: str = Field(default="", max_length=600)
    label: str = Field(default="Локальное видео", max_length=180)


class Voices(BaseModel):
    streamer: str
    guest: str

    def checked(self) -> dict:
        voices = self.model_dump()
        if any(not editor.VOICE_ID.fullmatch(value) for value in voices.values()):
            raise media.EditError("Выберите два голоса Fish Audio или укажите их ID.")
        return voices


class TurnEdit(BaseModel):
    start: FiniteFloat
    end: FiniteFloat
    role: Literal["streamer", "guest", "unknown"]
    russian: str = Field(min_length=1, max_length=400)


class ClipEdit(BaseModel):
    id: str
    selected: bool
    start: FiniteFloat
    end: FiniteFloat
    camera_mode: Literal["auto", "manual", "off"] = "auto"
    camera_box: list[FiniteFloat] | None = Field(default=None, max_length=4)
    ad_enabled: bool = True
    ad_at: FiniteFloat | None = None
    turns: list[TurnEdit] | None = Field(default=None, max_length=55)


class Action(BaseModel):
    action: Literal["load", "analyze", "script", "preview", "export", "retry", "back"]
    voices: Voices | None = None
    clips: list[ClipEdit] | None = Field(default=None, max_length=editor.TOP_CLIPS)


def edited_clips(state: dict, edits: list[ClipEdit], action: str) -> list[dict]:
    originals = state["clips"]
    by_id = {edit.id: edit for edit in edits}
    if len(edits) != len(originals) or set(by_id) != {clip["id"] for clip in originals}:
        raise media.EditError("Список клипов изменился. Обновите задачу.")
    result = []
    for original in originals:
        edit = by_id[original["id"]]
        lo, hi = round(edit.start, 3), round(edit.end, 3)
        if not (max(0, original["reference_start"] - .001) <= lo < hi <= original["reference_end"] + .001
                and lo <= original["peak"] <= hi and 1 <= hi - lo <= 60):
            raise media.EditError(f"Проверьте границы клипа «{original['title']}». "
                                  "Пик должен оставаться внутри выбранного фрагмента.")
        changed = (lo, hi) != (original["start"], original["end"])
        box = media.camera_box(edit.camera_box or []) if edit.camera_mode == "manual" else None
        clip = {**original, "selected": edit.selected, "start": lo, "end": hi,
                "camera_box": box, "camera_mode": edit.camera_mode, "ad_at": edit.ad_at,
                "ad_enabled": edit.ad_enabled}
        if not changed and edit.ad_at is not None and not 0 < media.frames(edit.ad_at) < media.frames(hi - lo):
            raise media.EditError(f"Рекламная пауза должна быть внутри клипа «{clip['title']}».")
        if changed:
            clip.update(turns=[], scripted=False, spoken=[], preview_key=None, ad_at=None)
        elif edit.turns is not None:
            if len(edit.turns) != len(original["turns"]):
                raise media.EditError("Число реплик изменилось. Обновите предпросмотр.")
            updated = []
            for turn, update in zip(original["turns"], edit.turns):
                start, end = round(update.start, 3), round(update.end, 3)
                if (not 0 <= start < end <= hi - lo + .001 or end - start < 0.17
                        or (updated and start < updated[-1]["end"]) or not update.russian.strip()):
                    raise media.EditError(f"Проверьте текст и границы реплик клипа «{clip['title']}».")
                text = update.russian.strip()
                if "[" in text or "]" in text:
                    raise media.EditError("Используйте обычный текст реплики, без ремарок в квадратных скобках.")
                updated.append({**turn, "start": start, "end": min(end, hi - lo),
                                "role": update.role, "russian": text})
            clip["turns"] = updated
        result.append(clip)
    selected = [clip for clip in result if clip["selected"]]
    if not selected:
        raise media.EditError("Выберите хотя бы один клип.")
    if action == "preview" and any(not clip["scripted"] for clip in selected):
        raise media.EditError("Сначала подготовьте перевод для всех выбранных клипов и новых границ.")
    return result


def create_app(jobs: Jobs | None = None) -> FastAPI:
    from .twitch_live import LiveSessions, channel_name
    manager = jobs or Jobs()
    live = LiveSessions(manager)
    token = secrets.token_urlsafe(24)

    @asynccontextmanager
    async def lifespan(_):
        manager.recover()
        yield
        live.close()
        manager.worker.shutdown(wait=False)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])
    app.mount("/ui", StaticFiles(directory=STATIC), name="ui")

    @app.exception_handler(media.EditError)
    async def edit_error(_: Request, error: media.EditError):
        return JSONResponse({"detail": str(error)}, status_code=400)

    def authorized(request: Request):
        if not secrets.compare_digest(token, request.headers.get("x-twitch-token", "")):
            raise HTTPException(403, "Обновите страницу, чтобы продолжить.")

    write = [Depends(authorized)]

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/health")
    def health():
        with manager.lock:
            busy = bool(live.sessions) or bool(manager.uploading) or any(job["status"] in BUSY.values() for job in manager.all())
        return JSONResponse({
            "workspace": sha256(os.fsencode(ROOT.resolve())).hexdigest(),
            "busy": busy,
        }, headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status():
        provider, model, ready = editor.analysis_config()
        return JSONResponse({
            "token": token, "analysis": {"provider": provider, "model": model, "ready": ready},
            "fish": bool(os.environ.get("FISH_API_KEY")), "jobs": manager.all(),
        }, headers={"Cache-Control": "no-store"})

    @app.post("/api/jobs", dependencies=write)
    def new_job(options: NewJob):
        channel = channel_name(options.url)
        if channel:
            if not editor.analysis_config()[2]:
                raise media.EditError("Для анализа эфира настройте провайдера.")
            return live.start(channel)
        job = manager.create(options.url.strip(), Path(options.label).name)
        return manager.start(job, "load") if options.url.strip() else job.read()

    @app.post("/api/jobs/{job_id}/rescan", dependencies=write)
    def rescan(job_id: str):
        if not editor.analysis_config()[2]:
            raise media.EditError("Для анализа настройте провайдера.")
        return manager.rescan(manager.get(job_id))

    @app.post("/api/jobs/{job_id}/stop-live", dependencies=write)
    def stop_live(job_id: str):
        live.stop(job_id)
        return manager.get(job_id).read()

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: str):
        state = manager.get(job_id).read()
        if state["status"] == "source_ready":
            state["estimate"] = editor.estimate(state["source_info"]["duration"], twitch_vod=bool(state["vod"]))
        return JSONResponse(state, headers={"Cache-Control": "no-store"})

    @app.put("/api/jobs/{job_id}/asset/{kind}", dependencies=write)
    async def upload(job_id: str, kind: Literal["source", "ad"], request: Request, name: str = Query(max_length=200)):
        job = manager.get(job_id)
        extension = Path(name).suffix.lower()
        if extension not in media.VIDEO_EXTENSIONS:
            raise media.EditError("Загрузите видео MP4, MOV, MKV или WebM.")
        maximum = media.MAX_UPLOAD if kind == "source" else media.MAX_AD_UPLOAD
        raw_size = request.headers.get("content-length", "0")
        try:
            expected = int(raw_size)
        except ValueError:
            raise media.EditError("Размер файла указан некорректно.") from None
        if expected < 0 or expected > maximum:
            raise media.EditError("Файл превышает максимальный размер.")
        if expected and shutil.disk_usage(job.folder).free < expected + 128 * 1024**2:
            raise media.EditError("Для загрузки файла нужно освободить место на диске.")
        with manager.lock:
            state = job.read()
            if job_id in manager.uploading or state["status"] in BUSY.values() or state["status"] == "done":
                raise media.EditError("Задача занята обработкой.")
            if kind == "source" and (state["status"] != "created" or state["vod"]):
                raise media.EditError("Для замены исходника создайте новую задачу.")
            if kind == "ad" and state.get("operation") == "export" and state["status"] == "failed":
                raise media.EditError("Завершите прерванный экспорт перед заменой рекламы.")
            manager.uploading.add(job_id)

        temporary = job.folder / f"{kind}.upload{extension}"
        try:
            size = 0
            with temporary.open("wb") as out:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > maximum:
                        raise media.EditError("Файл превышает максимальный размер.")
                    out.write(chunk)
            info = await run_in_threadpool(media.probe, temporary)
            with manager.lock:
                if kind == "source":
                    dest = job.folder / f"source{extension}"
                    temporary.replace(dest)
                    current = job.update(source=dest.name, label=Path(name).name)
                else:
                    if not info.audio:
                        raise media.EditError("У рекламного видео должна быть собственная звуковая дорожка.")
                    if info.duration < 1 / media.FPS:
                        raise media.EditError("В рекламном ролике нет целого видеокадра.")
                    dest = job.folder / f"ad-{secrets.token_hex(5)}{extension}"
                    temporary.replace(dest)
                    old = state.get("ad")
                    patch = {"ad": {"file": dest.name, "label": Path(name).name, "info": asdict(info)}}
                    if state["status"] == "preview":
                        patch.update(status="scripted")
                    current = job.update(**patch)
                    if old:
                        (job.folder / old["file"]).unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)
            with manager.lock:
                manager.uploading.discard(job_id)
        return manager.start(job, "load") if kind == "source" else current

    @app.post("/api/jobs/{job_id}/action", dependencies=write)
    def act(job_id: str, options: Action):
        job = manager.get(job_id)
        with manager.lock:
            state = job.read()
            action = options.action
            if action == "back":
                if state["status"] != "failed" or state.get("operation") == "export" or job_id in manager.uploading:
                    raise media.EditError("Вернуться с этого этапа сейчас нельзя.")
                return job.update(status=state["previous"], error=None)
            changes = {}
            if action in {"analyze", "script", "preview"} and options.voices:
                changes["voices"] = options.voices.checked()
            if action == "analyze":
                if not changes.get("voices"):
                    raise media.EditError("Выберите голоса для стримера и собеседников.")
                provider, _, ready = editor.analysis_config()
                if not ready:
                    key = "RIVER_API_KEY" if provider == "River" else "GEMINI_API_KEY"
                    raise media.EditError(f"Задайте {key} и перезапустите приложение.")
                if not os.environ.get("FISH_API_KEY"):
                    raise media.EditError("Задайте FISH_API_KEY и перезапустите приложение.")
            if action in {"script", "preview"}:
                if options.clips is None:
                    raise media.EditError("Выберите клипы в списке.")
                changes["clips"] = edited_clips(state, options.clips, action)
            return manager.start(job, action, changes=changes)

    @app.get("/api/voices")
    def voices(q: str = Query(default="", max_length=80), page: int = Query(default=1, ge=1, le=100)):
        fish = editor.FishVoices()
        try:
            return fish.voices(q.strip(), page)
        finally:
            fish.close()

    @app.post("/api/voice-sample/{voice_id}", dependencies=write)
    def voice_sample(voice_id: str):
        if not editor.VOICE_ID.fullmatch(voice_id):
            raise media.EditError("Введите корректный ID голоса.")
        samples = manager.root / "_voice_samples"
        samples.mkdir(parents=True, exist_ok=True)
        target = samples / f"{voice_id}.wav"
        if not target.is_file():
            fish = editor.FishVoices()
            temporary = target.with_name(f"{voice_id}-{secrets.token_hex(5)}.partial.wav")
            try:
                temporary.write_bytes(fish.synthesize(
                    voice_id, "Подожди, ты это видел? У меня один патрон. Ладно, погнали!", "excited"))
                media.probe(temporary, require_video=False)
                temporary.replace(target)
            finally:
                fish.close()
                temporary.unlink(missing_ok=True)
        return FileResponse(target, media_type="audio/wav")

    @app.get("/api/jobs/{job_id}/source.jpg")
    def source_image(job_id: str):
        path = manager.get(job_id).folder / "source.jpg"
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/jobs/{job_id}/media/{clip_id}/{kind}")
    def clip_media(job_id: str, clip_id: str, kind: Literal["reference", "preview"]):
        if not CLIP_ID.fullmatch(clip_id):
            raise HTTPException(404)
        name, mime = FILENAME[kind]
        job = manager.get(job_id)
        path = job.folder / clip_id / name
        if not path.is_file() or clip_id not in {clip["id"] for clip in job.read()["clips"]}:
            raise HTTPException(404)
        return FileResponse(path, media_type=mime, headers={"Cache-Control": "no-cache"})

    @app.post("/api/jobs/{job_id}/open", dependencies=write)
    def open_folder(job_id: str):
        job = manager.get(job_id)
        if job.read()["status"] != "done" or not job.output.is_dir():
            raise media.EditError("Папка экспорта ещё недоступна.")
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(job.output)])
        elif sys.platform.startswith("win"):
            os.startfile(str(job.output))
        else:
            subprocess.Popen(["xdg-open", str(job.output)])
        return {"path": str(job.output)}

    return app


def main():
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", interpolate=False)
    parser = argparse.ArgumentParser(description="Локальная студия Twitch-нарезок MUSORZALIV")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Открыть интерфейс в браузере")
    args = parser.parse_args()
    if args.open:
        from threading import Timer
        Timer(1, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, proxy_headers=False)


if __name__ == "__main__":
    main()
