"""Record and evaluate overlapping Twitch live windows while capture continues."""
import csv
from collections import deque
import socket
import ssl
import time
from random import randrange
import re
import shutil
import signal
import subprocess
from threading import Event, Lock, Thread
from urllib.parse import urlsplit

from . import twitch_media as media
from .twitch_river import audio_peaks


def channel_name(url):
    parsed = urlsplit(url.strip())
    match = re.fullmatch(r"/([A-Za-z0-9_]{2,25})/?", parsed.path)
    if parsed.scheme == "https" and parsed.netloc in {"www.twitch.tv", "twitch.tv"} and match:
        return match[1].lower()
    return None


class ChatPulse:
    """Record only burst timestamps so analysis can catch up with the capture."""
    def __init__(self):
        self.started = time.monotonic()
        self.recent = deque(maxlen=20000)
        self.bursts = []
        self.lock = Lock()

    def message(self, second, nick, line):
        while self.recent and self.recent[0][0] < second - 180:
            self.recent.popleft()
        self.recent.append((second, nick))
        if self.bursts and second - self.bursts[-1] < 8:
            return
        recent = set()
        for stamp, author in reversed(self.recent):
            if stamp < second - 8:
                break
            recent.add(author)
        span = min(172, max(1, second - 8))
        threshold = max(5, 2.5 * len(self.recent) * 8 / span)
        if len(recent) >= threshold or (len(recent) >= 2 and "clip that" in line.lower()):
            with self.lock:
                self.bursts.append(second)

    def times(self, start, end):
        with self.lock:
            return [t for t in self.bursts if start <= t <= end]

    def between(self, start, end):
        return bool(self.times(start, end))

    def listen(self, channel, done):
        host = "irc.chat.twitch.tv"
        while not done.is_set():
            try:
                with socket.create_connection((host, 6697), timeout=8) as raw:
                    with ssl.create_default_context().wrap_socket(raw, server_hostname=host) as conn:
                        conn.sendall(f"NICK justinfan{randrange(10000,99999)}\r\nJOIN #{channel}\r\n".encode())
                        conn.settimeout(3)
                        pending = b""
                        while not done.is_set():
                            try:
                                packet = conn.recv(8192)
                            except socket.timeout:
                                continue
                            if not packet:
                                break
                            *lines, pending = (pending + packet).split(b"\r\n")
                            for line in lines:
                                if line.startswith(b"PING "):
                                    conn.sendall(b"PONG " + line[5:] + b"\r\n")
                                elif b" PRIVMSG #" in line:
                                    author, _, message = line.decode(errors="replace").partition(" PRIVMSG #")
                                    self.message(int(time.monotonic()-self.started),
                                                 author.split("!")[0], message)
            except OSError:
                pass
            done.wait(5)


class LiveSessions:
    def __init__(self, manager):
        self.manager = manager
        self.sessions = {}
        self.lock = Lock()

    def start(self, channel):
        with self.lock:
            if self.sessions:
                raise media.EditError("Сначала завершите текущее наблюдение.")
            job = self.manager.create("", f"Прямой эфир · {channel}")
            state = job.update(status="watching", channel=channel, batches=[],
                               progress={"message": "Подключение к Twitch…", "done": 0, "total": 0})
            stop = Event()
            self.sessions[job.id] = stop
            Thread(target=self._run, args=(job, channel, stop), daemon=True).start()
            return state

    def stop(self, job_id):
        with self.lock:
            if job_id not in self.sessions:
                raise media.EditError("Наблюдение уже завершено.")
            self.sessions[job_id].set()

    def close(self):
        with self.lock:
            for stop in self.sessions.values():
                stop.set()

    def _run(self, job, channel, stop):
        from yt_dlp import YoutubeDL
        capture = None
        chat_done = Event()
        try:
            with YoutubeDL({"quiet": True, "no_warnings": True, "socket_timeout": 15,
                           "format": "b[height<=1080]/b"}) as downloader:
                info = downloader.extract_info("https://www.twitch.tv/" + channel, download=False)
            if not info or not info.get("is_live"):
                raise media.EditError("Канал сейчас вне эфира.")
            if not stop.is_set():
                capture = subprocess.Popen([media.executable("ffmpeg"), "-nostdin", "-loglevel", "error",
                    "-rw_timeout", "30000000", "-i", info["url"], "-map", "0:v:0", "-map", "0:a:0?",
                    "-c", "copy", "-f", "segment", "-segment_time", "30", "-reset_timestamps", "1",
                    "-segment_list", "pieces.csv", "-segment_list_type", "csv", "piece-%08d.ts"],
                    cwd=job.folder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                Thread(target=self._guard, args=(capture, stop, job.folder), daemon=True).start()
                chat = ChatPulse()
                Thread(target=chat.listen, args=(channel, chat_done), daemon=True).start()
                self._consume(job, capture, stop, chat)
            job.update(status="live_done", progress=None, live_summary=job.read().get("progress", {}).get("message"))
        except Exception as exc:
            message = str(exc) if isinstance(exc, media.EditError) else (
                "Twitch недоступен или канал вне эфира." if capture is None else
                f"Запись эфира прервалась ({type(exc).__name__}).")
            job.update(status="live_done", error=message, progress=None)
        finally:
            chat_done.set()
            if capture and capture.poll() is None:
                capture.terminate()
                try:
                    capture.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    capture.kill()
                    capture.wait()
            if not job.read().get("error"):
                for piece in job.folder.glob("piece-*.ts"):
                    piece.unlink(missing_ok=True)
            with self.lock:
                self.sessions.pop(job.id, None)

    @staticmethod
    def _guard(proc, stop, folder):
        while proc.poll() is None:
            if stop.wait(1) or shutil.disk_usage(folder).free < 2 * 1024**3:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGINT)
                return

    def _consume(self, job, proc, stop, chat):
        index, quiet, first = 0, 0, 0
        seen = []
        signals = []
        job.progress("Запись эфира · поиск реакций в звуке и чате", 0, 0)
        while True:
            ended = proc.poll() is not None
            manifest = job.folder / "pieces.csv"
            rows = []
            if manifest.exists():
                lines = manifest.read_text().splitlines(keepends=True)
                rows = [r for r in csv.reader(line for line in lines if line.endswith("\n")) if len(r) == 3]
            while index < len(rows):
                duration = float(rows[index][2]) - float(rows[index][1])
                if duration < 1:
                    index += 1
                    continue
                quiet += duration
                index += 1
                # Include the following segment for the payoff after an audio reaction.
                spike = False
                if index - first >= 3:
                    name, start, end = rows[index-2]
                    start, end = float(start), float(end)
                    chat_times = chat.times(start, end + 5)
                    piece = job.folder / name
                    jumps = audio_peaks(piece) if media.probe(piece).audio else []
                    signals.extend((start + t, "audio") for t in jumps)
                    signals.extend((t, "chat") for t in chat_times)
                    spike = bool(chat_times or jumps)
                if spike or quiet >= 180 or (ended and index == len(rows)):
                    self._batch(job, rows[first:index], seen, signals)
                    signals = [(t, kind) for t, kind in signals if t >= float(rows[index-1][1])]
                    keep = index-1
                    for name, _, _ in rows[first:keep]:
                        (job.folder / name).unlink(missing_ok=True)
                    first, quiet = keep, 0
            if ended:
                if quiet >= 3:
                    self._batch(job, rows[first:index], seen, signals)
                if shutil.disk_usage(job.folder).free < 2 * 1024**3:
                    raise media.EditError("Наблюдение остановлено: на диске осталось менее 2 ГиБ.")
                if not stop.is_set() and proc.returncode:
                    raise media.EditError("Запись эфира прервалась. Завершённые подборки сохранены.")
                return
            time.sleep(1)

    def _batch(self, parent, rows, seen, signals=()):
        from dataclasses import asdict
        origin = float(rows[0][1])
        channel = parent.read()["channel"]
        child = self.manager.create("", f"{channel} · эфир +{int(origin)//60} мин")
        child.update(live_pending=True, live_origin=origin,
                     live_signals=[(round(t-origin, 2), kind) for t, kind in signals if t >= origin])
        manifest = parent.folder / "window.txt"
        manifest.write_text("".join(f"file '{r[0]}'\n" for r in rows))
        try:
            target = child.folder / "source.mkv"
            media.ffmpeg("-f", "concat", "-safe", "1", "-i", str(manifest), "-c", "copy", str(target))
            info = media.probe(target)
            media.still(target, min(1, info.duration/2), child.folder / "source.jpg")
            child.update(status="analyzing", operation="analyze", previous="source_ready",
                         source=target.name, source_info=asdict(info))
            child.analyze()
            clips = []
            for clip in child.read()["clips"]:
                peak = origin + clip["peak"]
                if all(abs(peak-other) > 5 for other in seen):
                    clips.append(clip)
                    seen.append(peak)
            if clips:
                child.update(clips=clips, live_pending=False)
                parent.update(batches=parent.read()["batches"] + [child.id])
            else:
                shutil.rmtree(child.folder)
            parent.progress(f"Проверено до {int(float(rows[-1][2]))} с эфира · найдено {len(seen)}", 0, 0)
        except Exception as exc:
            child.update(status="failed", operation="analyze", previous="source_ready",
                         error=str(exc) if isinstance(exc, media.EditError) else "Не удалось проверить окно эфира.", progress=None)
            raise media.EditError("Ошибка анализа окна эфира. Подробности в задаче " + child.id + ".") from exc
