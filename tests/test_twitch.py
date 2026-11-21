"""Regression tests for the Twitch editor's persistent and frame-aligned workflow."""

from copy import deepcopy
from io import BytesIO
import base64
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch
import wave

from fastapi.testclient import TestClient
import httpx
import numpy as np
from PIL import Image

from pipeline import twitch_editor as editor
from pipeline import twitch_media as media
from pipeline import twitch_river as river
from pipeline.twitch_web import create_app
from pipeline.twitch_workflow import Jobs


def event(start, peak, end, title="Клатч", **fields):
    return editor.Event(start=start, peak=peak, end=end, title=title, kind="play",
                        evidence="Видно, как игрок выигрывает раунд.", reason="Ясная развязка.",
                        hook=85, payoff=95, standalone=88, energy=80, **fields)


class EditorChecks(unittest.TestCase):
    def test_late_hook_trims_leadin_and_routine_action_is_rejected(self):
        seed = {"start": 100, "end": 125, "score": 88}
        verdict = editor.OpeningReview(publish=True, novelty=85, hook_at=15, payoff_at=19,
                                       opening="За секунду до аварии", payoff="Пассажир встаёт с сена")
        clip = editor.apply_opening(seed, verdict, 78)
        self.assertEqual((clip["start"], clip["peak"], clip["end"]), (114.65, 119, 125))
        self.assertIsNone(editor.apply_opening(seed, verdict.model_copy(update={"publish": False}), 78))
        self.assertIsNone(editor.apply_opening(seed, verdict.model_copy(update={"payoff_at": 30}), 78))

    def test_vod_uses_reserve_and_keeps_verified_hook_below_old_threshold(self):
        usage = {"input": 0, "output": 0}

        class FakeEditor:
            def scan(self, video, duration):
                return editor.Scan(events=[event(4, 10, 15, title="Сюрприз")]), usage

            def nominate(self, candidates):
                return editor.Nominations(indices=[]), usage

            def edit_highlight(self, video, transcript, duration, seed):
                return editor.HighlightEdit(
                    publish=True, start_at=4, hook_at=4.5, payoff_at=10, end_at=14,
                    title="Неожиданная развязка", opening="Сомнительное обещание", payoff="Видно результат",
                    novelty=72, cameras=[]), usage

            def verify_highlight(self, video, transcript, duration):
                return editor.OpeningReview(publish=True, hook_at=.5, payoff_at=6,
                                            opening="Сомнительное обещание", payoff="Видно результат",
                                            novelty=72), usage

            def rank_highlights(self, clips):
                return editor.HighlightRanking(indices=[0]), usage

            def close(self):
                pass

        with TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = Jobs(root / "jobs", root / "exports")
            job = jobs.create("", "Архив")
            job.update(source="fixture.mp4", source_info={"duration": 3600, "audio": False})
            with (patch.object(editor, "create_editor", return_value=FakeEditor()),
                  patch.object(media, "cloud_window", side_effect=lambda src, lo, hi, dst, **kw: dst.touch())):
                job.analyze()
            result = job.read()
            self.assertEqual(result["finalists"], 2)
            self.assertEqual(len(result["clips"]), 1)
            clip = result["clips"][0]
            self.assertEqual(clip["score"], 72)
            self.assertAlmostEqual(clip["hook_time"] - clip["start"], .5)

    def test_viewer_clip_uses_end_offset_and_same_archive(self):
        from pipeline.twitch_audience import seeds
        def clip(vod, end, views):
            return {"node": {"video": {"id": vod}, "title": "Clip", "durationSeconds": 35,
                             "videoOffsetSeconds": end, "viewCount": views}}
        found = seeds([clip("other", 100, 5000), clip("vod", 25092, 30),
                       clip("vod", 25089, 28000)], "vod", 52102)
        self.assertEqual([(s["start"], s["end"]) for s in found], [(25054, 25089)])

    def test_live_chat_spike_needs_distinct_viewers(self):
        from pipeline.twitch_live import ChatPulse, channel_name
        chat = ChatPulse()
        for _ in range(60):
            chat.message(185, "same", "LOL")
        self.assertFalse(chat.between(180, 190))
        for i in range(12):
            chat.message(400, str(i), "LOL")
        self.assertTrue(chat.between(396, 402))
        self.assertIsNone(channel_name("https://www.twitch.tv/videos/123456"))
        self.assertEqual(channel_name("https://www.twitch.tv/CaseOh_"), "caseoh_")

    def test_live_publishes_first_batch_and_suppresses_repeat(self):
        from pipeline.twitch_live import LiveSessions
        from pipeline.twitch_workflow import Job
        with TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = Jobs(root / "jobs", root / "exports")
            live = LiveSessions(jobs)
            parent = jobs.create("", "Live")
            parent.update(channel="fixture", batches=[])
            media.ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15",
                         "-t", "4", "-c:v", "libx264", "-f", "mpegts", str(parent.folder / "piece.ts"))
            self.assertTrue(media.has_visual_change(parent.folder / "piece.ts", 0, 3))
            rows = [["piece.ts", "0", "4"]]
            seen = []
            with patch.object(Job, "analyze", lambda job: job.update(status="review", clips=[{"peak": 2}])):
                live._batch(parent, rows, seen)
                live._batch(parent, rows, seen)
            self.assertEqual(seen, [2])
            batches = parent.read()["batches"]
            self.assertEqual(len(batches), 1)
            self.assertTrue(jobs.get(batches[0]).source.is_file())
            self.assertEqual(len(jobs.all()), 2)

    def test_last_voice_can_finish_between_video_frames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            take = root / "take.wav"
            with wave.open(str(take), "wb") as output:
                output.setparams((1, 2, media.RATE, 0, "NONE", ""))
                output.writeframes(bytes(round(.94 * media.RATE) * 2))
            media.voice_track([{"start": 1, "end": 1.94, "path": take}], 1.94, root / "mix.wav")
            with wave.open(str(root / "mix.wav")) as track:
                self.assertGreaterEqual(track.getnframes(), round(1.94 * media.RATE))

    def test_all_vod_seconds_are_covered(self):
        windows = editor.window_ranges(7312.5)
        self.assertEqual(windows[0], (0, 150))
        self.assertEqual(windows[-1][1], 7312.5)
        for previous, following in zip(windows, windows[1:]):
            self.assertGreaterEqual(previous[1] - following[0], 20)

    def test_dedupe_keeps_distinct_adjacent_plays(self):
        first = editor.checked_event(event(2, 15, 25), 0, 90)
        duplicate = editor.checked_event(event(0, 6.5, 16), 9, 90)
        adjacent = editor.checked_event(event(15, 21, 28, title="Погоня"), 0, 90)
        self.assertEqual(editor.deduplicate([duplicate, adjacent, first]), [first, adjacent])
        self.assertIsNone(editor.checked_event(event(30, 40, 80), 0, 70))

    def test_pause_uses_completed_replies(self):
        clip = {"start": 100, "end": 120, "peak": 113, "turns": [
            {"start": 0, "end": 5}, {"start": 9, "end": 12}, {"start": 12.01, "end": 18.5},
        ]}
        self.assertEqual(media.choose_ad_point(clip), 7)
        crossing_peak = {"start": 1, "end": 11, "peak": 7, "turns": [{"start": .3, "end": 2}]}
        self.assertTrue(3 <= media.choose_ad_point(crossing_peak) <= 4.8)
        clip["turns"] = [{"start": 0, "end": 20}]
        self.assertIsNone(media.choose_ad_point(clip))

    def test_side_camera_is_cut_symmetrically_from_gameplay(self):
        source = media.Media(24, 1280, 720, True)
        left_camera = media._camera_crop(source, [0, .128, .209, .274])
        right_camera = media._camera_crop(source, [.791, .128, .209, .274])
        self.assertEqual(left_camera, (268, 198, 0, 92))
        self.assertEqual(right_camera, (268, 198, 1012, 92))
        left_game = media._gameplay_crop(source, left_camera, 1080 / 1098)
        right_game = media._gameplay_crop(source, right_camera, 1080 / 1098)
        self.assertEqual(left_game, (744, 720, 268, 0))
        self.assertEqual(right_game, left_game)
        centered_camera = media._camera_crop(source, [.31, 0, .38, .25])
        centered_game = media._gameplay_crop(source, centered_camera, 1)
        self.assertEqual(centered_game, (1280, 540, 0, 180))
        # A wider output fits more gameplay by shifting to the right of the webcam.
        wide_game = media._gameplay_crop(source, left_camera, 1.25)
        self.assertEqual(wide_game, (1012, 720, 268, 0))

    def test_subtitles_leave_the_ad_interval_empty(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "test.ass"
            media.write_ass(path, [{"start": 1.5, "end": 3, "text": "Русская реплика"}], 2, 1.4,
                            positions=[(0, 2.5, 910), (2.5, 8, 1490)])
            text = path.read_text()
        self.assertIn("0:00:01.50,0:00:02.00", text)
        self.assertIn(r"0:00:03.40,0:00:03.90,Subtitle,,0,0,0,,{\pos(540,910)}", text)
        self.assertIn(r"0:00:03.90,0:00:04.40,Subtitle,,0,0,0,,{\pos(540,1490)}", text)
        self.assertIn("РУССКАЯ РЕПЛИКА", text)

    def test_small_boundary_snap_preserves_the_opening_word(self):
        clip = editor.checked_event(event(2, 6, 10), 50, 40)
        snapped = editor.snap_speech(clip, [{"start": 1.65, "end": 2.6}], 50, 90)
        self.assertEqual(snapped["start"], 51.59)
        self.assertEqual(snapped["peak"], 56)

    def test_gameplay_cut_keeps_a_nearby_reaction(self):
        clip = editor.checked_event(event(2, 10, 10.6), 100, 30)
        reaction = [{"start": 11.25, "end": 13.3, "text": "We won!"}]
        self.assertEqual(editor.include_reaction_tail(clip, reaction, 100, 130)["end"], 113.42)
        self.assertEqual(editor.include_reaction_tail(clip, reaction, 100, 112)["end"], 110.6)


class WorkflowChecks(unittest.TestCase):
    def test_mocked_providers_real_video_and_audio(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, ad, dub = (root / "source.mp4", root / "ad.mp4", root / "dub.wav")
            media.ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30",
                         "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000",
                         "-t", "14", "-vf", "eq=brightness=0.15*sin(t):eval=frame",
                         "-c:v", "libx264", "-threads", "2", "-c:a", "aac", str(source))
            media.ffmpeg("-f", "lavfi", "-i", "color=magenta:size=192x320:rate=25",
                         "-f", "lavfi", "-i", "sine=frequency=1050:sample_rate=48000",
                         "-t", "1.4", "-c:v", "libx264", "-threads", "2", "-c:a", "aac", str(ad))
            media.ffmpeg("-f", "lavfi", "-i", "sine=frequency=600:sample_rate=48000",
                         "-af", "atrim=duration=0.85,adelay=300,apad=pad_dur=1",
                         "-ac", "1", "-c:a", "pcm_s16le", str(dub))

            usage = {"input": 15, "output": 5}
            calls = {"scan": 0, "refine": 0, "adapt": 0, "fish": []}

            class FakeGemini:
                model = editor.GEMINI_MODEL

                def scan(self, path, duration):
                    assert media.probe(path).audio
                    calls["scan"] += 1
                    return editor.Scan(events=[event(1, 8, 12)]), usage

                def refine(self, path, seed, transcript, duration):
                    calls["refine"] += 1
                    if calls["refine"] == 1:
                        raise media.EditError("Тестовое прерывание сети")
                    return editor.Refinement(event=event(1, 8, 12), cameras=[
                        editor.Camera(start=1, end=3.5, box=[50, 50, 350, 400]),
                        editor.Camera(start=4, end=12, box=[360, 60, 300, 400]),
                    ]), usage

                def judge_opening(self, path, dialogue, duration):
                    return editor.OpeningReview(publish=True, novelty=91, hook_at=0, payoff_at=7, opening="Go!", payoff="Victory"), usage

                def locate_camera(self, path, duration):
                    return editor.CameraReview(cameras=[
                        editor.Camera(start=0, end=2.5, box=[50, 50, 350, 400]),
                        editor.Camera(start=3, end=11, box=[360, 60, 300, 400])]), usage

                def adapt(self, path, transcript, duration):
                    calls["adapt"] += 1
                    return editor.Script(turns=[
                        editor.Turn(start=0.5, end=2.0, role="streamer", english="Go!",
                                    russian="Погнали!", emotion="excited"),
                        editor.Turn(start=8.5, end=10, role="guest", english="Wait!",
                                    russian="Подожди!", emotion="neutral"),
                    ], cameras=[editor.Camera(start=0, end=2.5, box=[50, 50, 350, 400]),
                                editor.Camera(start=3, end=11, box=[360, 60, 300, 400])]), usage

                def close(self):
                    pass

            class FakeFish:
                model = editor.FISH_MODEL

                def synthesize(self, voice_id, text, tone):
                    calls["fish"].append((voice_id, text))
                    return dub.read_bytes()

                def close(self):
                    pass

            class FakeSpeech:
                def from_video(self, path, work, has_audio, vad_filter=True):
                    return [{"start": 1.3, "end": 2, "text": "Go!"}] if has_audio else []

                def read(self, path, language, vad_filter=True):
                    return []  # Also verifies timed phrase captions on ambiguous word alignment.

            jobs = Jobs(root / "jobs", root / "output")
            with (patch.object(editor, "GeminiEditor", FakeGemini),
                  patch.object(editor, "FishVoices", FakeFish),
                  patch.object(editor, "LocalSpeech", FakeSpeech),
                  patch.dict("os.environ", {"GEMINI_API_KEY": "fixture", "FISH_API_KEY": "fixture",
                                            "TWITCH_ANALYSIS_PROVIDER": "gemini"}),
                  TestClient(create_app(jobs), base_url="http://localhost") as client):
                token = client.get("/api/status").json()["token"]
                headers = {"x-twitch-token": token}
                self.assertEqual(client.get("/").status_code, 200)
                self.assertEqual(client.get("/ui/studio.js").status_code, 200)
                self.assertEqual(client.post("/api/jobs", json={}).status_code, 403)
                job = client.post("/api/jobs", headers=headers, json={"label": "Fixture"}).json()
                path = "/api/jobs/" + job["id"]

                def wait_for(expected):
                    for _ in range(360):
                        current = client.get(path).json()
                        if current["status"] in {expected, "failed"}:
                            self.assertEqual(current["status"], expected, current.get("error"))
                            return current
                        time.sleep(.05)
                    self.fail(f"Timed out waiting for {expected}")

                def action(name, **values):
                    response = client.post(path + "/action", headers=headers, json={"action": name, **values})
                    self.assertEqual(response.status_code, 200, response.text)

                self.assertEqual(client.put(path + "/asset/source", params={"name": "fixture.mp4"},
                                            headers=headers, content=source.read_bytes()).status_code, 200)
                job = wait_for("source_ready")
                self.assertEqual(job["estimate"]["windows"], 1)
                self.assertEqual(client.put(path + "/asset/ad", params={"name": "blue.mp4"},
                                            headers=headers, content=ad.read_bytes()).status_code, 200)
                voices = {"streamer": "a" * 32, "guest": "b" * 32}
                action("analyze", voices=voices)
                wait_for("failed")
                action("retry")
                job = wait_for("review")
                self.assertEqual((calls["scan"], calls["refine"]), (1, 2))
                self.assertEqual(len(job["clips"]), 1)
                clip_id = job["clips"][0]["id"]
                sample = client.get(f"{path}/media/{clip_id}/reference", headers={"Range": "bytes=0-100"})
                self.assertEqual((sample.status_code, len(sample.content)), (206, 101))

                action("script", clips=job["clips"])
                job = wait_for("scripted")
                self.assertEqual(calls["adapt"], 1)
                edited = deepcopy(job["clips"])
                edited[0]["turns"][1]["russian"] = "Стой!"
                edited[0]["ad_at"] = 5
                action("preview", clips=edited, voices=voices)
                job = wait_for("preview")
                self.assertEqual(calls["fish"], [(voices["streamer"], "Погнали!"), (voices["guest"], "Стой!")])
                self.assertEqual(job["clips"][0]["rendered_duration"], 12.4)
                preview = client.get(f"{path}/media/{clip_id}/preview")
                self.assertEqual(preview.status_code, 200)

                action("export")
                job = wait_for("done")
                final = Path(job["output"]) / job["files"][0]
                media.verify_export(final, (1080, 1920), 12.4)
                manifest = root / "jobs" / job["id"] / clip_id / "prepared.json"
                cues = json.loads(manifest.read_text())["cues"]
                self.assertEqual([cue["text"] for cue in cues], ["Погнали!", "Стой!"])

            audio = root / "final.wav"
            media.ffmpeg("-i", str(final), "-vn", "-ar", "48000", "-ac", "1",
                         "-c:a", "pcm_s16le", str(audio))
            with wave.open(str(audio), "rb") as track:
                samples = np.frombuffer(track.readframes(track.getnframes()), dtype="<i2")

            def tone(at):
                chunk = samples[round(at * 48000):][:4800]
                return int(np.argmax(abs(np.fft.rfft(chunk)))) * 10

            self.assertEqual(tone(0.75), 600)
            self.assertEqual(tone(3), 300)
            self.assertEqual(tone(5.3), 1050)
            self.assertEqual(tone(6.75), 300)
            self.assertEqual(tone(10.1), 600)

            def frame(at):
                # The advertising box is centered. The outside pixels must freeze independently.
                raw = root / "frame.raw"
                media.ffmpeg("-ss", str(at), "-i", str(final), "-an", "-frames:v", "1",
                             "-vf", "crop=90:400:0:920,format=gray", "-f", "rawvideo", str(raw))
                return np.frombuffer(raw.read_bytes(), dtype=np.uint8).astype(np.int16)

            self.assertLess(np.abs(frame(5.2) - frame(6.1)).mean(), 0.5)
            self.assertGreater(np.abs(frame(5.2) - frame(7)).mean(), 1)


class RiverChecks(unittest.TestCase):
    def test_key_selects_river_and_keeps_explicit_gemini(self):
        with patch.dict("os.environ", {"RIVER_API_KEY": "fixture"}, clear=True):
            self.assertEqual(editor.analysis_config(), ("River", river.RIVER_MODEL, True))
            self.assertEqual(editor.estimate(3600)["provider"], "River")
            self.assertGreater(editor.estimate(3600)["usd"], 0)
        with patch.dict("os.environ", {"RIVER_API_KEY": "fixture", "GEMINI_API_KEY": "fixture",
                                     "TWITCH_ANALYSIS_PROVIDER": "gemini"}, clear=True):
            self.assertEqual(editor.analysis_config(), ("Gemini", editor.GEMINI_MODEL, True))

    def test_river_receives_frames_transcript_and_strict_schema(self):
        calls = []

        def reply(request):
            calls.append(request)
            self.assertEqual(request.url.path, "/v1/chat/completions")
            payload = json.loads(request.content)
            content = payload["messages"][0]["content"]
            self.assertIn("Fire!", content[0]["text"])
            self.assertIn("contact sheets", content[0]["text"])
            self.assertEqual(len(content), 3)  # 14 frames across two sheets.
            for part in content[1:]:
                url = part["image_url"]["url"]
                with Image.open(BytesIO(base64.b64decode(url.split(",")[1]))) as board:
                    self.assertEqual(board.format, "JPEG")
                    self.assertEqual(board.size[0], 384 * river.BOARD_COLUMNS)
            schema = payload["response_format"]["json_schema"]
            self.assertTrue(schema["strict"])
            self.assertFalse(schema["schema"]["$defs"]["Event"]["additionalProperties"])
            result = {"events": [event(1, 3, 5).model_dump()]}
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}],
                                             "usage": {"prompt_tokens": 100, "completion_tokens": 10}})

        with TemporaryDirectory() as directory, patch.dict("os.environ", {"RIVER_API_KEY": "fixture"}):
            path = Path(directory) / "video.mp4"
            media.ffmpeg("-f", "lavfi", "-i", "testsrc2=size=96x54:rate=10",
                         "-t", "7", "-c:v", "libx264", str(path))
            model = river.RiverEditor()
            model.close()
            model.client = httpx.Client(base_url=river.RIVER_URL, transport=httpx.MockTransport(reply))
            try:
                raw, usage = model.scan(path, 7, transcript=[{"start": 1, "end": 2, "text": "Fire!"}])
                self.assertEqual(raw.events[0].peak, 3)
                self.assertEqual(usage, {"input": 100, "output": 10})
                self.assertEqual(len(calls), 1)
            finally:
                model.close()
