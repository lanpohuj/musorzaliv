"""Unit tests for live stream pipeline improvements: relaxed thresholds, async batching, 60s overlap, and signal fallback."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest
from unittest.mock import MagicMock, patch

from pipeline import twitch_editor as editor
from pipeline import twitch_media as media
from pipeline.twitch_live import LiveSessions, ChatPulse
from pipeline.twitch_workflow import Jobs, Job


class LivePipelineRealtimeThresholdsTests(unittest.TestCase):
    def test_realtime_relaxed_thresholds_and_pool_size(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = Jobs(root / "jobs", root / "exports")
            job = jobs.create("", "Live Batch Test")
            video_file = job.folder / "source.mkv"

            # Create synthetic 120s video
            media.ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15",
                         "-f", "lavfi", "-i", "sine=frequency=400:sample_rate=16000",
                         "-t", "120", "-c:v", "libx264", "-c:a", "pcm_s16le", str(video_file))

            job.update(
                source=video_file.name,
                source_info={"duration": 120.0, "width": 320, "height": 180, "audio": True},
                live_origin=0.0,
                live_signals=[(35.0, "audio"), (75.0, "chat")],
                vod=None,
            )

            # Mock scan to return 6 completely distinct events
            titles = ["Клатч с AWP", "Смешной донат", "Падение в лаву", "Победа в раунде", "Крик ужаса", "Открытие кейса"]
            scan_events = [
                editor.Event(start=float(i * 19.0), peak=float(i * 19.0 + 3.0), end=float(i * 19.0 + 5.0),
                             title=titles[i], kind="play" if i % 2 == 0 else "reaction",
                             evidence=f"Улика {i}", reason=f"Причина {i}", hook=70, payoff=70, standalone=70, energy=70)
                for i in range(6)
            ]

            # Mock refine to return score 58 (which would fail old score >= 65, but passes score >= 56)
            # and peak drift of 20s (which would fail old drift <= 15, but passes drift <= 25)
            refine_event = editor.Event(start=0.0, peak=28.0, end=17.0, title="Refined Play", kind="play",
                                        evidence="Kill", reason="Play", hook=60, payoff=60, standalone=60, energy=60)

            class MockGemini:
                model = "gemini-mock"
                def scan(self, path, duration):
                    return editor.Scan(events=scan_events), {"input": 10, "output": 10}

                def nominate(self, group):
                    # In realtime, nominate returns indices
                    return editor.Nominations(indices=[0, 1, 2, 3, 4]), {"input": 10, "output": 10}

                def refine(self, path, seed, hints, duration):
                    return editor.Refinement(event=refine_event, cameras=[]), {"input": 10, "output": 10}

                def judge_opening(self, path, dialogue, duration):
                    # Novelty 55 (would fail old novelty >= 78, but passes new novelty >= 52)
                    return editor.OpeningReview(
                        publish=True, novelty=55, hook_at=1.0, payoff_at=10.0,
                        opening="Opening", payoff="Payoff"
                    ), {"input": 10, "output": 10}

                def close(self):
                    pass

            class MockSpeech:
                def from_video(self, path, work, has_audio, vad_filter=True):
                    return []
                def read(self, path, language, vad_filter=True):
                    return []

            with patch.object(editor, "create_editor", return_value=MockGemini()), \
                 patch.object(editor, "LocalSpeech", return_value=MockSpeech()), \
                 patch.object(media, "has_visual_change", return_value=True):
                job.analyze()

            state = job.read()
            self.assertEqual(state["status"], "review")
            self.assertGreater(len(state["clips"]), 0, "Realtime clips should be found with relaxed thresholds")
            debug_stats = state.get("debug_stats", {})
            self.assertTrue(debug_stats.get("realtime"))
            self.assertGreaterEqual(debug_stats.get("nominated", 0), 4, "Nominated pool should retain up to 5-6 candidates, not 2")
            self.assertGreater(debug_stats.get("refined", 0), 0)

    def test_realtime_signal_fallback_on_empty_refined(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = Jobs(root / "jobs", root / "exports")
            job = jobs.create("", "Live Fallback Test")
            video_file = job.folder / "source.mkv"

            media.ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15",
                         "-f", "lavfi", "-i", "sine=frequency=400:sample_rate=16000",
                         "-t", "120", "-c:v", "libx264", "-c:a", "pcm_s16le", str(video_file))

            job.update(
                source=video_file.name,
                source_info={"duration": 120.0, "width": 320, "height": 180, "audio": True},
                live_origin=0.0,
                live_signals=[(45.0, "audio"), (85.0, "chat")],
                vod=None,
            )

            class MockGeminiDroppingRefine:
                model = "gemini-mock"
                def scan(self, path, duration):
                    ev = editor.Event(start=30.0, peak=45.0, end=60.0, title="Audio Jump", kind="reaction",
                                      evidence="Loud scream", reason="Reaction", hook=70, payoff=70, standalone=70, energy=85)
                    return editor.Scan(events=[ev]), {"input": 10, "output": 10}

                def nominate(self, group):
                    return editor.Nominations(indices=[0]), {"input": 10, "output": 10}

                def refine(self, path, seed, hints, duration):
                    # Refine drops event (returns None)
                    return editor.Refinement(event=None, cameras=[]), {"input": 10, "output": 10}

                def close(self):
                    pass

            class MockSpeech:
                def from_video(self, path, work, has_audio, vad_filter=True):
                    return []
                def read(self, path, language, vad_filter=True):
                    return []

            with patch.object(editor, "create_editor", return_value=MockGeminiDroppingRefine()), \
                 patch.object(editor, "LocalSpeech", return_value=MockSpeech()):
                job.analyze()

            state = job.read()
            self.assertEqual(state["status"], "review")
            self.assertEqual(len(state["clips"]), 1, "Should pass best candidate from pool via signal fallback")
            clip = state["clips"][0]
            self.assertEqual(clip.get("live_rating"), "signal_fallback")

    def test_realtime_120s_window_passes_highlight_score58_novelty60(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = Jobs(root / "jobs", root / "exports")
            job = jobs.create("", "Live 120s Score 58 Test")
            video_file = job.folder / "source.mkv"

            media.ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15",
                         "-f", "lavfi", "-i", "sine=frequency=400:sample_rate=16000",
                         "-t", "120", "-c:v", "libx264", "-c:a", "pcm_s16le", str(video_file))

            job.update(
                source=video_file.name,
                source_info={"duration": 120.0, "width": 320, "height": 180, "audio": True},
                live_origin=0.0,
                live_signals=[(42.0, "chat")],
                vod=None,
            )

            scan_event = editor.Event(
                start=20.0, peak=42.0, end=50.0, title="Epic Play Seed", kind="play",
                evidence="Great move", reason="Play", hook=58, payoff=58, standalone=58, energy=58
            )
            refine_event = editor.Event(
                start=18.0, peak=44.0, end=52.0, title="Epic Play Refined", kind="play",
                evidence="Great move refined", reason="Play", hook=58, payoff=58, standalone=58, energy=58
            )

            class MockGemini58:
                model = "gemini-mock"
                def scan(self, path, duration):
                    return editor.Scan(events=[scan_event]), {"input": 10, "output": 10}

                def nominate(self, group):
                    return editor.Nominations(indices=[0]), {"input": 10, "output": 10}

                def refine(self, path, seed, hints, duration):
                    return editor.Refinement(event=refine_event, cameras=[]), {"input": 10, "output": 10}

                def judge_opening(self, path, dialogue, duration):
                    # Novelty 60 (passes realtime novelty >= 52)
                    return editor.OpeningReview(
                        publish=True, novelty=60, hook_at=1.0, payoff_at=8.0,
                        opening="Opening hook", payoff="Great payoff"
                    ), {"input": 10, "output": 10}

                def close(self):
                    pass

            class MockSpeech:
                def from_video(self, path, work, has_audio, vad_filter=True):
                    return []
                def read(self, path, language, vad_filter=True):
                    return []

            with patch.object(editor, "create_editor", return_value=MockGemini58()), \
                 patch.object(editor, "LocalSpeech", return_value=MockSpeech()), \
                 patch.object(media, "has_visual_change", return_value=True):
                job.analyze()

            state = job.read()
            self.assertEqual(state["status"], "review")
            self.assertEqual(len(state["clips"]), 1, "Should retain 1 clip with score 58 and novelty 60")
            clip = state["clips"][0]
            self.assertEqual(clip["title"], "Epic Play Refined")
            self.assertEqual(clip["evidence"], "Great payoff")
            self.assertEqual(clip["reason"], "Opening hook")
            self.assertGreaterEqual(clip["score"], 58.0)


class LiveSessionsAsyncBatchingTests(unittest.TestCase):
    def test_live_sessions_async_consume_window_and_overlap(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = Jobs(root / "jobs", root / "exports")
            live = LiveSessions(jobs)
            parent = jobs.create("", "Live Stream Test")
            parent.update(channel="streamer", batches=[])

            # Create 6 fake ts piece files of 30 seconds each
            piece_names = []
            for i in range(6):
                name = f"piece-{i:08d}.ts"
                target = parent.folder / name
                # Make small dummy file
                media.ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=15",
                             "-t", "1", "-c:v", "libx264", "-f", "mpegts", str(target))
                piece_names.append(name)

            manifest = parent.folder / "pieces.csv"
            # 6 segments of 30s = 180s total
            manifest_lines = [
                f"{name},{i * 30.0},{(i + 1) * 30.0}\n"
                for i, name in enumerate(piece_names)
            ]
            manifest.write_text("".join(manifest_lines))

            class FakeProc:
                poll_count = 0
                returncode = 0
                def poll(self):
                    self.poll_count += 1
                    # End after 3 iterations
                    return 0 if self.poll_count >= 3 else None

            class FakeChat:
                def times(self, start, end):
                    return []

            class FakeExecutor:
                def __init__(self):
                    self.submitted = []
                def submit(self, fn, *args):
                    self.submitted.append((fn, args))

            fake_executor = FakeExecutor()
            stop = MagicMock()
            stop.is_set.return_value = False
            stop.wait.return_value = False

            with patch.object(media, "has_visual_change", return_value=True):
                live._consume(parent, FakeProc(), stop, FakeChat(), executor=fake_executor)

            # At least one batch should have been dispatched via executor
            self.assertGreater(len(fake_executor.submitted), 0)

            # Check that the overlap segments (segments 4 and 5) were kept while segment 0, 1, 2 were unlinked
            # Segment 0, 1 should be deleted after keeping 2 overlap segments
            self.assertFalse((parent.folder / piece_names[0]).exists())
            self.assertFalse((parent.folder / piece_names[1]).exists())

    def test_live_sessions_batch_error_resilience(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = Jobs(root / "jobs", root / "exports")
            live = LiveSessions(jobs)
            parent = jobs.create("", "Error Resilience Test")
            parent.update(channel="streamer", batches=[])

            piece_file = parent.folder / "piece.ts"
            media.ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=15",
                         "-t", "2", "-c:v", "libx264", "-f", "mpegts", str(piece_file))

            rows = [["piece.ts", "0", "30"]]
            seen = []

            # Mock Job.analyze to raise an unexpected exception
            with patch.object(Job, "analyze", side_effect=RuntimeError("Unexpected model failure")):
                # _batch must NOT raise an unhandled exception or kill the caller
                try:
                    live._batch(parent, rows, seen)
                except Exception as e:
                    self.fail(f"_batch should not re-raise error to crash caller, but raised: {e}")

            # Parent batches should remain empty, but parent job itself did not fail
            self.assertEqual(len(parent.read()["batches"]), 0)


class ChatPulseHighVolumeAndSyncTests(unittest.TestCase):
    def test_chatpulse_high_volume_spike_detection(self):
        chat = ChatPulse()

        # Simulate baseline: 100 messages/sec for 100 seconds from 80 distinct users per second
        for sec in range(1, 101):
            for i in range(100):
                author = f"user_{i % 80}"
                chat.message(sec, author, "chat message")

        # In baseline, no spikes should be detected
        self.assertEqual(chat.times(10, 90), [], "No spikes should trigger during constant 100 msg/s baseline")

        # Surge at second 105: 250 messages from 180 distinct users
        for i in range(250):
            author = f"surge_user_{i % 180}"
            chat.message(105, author, "LETS GOOOOO W")

        # Spike at second 105 must be detected
        bursts = chat.times(100, 110)
        self.assertGreater(len(bursts), 0, "Spike at second 105 must be detected under high baseline")
        self.assertIn(105, bursts)

        # Ensure memory bounded: 1-second buckets maxlen is 180
        self.assertLessEqual(len(chat.buckets), 180)

    def test_chatpulse_single_user_spam_filtered(self):
        chat = ChatPulse()

        # 300 messages at second 50 from a single user
        for _ in range(300):
            chat.message(50, "spammer_bot", "BUY FOLLOWERS NOW")

        self.assertFalse(chat.between(45, 55), "Single spammer must not trigger a chat pulse spike")

    def test_chatpulse_split_irc_chunks_reassembly(self):
        chat = ChatPulse()
        stop = Event()

        chunks = [
            b":user1!u1@tmi.twitch.tv PRIV",
            b"MSG #kaicenat :POGGERS in the chat\r\n:user2!u2@tmi.twitch.tv PRI",
            b"VMSG #kaicenat :W STREAMER\r\n",
            b":user3!u3@tmi.twitch.tv PRIVMSG #kaicenat :Clip that right now!\r\n",
        ]

        class FakeSocket:
            def __init__(self):
                self.chunks = list(chunks)
            def recv(self, size):
                if self.chunks:
                    return self.chunks.pop(0)
                stop.set()
                return b""
            def sendall(self, data):
                pass
            def settimeout(self, t):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        with patch("socket.create_connection", return_value=FakeSocket()), \
             patch("ssl.create_default_context") as mock_ssl:
            mock_ssl.return_value.wrap_socket.return_value = FakeSocket()
            chat.listen("kaicenat", stop)

        all_nicks = set()
        for b in chat.buckets:
            all_nicks.update(b["nicks"])
        self.assertIn("user1", all_nicks)
        self.assertIn("user2", all_nicks)
        self.assertIn("user3", all_nicks)

    def test_chatpulse_clock_sync_with_first_segment(self):
        chat = ChatPulse()
        chat.started = 1000.0

        # Message before sync
        chat.message(10, "viewer1", "stream starting!")
        self.assertEqual(chat.buckets[0]["sec"], 10)

        # Sync at monotonic 1035 with segment ending at 30.0s (buffer_delay=2.5)
        # stream_time_now = 30.0 - 2.5 = 27.5
        # new_started = 1035.0 - 27.5 = 1007.5
        # delta = int(1000.0 - 1007.5) = -7
        with patch("time.monotonic", return_value=1035.0):
            chat.sync_origin(stream_end=30.0, buffer_delay=2.5)

        self.assertTrue(chat.synced)
        # Verify early message shifted: 10 + (-7) = 3
        self.assertEqual(chat.buckets[0]["sec"], 3)


if __name__ == "__main__":
    unittest.main()
