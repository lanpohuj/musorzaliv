import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

from pipeline import twitch_editor as editor
from pipeline import twitch_media as media


class DubSyncTests(unittest.TestCase):
    def test_align_turns_to_speech_snaps_to_word_timestamps(self):
        script = editor.Script(
            turns=[
                editor.Turn(
                    start=1.0,
                    end=3.5,
                    role="streamer",
                    english="What an incredible shot",
                    russian="Какой невероятный выстрел!",
                    emotion="excited",
                ),
                editor.Turn(
                    start=6.0,
                    end=8.0,
                    role="guest",
                    english="No way",
                    russian="Да ладно!",
                    emotion="surprised",
                ),
            ],
            cameras=[],
        )

        transcript = [
            {
                "start": 1.2,
                "end": 3.1,
                "text": "What an incredible shot",
                "words": [
                    {"word": "What", "start": 1.22, "end": 1.45},
                    {"word": "an", "start": 1.48, "end": 1.60},
                    {"word": "incredible", "start": 1.62, "end": 2.40},
                    {"word": "shot", "start": 2.45, "end": 3.05},
                ],
            },
            {
                "start": 6.3,
                "end": 7.5,
                "text": "No way",
                "words": [
                    {"word": "No", "start": 6.35, "end": 6.65},
                    {"word": "way", "start": 6.70, "end": 7.42},
                ],
            },
        ]

        aligned = editor.align_turns_to_speech(script, transcript, duration=10.0)
        self.assertEqual(len(aligned.turns), 2)

        t1 = aligned.turns[0]
        # Exact word boundaries are 1.22 and 3.05, with 0.05s speech padding: ~1.17 to 3.10
        self.assertAlmostEqual(t1.start, 1.17, places=2)
        self.assertAlmostEqual(t1.end, 3.10, places=2)

        t2 = aligned.turns[1]
        # Exact word boundaries are 6.35 and 7.42, with 0.05s speech padding: ~6.30 to 7.47
        self.assertAlmostEqual(t2.start, 6.30, places=2)
        self.assertAlmostEqual(t2.end, 7.47, places=2)

    def test_checked_turns_eliminates_cascading_forward_drift(self):
        # Turn 1 and Turn 2 slightly overlap, while Turn 3 starts much later at 8.0s
        script = editor.Script(
            turns=[
                editor.Turn(
                    start=1.0,
                    end=3.2,
                    role="streamer",
                    english="First sentence here",
                    russian="Первая фраза тут",
                    emotion="neutral",
                ),
                editor.Turn(
                    start=3.1,
                    end=4.5,
                    role="streamer",
                    english="Second sentence follows",
                    russian="Вторая фраза следует",
                    emotion="neutral",
                ),
                editor.Turn(
                    start=8.0,
                    end=10.0,
                    role="guest",
                    english="Third sentence way later",
                    russian="Третья фраза намного позже",
                    emotion="neutral",
                ),
            ],
            cameras=[],
        )

        turns = editor.checked_turns(script, duration=12.0)
        self.assertEqual(len(turns), 3)

        # Turn 1 tail trimmed slightly, Turn 2 slightly shifted to avoid collision
        self.assertLessEqual(turns[0]["end"], 3.2)
        self.assertGreaterEqual(turns[1]["start"], turns[0]["end"] + 0.04)

        # Crucially: Turn 3 MUST NOT be shifted forward by a cascading cursor!
        self.assertEqual(turns[2]["start"], 8.0)
        self.assertEqual(turns[2]["end"], 10.0)

    def test_captions_proportional_fallback_chunking(self):
        turn = {
            "start": 2.0,
            "end": 6.5,
            "russian": "Посмотри как круто он только что запушил эту точку на карте",
            "emotion": "excited",
        }
        # heard is empty (ASR match 0% < 50%)
        heard = []
        speech_length = 4.0

        cues = editor.captions_for_turn(turn, heard, speech_length)
        # Should NOT be a single cue ("простыня")
        self.assertGreater(len(cues), 1)

        # Check chronological continuity
        prev_end = turn["start"]
        for cue in cues:
            self.assertGreaterEqual(cue["start"], prev_end - 0.01)
            self.assertGreater(cue["end"], cue["start"])
            words = cue["text"].split()
            self.assertGreaterEqual(len(words), 2)
            self.assertLessEqual(len(words), 6)
            prev_end = cue["end"]

        # Last cue ends at start + speech_length
        self.assertAlmostEqual(cues[-1]["end"], turn["start"] + speech_length, places=2)

    def test_ducking_volume_expression_syntax_and_ffmpeg_eval(self):
        intervals = [(1.5, 4.0), (4.15, 6.0), (8.5, 11.0)]
        expr = media.build_ducking_volume_expr(intervals, base_vol=0.78, duck_vol=0.16, fade=0.10)

        # Contains ducking levels
        self.assertIn("0.78", expr)
        self.assertIn("0.16", expr)
        self.assertIn("between(t,", expr)

        # Validate with FFmpeg evaluation
        cmd = [
            "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=1000:duration=12",
            "-af", f"volume=eval=frame:volume='{expr}'",
            "-f", "null", "-",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(res.returncode, 0, f"FFmpeg failed with: {res.stderr}")

    def test_dub_turn_atempo_up_to_132(self):
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            # Create a 2.5s 48kHz mono test tone representing speech
            raw_audio = work / "synth.wav"
            media.ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                         "-t", "2.5", "-c:a", "pcm_s16le", str(raw_audio))

            class MockFish:
                model = "mock-fish"
                def synthesize(self, voice_id, text, emotion):
                    return raw_audio.read_bytes()

            class MockModel:
                model = "mock-gemini"
                def shorten(self, turn, factor):
                    return "Короче"

            class MockSpeech:
                def read(self, path, language):
                    return []

            # Time slot is 2.0s. 2.5s / 2.0s = 1.25x (which is > 1.18x old limit, but <= 1.32x new limit)
            turn = {
                "start": 1.0,
                "end": 3.0,
                "russian": "Это довольно длинное предложение для озвучки",
                "english": "This is quite a long sentence for dubbing",
                "emotion": "neutral",
            }

            target, res_turn, cues = editor.dub_turn(
                turn,
                voice_id="0123456789abcdef0123456789abcdef",
                work=work,
                fish=MockFish(),
                model=MockModel(),
                speech=MockSpeech(),
            )

            self.assertTrue(target.is_file())
            dur = media.probe(target, require_video=False).duration
            # Resulting duration must fit inside the 2.0s limit (with allowed 0.015s float tolerance)
            self.assertLessEqual(dur, 2.0 + 0.015)
            self.assertGreater(len(cues), 0)


if __name__ == "__main__":
    unittest.main()
