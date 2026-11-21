"""Comprehensive regression suite for long VOD scaling, semantic merging, ASR configuration, and dubbing sync."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from pipeline import transcriber
from pipeline import twitch_editor as editor
from pipeline import twitch_media as media


class LongVodScalingRegressionTests(unittest.TestCase):
    def test_long_vod_scaling_15h(self):
        # 15 hours = 54,000 seconds
        targets = editor.dynamic_vod_targets(54000)
        self.assertGreaterEqual(targets["target"], 20, "15h VOD target clips must not be capped at 4-5")
        self.assertLessEqual(targets["target"], 25)
        self.assertGreaterEqual(targets["max"], 25)
        self.assertGreaterEqual(targets["refine_limit"], 50, "Candidate pool for 15h VOD must be >= 50")
        self.assertGreaterEqual(targets["reserve_limit"], 35, "Reserve limit for 15h VOD must be >= 35")

    def test_checked_highlight_accepts_60_to_90s(self):
        # 75s clip with hook at 3.5s
        cut_75 = editor.HighlightEdit(
            publish=True,
            start_at=1.0,
            hook_at=4.5,   # hook offset = 3.5s (<= 5.0s)
            payoff_at=50.0,
            end_at=76.0,   # duration = 75.0s
            title="75s clutch play",
            opening="Setup",
            payoff="Victory",
            novelty=85,
            cameras=[]
        )
        seed = {"start": 100.0, "end": 180.0, "kind": "play"}
        highlight = editor.checked_highlight(cut_75, offset=100.0, length=90.0, seed=seed)
        self.assertIsNotNone(highlight)
        self.assertEqual(highlight["end"] - highlight["start"], 75.0)

        # 89s clip with hook at 4.8s
        cut_89 = editor.HighlightEdit(
            publish=True,
            start_at=1.0,
            hook_at=5.8,   # hook offset = 4.8s (<= 5.0s)
            payoff_at=60.0,
            end_at=90.0,   # duration = 89.0s
            title="89s extended highlight",
            opening="Setup",
            payoff="Climax",
            novelty=90,
            cameras=[]
        )
        highlight_89 = editor.checked_highlight(cut_89, offset=100.0, length=95.0, seed=seed)
        self.assertIsNotNone(highlight_89)
        self.assertEqual(highlight_89["end"] - highlight_89["start"], 89.0)

        # Reject clips exceeding 95s
        cut_105 = editor.HighlightEdit(
            publish=True,
            start_at=0.0,
            hook_at=2.0,
            payoff_at=60.0,
            end_at=105.0,  # 105s > 95s
            title="Too long",
            opening="Setup",
            payoff="Climax",
            novelty=90,
            cameras=[]
        )
        self.assertIsNone(editor.checked_highlight(cut_105, offset=100.0, length=120.0, seed=seed))

    def test_snap_speech_preserves_60_to_90s(self):
        event = {"start": 10.0, "end": 85.0, "peak": 50.0}
        transcript = [
            {"start": 9.5, "end": 13.0, "text": "Opening words", "words": [{"word": "Opening", "start": 9.5, "end": 10.2}]},
            {"start": 82.0, "end": 85.5, "text": "Ending reaction", "words": [{"word": "Ending", "start": 84.5, "end": 85.5}]},
        ]
        snapped = editor.snap_speech(event, transcript, offset=0.0, end=120.0)
        dur = snapped["end"] - snapped["start"]
        self.assertGreaterEqual(dur, 60.0)
        self.assertLessEqual(dur, 95.0)


class SemanticMergingRegressionTests(unittest.TestCase):
    def test_semantic_merging_preserves_hook_and_reaction_tail(self):
        ev1 = {
            "start": 100.0,
            "peak": 115.0,
            "end": 130.0,
            "title": "Клатч 1 в 4",
            "kind": "play",
            "reason": "Игрок остается один против четверых",
            "evidence": "Выбивает плент и ставит бомбу",
            "hook": 90,
            "payoff": 85,
            "standalone": 80,
            "energy": 85,
            "score": 85,
        }
        ev2 = {
            "start": 131.0,
            "peak": 145.0,
            "end": 165.0,
            "title": "Реакция стримера на клатч",
            "kind": "reaction",
            "reason": "Стример кричит от радости",
            "evidence": "Громкий смех и крики победы",
            "hook": 75,
            "payoff": 95,
            "standalone": 85,
            "energy": 95,
            "score": 90,
        }

        merged = editor.merge_semantic_events([ev1, ev2], max_duration=95.0)
        self.assertEqual(len(merged), 1)

        result = merged[0]
        # Preserves earliest start of setup/hook
        self.assertEqual(result["start"], 100.0)
        # Preserves latest tail of reaction
        self.assertEqual(result["end"], 165.0)
        # Total duration = 65.0s <= 95.0s
        self.assertEqual(result["end"] - result["start"], 65.0)
        # High energy peak and boosted synergy score retained
        self.assertEqual(result["peak"], 145.0)
        self.assertEqual(result["score"], 92.0)
        # Reasoning and evidence merged
        self.assertIn("Выбивает плент", result["evidence"])
        self.assertIn("Громкий смех", result["evidence"])

    def test_semantic_merging_respects_max_duration_ceiling(self):
        ev1 = {
            "start": 10.0,
            "peak": 30.0,
            "end": 65.0,
            "title": "Первая длинная сцена",
            "kind": "play",
            "reason": "Сцена 1",
            "evidence": "Действие 1",
            "hook": 80, "payoff": 80, "standalone": 80, "energy": 80, "score": 80,
        }
        ev2 = {
            "start": 66.0,
            "peak": 80.0,
            "end": 120.0,
            "title": "Вторая сцена",
            "kind": "reaction",
            "reason": "Сцена 2",
            "evidence": "Действие 2",
            "hook": 80, "payoff": 80, "standalone": 80, "energy": 80, "score": 80,
        }
        # 120.0 - 10.0 = 110.0s > 95.0s -> cannot merge
        merged = editor.merge_semantic_events([ev1, ev2], max_duration=95.0)
        self.assertEqual(len(merged), 2)


class AsrConfigurationRegressionTests(unittest.TestCase):
    def test_local_speech_config_and_env_propagation(self):
        with patch.dict(os.environ, {
            "TWITCH_WHISPER_MODEL": "medium",
            "TWITCH_WHISPER_DEVICE": "cpu",
            "TWITCH_WHISPER_COMPUTE": "int8",
        }):
            speech = editor.LocalSpeech()
            cfg = speech._resolve_transcriber_cfg()
            self.assertEqual(cfg["model_size"], "medium")
            self.assertEqual(cfg["device"], "cpu")
            self.assertEqual(cfg["compute_type"], "int8")

    def test_transcriber_propagates_vad_parameters_and_options(self):
        fake_model = MagicMock()
        fake_info = MagicMock(duration=20.0, language="ru")
        fake_model.transcribe.return_value = ([], fake_info)

        with patch.object(transcriber.Transcriber, "_init_model", return_value=(fake_model, None)):
            t = transcriber.Transcriber({
                "whisper": {
                    "device": "cpu",
                    "vad_parameters": {
                        "threshold": 0.28,
                        "min_speech_duration_ms": 100,
                        "min_silence_duration_ms": 250,
                        "speech_pad_ms": 350,
                    },
                    "condition_on_previous_text": True,
                    "no_speech_threshold": 0.45,
                }
            })
            # Test with VAD enabled
            t.transcribe("test.wav", vad_filter=True)
            fake_model.transcribe.assert_called_once()
            kwargs = fake_model.transcribe.call_args[1]
            self.assertTrue(kwargs["vad_filter"])
            self.assertEqual(kwargs["condition_on_previous_text"], True)
            self.assertEqual(kwargs["no_speech_threshold"], 0.45)
            self.assertEqual(kwargs["vad_parameters"]["threshold"], 0.28)
            self.assertEqual(kwargs["vad_parameters"]["min_speech_duration_ms"], 100)
            self.assertEqual(kwargs["vad_parameters"]["min_silence_duration_ms"], 250)
            self.assertEqual(kwargs["vad_parameters"]["speech_pad_ms"], 350)

            # Test with VAD disabled (for cut clips)
            fake_model.transcribe.reset_mock()
            t.transcribe("clip.wav", vad_filter=False)
            fake_model.transcribe.assert_called_once()
            kwargs_no_vad = fake_model.transcribe.call_args[1]
            self.assertFalse(kwargs_no_vad["vad_filter"])
            self.assertIsNone(kwargs_no_vad["vad_parameters"])


class DubbingSyncRegressionTests(unittest.TestCase):
    def test_checked_turns_eliminates_cascading_forward_drift_multi_turn(self):
        # 4 turns: Turn 1 and Turn 2 overlap; Turn 3 and 4 are independent later
        script = editor.Script(
            turns=[
                editor.Turn(start=0.0, end=2.2, role="streamer", english="One", russian="Раз", emotion="neutral"),
                editor.Turn(start=2.0, end=4.0, role="streamer", english="Two", russian="Два", emotion="neutral"),
                editor.Turn(start=6.0, end=8.0, role="guest", english="Three", russian="Три", emotion="neutral"),
                editor.Turn(start=10.0, end=12.0, role="guest", english="Four", russian="Четыре", emotion="neutral"),
            ],
            cameras=[],
        )

        turns = editor.checked_turns(script, duration=15.0)
        self.assertEqual(len(turns), 4)

        # Overlap between turn 0 and 1 is resolved locally
        self.assertLessEqual(turns[0]["end"], 2.2)
        self.assertGreaterEqual(turns[1]["start"], turns[0]["end"] + 0.04)

        # Turns 2 and 3 must strictly retain their original starts without cascading delay!
        self.assertEqual(turns[2]["start"], 6.0)
        self.assertEqual(turns[2]["end"], 8.0)
        self.assertEqual(turns[3]["start"], 10.0)
        self.assertEqual(turns[3]["end"], 12.0)

    def test_dub_turn_atempo_and_shorten_fallback(self):
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            audio_path = work / "synth.wav"
            # Generate 2.5s audio tone
            media.ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                         "-t", "2.5", "-c:a", "pcm_s16le", str(audio_path))

            class MockFish:
                model = "mock-fish"
                def synthesize(self, voice_id, text, emotion):
                    return audio_path.read_bytes()

            class MockModel:
                model = "mock-model"
                shorten_called = False
                def shorten(self, turn, factor):
                    self.shorten_called = True
                    return "Короткая фраза"

            class MockSpeech:
                def read(self, path, language):
                    return []

            # 1. Test 1.25x tempo (2.5s / 2.0s = 1.25x <= 1.32x limit) -> succeeds via atempo
            turn_atempo = {
                "start": 1.0,
                "end": 3.0,
                "russian": "Предложение укладывающееся с ускорением",
                "english": "A sentence that fits with acceleration",
                "emotion": "neutral",
            }
            model_inst = MockModel()
            target, res, cues = editor.dub_turn(
                turn_atempo,
                voice_id="11111111111111111111111111111111",
                work=work,
                fish=MockFish(),
                model=model_inst,
                speech=MockSpeech(),
            )
            self.assertTrue(target.is_file())
            dur = media.probe(target, require_video=False).duration
            self.assertLessEqual(dur, 2.0 + 0.015)
            self.assertFalse(model_inst.shorten_called)

    def test_captions_proportional_chunking_on_whisper_mismatch(self):
        turn = {
            "start": 1.0,
            "end": 8.0,
            "russian": "Это очень длинная реплика которую модель должна разбить на несколько коротких читаемых титров",
            "emotion": "neutral",
        }
        # ASR returned 0 matching words
        cues = editor.captions_for_turn(turn, heard=[], speech_length=6.0)

        # Must NOT collapse into 1 giant static subtitle
        self.assertGreater(len(cues), 1)

        total_words = sum(len(c["text"].split()) for c in cues)
        self.assertEqual(total_words, len(turn["russian"].split()))

        # Check chronological ordering and time boundaries
        for i, cue in enumerate(cues):
            self.assertGreaterEqual(cue["start"], 1.0)
            self.assertLessEqual(cue["end"], 7.0 + 0.01)
            self.assertGreater(cue["end"], cue["start"])
            if i > 0:
                self.assertGreaterEqual(cue["start"], cues[i - 1]["start"])


if __name__ == "__main__":
    unittest.main()
