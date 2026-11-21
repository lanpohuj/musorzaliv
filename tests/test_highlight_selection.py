"""Tests for V2 highlight selection: dynamic scaling, context retention, merging, and multi-signal scoring."""

import math
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from pipeline import twitch_editor as editor
from pipeline import twitch_river as river


def sample_event(start, peak, end, title="Клатч", kind="play", score_val=80, **kw):
    return {
        "start": start,
        "peak": peak,
        "end": end,
        "title": title,
        "kind": kind,
        "hook": 85,
        "payoff": 90,
        "standalone": 80,
        "energy": 85,
        "score": score_val,
        **kw,
    }


class DynamicScalingTests(unittest.TestCase):
    def test_dynamic_targets_scale_with_stream_duration(self):
        short = editor.dynamic_vod_targets(3600)  # 1 hour
        self.assertEqual(short["target"], 4)
        self.assertEqual(short["max"], 5)
        self.assertEqual(short["refine_limit"], 16)
        self.assertEqual(short["reserve_limit"], 12)

        medium = editor.dynamic_vod_targets(18000)  # 5 hours
        self.assertGreaterEqual(medium["target"], 8)
        self.assertGreater(medium["refine_limit"], short["refine_limit"])
        self.assertGreater(medium["reserve_limit"], short["reserve_limit"])

        long_vod = editor.dynamic_vod_targets(54000)  # 15 hours
        self.assertGreaterEqual(long_vod["target"], 20)
        self.assertLessEqual(long_vod["target"], 25)
        self.assertGreaterEqual(long_vod["max"], 25)
        self.assertGreaterEqual(long_vod["refine_limit"], 50)
        self.assertGreaterEqual(long_vod["reserve_limit"], 35)


class SemanticEventMergingTests(unittest.TestCase):
    def test_merges_play_and_reaction_within_max_duration(self):
        play = sample_event(10.0, 18.0, 22.0, title="Клатч 1 в 3", kind="play", score_val=82,
                            reason="Игрок выходит на плент", evidence="Убивает троих")
        reaction = sample_event(23.5, 27.0, 32.0, title="Реакция на клатч", kind="reaction", score_val=78,
                                reason="Стример кричит", evidence="Крик радости")
        merged = editor.merge_semantic_events([play, reaction], max_duration=95.0)

        self.assertEqual(len(merged), 1)
        res = merged[0]
        self.assertEqual(res["start"], 10.0)
        self.assertEqual(res["end"], 32.0)
        self.assertGreaterEqual(res["score"], 82)
        self.assertIn("Убивает троих", res["evidence"])

    def test_does_not_merge_unrelated_or_distant_events(self):
        first = sample_event(10.0, 20.0, 30.0, title="Эпизод А")
        distant = sample_event(70.0, 80.0, 90.0, title="Эпизод Б")
        result = editor.merge_semantic_events([first, distant], max_duration=95.0)
        self.assertEqual(len(result), 2)


class AudioPeaksZScoreTests(unittest.TestCase):
    def test_detects_spikes_and_sustained_energy(self):
        # 8000 Hz, 16-bit mono PCM. 2000 samples = 0.25s block.
        # Create 40 blocks = 10 seconds of audio.
        sr = 8000
        duration_s = 10.0
        total_samples = int(sr * duration_s)
        t = np.linspace(0, duration_s, total_samples, endpoint=False)

        # Baseline noise (quiet, amplitude ~ 300)
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(total_samples) * 300).astype(np.int16)

        # Sharp spike at 3.0s - 3.25s (amplitude 12000)
        spike_start = int(3.0 * sr)
        spike_end = int(3.3 * sr)
        audio[spike_start:spike_end] += (np.sin(2 * np.pi * 500 * t[spike_start:spike_end]) * 12000).astype(np.int16)

        # Sustained laughter / high energy at 6.0s - 8.0s (amplitude 6000)
        sustained_start = int(6.0 * sr)
        sustained_end = int(8.0 * sr)
        audio[sustained_start:sustained_end] += (np.sin(2 * np.pi * 400 * t[sustained_start:sustained_end]) * 6000).astype(np.int16)

        raw_bytes = audio.tobytes()

        with patch.object(river, "_decode", return_value=raw_bytes):
            peaks = river.audio_peaks(Path("dummy.mp4"))
            self.assertTrue(len(peaks) > 0)
            # Check there is a peak near 3.0s
            has_spike_peak = any(2.75 <= p <= 3.5 for p in peaks)
            self.assertTrue(has_spike_peak, f"Expected peak near 3.0s, got {peaks}")
            # Check there is a peak in sustained region 6.0s - 8.0s
            has_sustained_peak = any(6.0 <= p <= 8.0 for p in peaks)
            self.assertTrue(has_sustained_peak, f"Expected sustained peak near 6-8s, got {peaks}")


class MultiSignalScoringTests(unittest.TestCase):
    def test_compute_speech_signals_detects_wpm_and_excitement(self):
        # Fast speech with exclamations
        transcript = [
            {"start": 0.0, "end": 2.0, "text": "ДА ЛАДНО! НЕВЕРОЯТНО!", "words": [
                {"word": "ДА", "start": 0.0, "end": 0.3},
                {"word": "ЛАДНО!", "start": 0.4, "end": 0.8},
                {"word": "НЕВЕРОЯТНО!", "start": 0.9, "end": 1.9},
            ]},
            {"start": 2.0, "end": 4.0, "text": "КАК ОН ЭТО СДЕЛАЛ?! ЖЕСТЬ!", "words": [
                {"word": "КАК", "start": 2.0, "end": 2.3},
                {"word": "ОН", "start": 2.4, "end": 2.6},
                {"word": "ЭТО", "start": 2.7, "end": 2.9},
                {"word": "СДЕЛАЛ?!", "start": 3.0, "end": 3.4},
                {"word": "ЖЕСТЬ!", "start": 3.5, "end": 3.9},
            ]},
        ]
        signals = editor.compute_speech_signals(transcript, 0.0, 4.0)
        self.assertGreater(signals["wpm"], 100)
        self.assertGreater(signals["emotion_score"], 40)
        self.assertGreater(signals["speech_signal"], 20)

    def test_multi_signal_score_balances_novelty_and_base(self):
        ev = sample_event(0, 5, 10, score_val=80)
        # Novelty 80, base 80 -> exactly 80
        score_equal = editor.multi_signal_score(ev, novelty=80)
        self.assertEqual(score_equal, 80.0)

        # High audience views adds boost
        score_boosted = editor.multi_signal_score(ev, novelty=80, audience_views=15000, speech_signal=70.0)
        self.assertGreater(score_boosted, score_equal)
        self.assertLessEqual(score_boosted, 100.0)


class LongClipConstraintTests(unittest.TestCase):
    def test_checked_highlight_accepts_75s_clip_and_hook_up_to_5s(self):
        cut = editor.HighlightEdit(
            publish=True,
            start_at=2.0,
            hook_at=6.0,   # hook offset = 4.0s (<= 5.0s)
            payoff_at=50.0,
            end_at=77.0,   # duration = 75.0s (<= 95s)
            title="Эпический раунд",
            opening="Игрок берет снайперку",
            payoff="Выигрывает раунд",
            novelty=88,
            cameras=[]
        )
        seed = {"start": 100.0, "end": 180.0, "kind": "play"}
        highlight = editor.checked_highlight(cut, offset=100.0, length=90.0, seed=seed)
        self.assertIsNotNone(highlight)
        self.assertEqual(highlight["end"] - highlight["start"], 75.0)

    def test_snap_speech_preserves_clip_longer_than_old_49s_limit(self):
        event = {"start": 10.0, "end": 80.0, "peak": 50.0}
        transcript = [
            {"start": 9.8, "end": 12.0, "text": "Начало", "words": [{"word": "Начало", "start": 9.8, "end": 10.5}]},
            {"start": 78.0, "end": 80.2, "text": "Конец", "words": [{"word": "Конец", "start": 79.5, "end": 80.2}]},
        ]
        snapped = editor.snap_speech(event, transcript, offset=0.0, end=100.0)
        self.assertGreater(snapped["end"] - snapped["start"], 60.0)
        self.assertLessEqual(snapped["end"] - snapped["start"], 95.0)


if __name__ == "__main__":
    unittest.main()
