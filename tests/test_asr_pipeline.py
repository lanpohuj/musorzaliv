"""Tests for ASR pipeline: Whisper model auto-selection, VAD tuning, and audio preprocessing."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from pipeline import transcriber
from pipeline import twitch_editor as editor
from pipeline import twitch_media as media
from pipeline.twitch_workflow import Jobs


class AsrPreprocessingTests(unittest.TestCase):
    def test_extract_audio_includes_highpass_and_dynaudnorm(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            transcriber.extract_audio("input.mp4", "output.wav")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            self.assertIn("-af", cmd)
            af_idx = cmd.index("-af")
            self.assertEqual(cmd[af_idx + 1], "highpass=f=85,dynaudnorm")
            self.assertIn("-acodec", cmd)
            self.assertIn("16000", cmd)

    def test_media_wav_includes_highpass_and_dynaudnorm(self):
        with patch.object(media, "ffmpeg") as mock_ffmpeg:
            media.wav(Path("source.mp4"), Path("target.wav"), start=5.0, end=10.0)
            mock_ffmpeg.assert_called_once()
            args = list(mock_ffmpeg.call_args[0])
            self.assertIn("-af", args)
            af_idx = args.index("-af")
            self.assertEqual(args[af_idx + 1], "highpass=f=85,dynaudnorm")
            self.assertIn("-ar", args)
            self.assertIn("16000", args)


class TranscriberParameterTests(unittest.TestCase):
    def test_vad_parameters_and_defaults(self):
        fake_model = MagicMock()
        fake_info = MagicMock(duration=10.0, language="ru")
        fake_model.transcribe.return_value = ([], fake_info)

        with patch.object(transcriber.Transcriber, "_init_model", return_value=(fake_model, None)):
            t = transcriber.Transcriber({"whisper": {"device": "cpu"}})
            t.transcribe("sample.wav")

            fake_model.transcribe.assert_called_once()
            kwargs = fake_model.transcribe.call_args[1]
            self.assertTrue(kwargs["vad_filter"])
            self.assertEqual(kwargs["condition_on_previous_text"], True)
            self.assertEqual(kwargs["no_speech_threshold"], 0.45)

            vad_params = kwargs["vad_parameters"]
            self.assertEqual(vad_params["threshold"], 0.28)
            self.assertEqual(vad_params["min_speech_duration_ms"], 100)
            self.assertEqual(vad_params["min_silence_duration_ms"], 250)
            self.assertEqual(vad_params["speech_pad_ms"], 350)

    def test_vad_filter_false_disables_vad(self):
        fake_model = MagicMock()
        fake_info = MagicMock(duration=5.0, language="en")
        fake_model.transcribe.return_value = ([], fake_info)

        with patch.object(transcriber.Transcriber, "_init_model", return_value=(fake_model, None)):
            t = transcriber.Transcriber({"whisper": {"device": "cpu"}})
            t.transcribe("clip.wav", vad_filter=False)

            fake_model.transcribe.assert_called_once()
            kwargs = fake_model.transcribe.call_args[1]
            self.assertFalse(kwargs["vad_filter"])
            self.assertIsNone(kwargs["vad_parameters"])

    def test_cuda_failure_falls_back_to_cpu_small(self):
        def fake_whisper_model(size, device="cpu", compute_type="int8"):
            if device == "cuda":
                raise RuntimeError("CUDA out of memory or not supported")
            mock = MagicMock()
            mock.device = "cpu"
            return mock

        with patch("faster_whisper.WhisperModel", side_effect=fake_whisper_model):
            t = transcriber.Transcriber({"whisper": {"device": "cuda", "model_size": "large-v3-turbo"}})
            self.assertEqual(t.model_size, "small")
            self.assertEqual(t.compute_type, "int8")


class LocalSpeechDeviceSelectionTests(unittest.TestCase):
    def test_gpu_defaults_to_large_v3_turbo_and_float16(self):
        with patch.object(editor, "_has_cuda", return_value=True), \
             patch.dict("os.environ", {}, clear=True):
            speech = editor.LocalSpeech()
            cfg = speech._resolve_transcriber_cfg()
            self.assertEqual(cfg["device"], "cuda")
            self.assertEqual(cfg["compute_type"], "float16")
            self.assertEqual(cfg["model_size"], "large-v3-turbo")

    def test_cpu_defaults_to_small_and_int8(self):
        with patch.object(editor, "_has_cuda", return_value=False), \
             patch.dict("os.environ", {}, clear=True):
            speech = editor.LocalSpeech()
            cfg = speech._resolve_transcriber_cfg()
            self.assertEqual(cfg["device"], "cpu")
            self.assertEqual(cfg["compute_type"], "int8")
            self.assertEqual(cfg["model_size"], "small")

    def test_env_model_override_respected(self):
        with patch.object(editor, "_has_cuda", return_value=True), \
             patch.dict("os.environ", {"TWITCH_WHISPER_MODEL": "medium"}):
            speech = editor.LocalSpeech()
            cfg = speech._resolve_transcriber_cfg()
            self.assertEqual(cfg["device"], "cuda")
            self.assertEqual(cfg["model_size"], "medium")

    def test_local_speech_passes_vad_filter_false(self):
        fake_transcriber = MagicMock()
        fake_transcriber.transcribe.return_value = ([], 0.0, "en")

        speech = editor.LocalSpeech()
        speech.transcriber = fake_transcriber
        speech.read(Path("dummy.wav"), "en", vad_filter=False)
        fake_transcriber.transcribe.assert_called_once_with(str(Path("dummy.wav")), vad_filter=False)


class WorkflowScriptClipVadFilterTests(unittest.TestCase):
    def test_script_clip_disables_vad(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = Jobs(root / "jobs", root / "exports")
            job = jobs.create("", "Тест")
            job.update(source="fixture.mp4", source_info={"duration": 60, "audio": True})

            clip = {"id": "c1", "start": 10.0, "peak": 18.0, "end": 25.0, "title": "Момент", "selected": True}
            fake_model = MagicMock()
            fake_turn = editor.Turn(start=1.0, end=2.0, role="streamer", english="Hello", russian="Привет", emotion="neutral")
            fake_model.adapt.return_value = (editor.Script(turns=[fake_turn], cameras=[]), {"input": 10, "output": 10})

            fake_speech = MagicMock()
            fake_speech.from_video.return_value = [{"start": 1.0, "end": 2.5, "text": "Привет"}]

            def fake_window(src, lo, hi, dst, **kw):
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.touch()

            with patch.object(media, "cloud_window", side_effect=fake_window):
                job._script_clip(clip, fake_model, fake_speech, has_audio=True)

            fake_speech.from_video.assert_called_once()
            _, kwargs = fake_speech.from_video.call_args
            self.assertEqual(kwargs.get("vad_filter"), False)


if __name__ == "__main__":
    unittest.main()
