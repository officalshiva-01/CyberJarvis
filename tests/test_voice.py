import unittest
from unittest.mock import patch, MagicMock
from voice.text_to_speech import TextToSpeech, clean_for_speech
from voice.speech_to_text import SpeechToText
from voice.audio_capture import AudioCapture, AudioDeviceManager, calculate_pcm_rms


class TestVoice(unittest.TestCase):
    def test_calculate_pcm_rms(self):
        # All zeros -> 0 RMS
        zeros = b"\x00\x00" * 100
        self.assertEqual(calculate_pcm_rms(zeros), 0)

        # Empty -> 0 RMS
        self.assertEqual(calculate_pcm_rms(b""), 0)

    def test_clean_for_speech(self):
        # Ensure markdown formatting and headers are removed
        md_text = "### Diagnostic Status: **HEALTHY**\n- CPU: 12.5%\n- RAM: 30%"
        cleaned = clean_for_speech(md_text)
        self.assertNotIn("###", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertIn("Diagnostic Status", cleaned)

        # Ensure shell scripts are stripped
        code_text = "```bash\n#!/bin/bash\ncat /proc/meminfo\n```\nHere is your memory info."
        cleaned_code = clean_for_speech(code_text)
        self.assertNotIn("#!/bin/bash", cleaned_code)
        self.assertNotIn("cat /proc", cleaned_code)
        self.assertIn("Here is your memory info", cleaned_code)

    def test_audio_device_manager(self):
        devices = AudioDeviceManager.get_input_devices()
        self.assertIsInstance(devices, list)
        self.assertTrue(len(devices) > 0)
        default_dev = AudioDeviceManager.get_default_input_device()
        self.assertIsNotNone(default_dev)

    def test_tts_initialization(self):
        tts = TextToSpeech(engine="spd-say")
        self.assertIsNotNone(tts)

    @patch("subprocess.run")
    def test_tts_speak_spd_say(self, mock_run):
        tts = TextToSpeech(engine="spd-say")
        tts.spd_say_available = True
        tts.speak("Testing CyberJarvis audio", blocking=True)
        self.assertTrue(mock_run.called)
        args, kwargs = mock_run.call_args
        self.assertIn("spd-say", args[0])

    def test_stt_hardware_detection(self):
        stt = SpeechToText()
        self.assertTrue(stt.is_voice_input_available())
        info = stt.get_microphone_info()
        self.assertIn("Microphone:", info)

    def test_stt_fallback_when_mic_missing(self):
        mock_capture = MagicMock(spec=AudioCapture)
        mock_capture.is_available.return_value = False
        stt = SpeechToText(audio_capture=mock_capture)
        self.assertFalse(stt.is_voice_input_available())

        with patch("builtins.input", return_value="diagnose computer"):
            result = stt.listen_and_transcribe()
            self.assertEqual(result, "diagnose computer")


if __name__ == "__main__":
    unittest.main()
