import unittest
from unittest.mock import patch, MagicMock
from voice.text_to_speech import TextToSpeech
from voice.speech_to_text import SpeechToText


class TestVoice(unittest.TestCase):
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

    def test_stt_fallback_when_mic_missing(self):
        stt = SpeechToText()
        stt.recognizer = None  # Force mic unavailable
        self.assertFalse(stt.is_voice_input_available())

        with patch("builtins.input", return_value="hello jarvis"):
            result = stt.listen_and_transcribe()
            self.assertEqual(result, "hello jarvis")


if __name__ == "__main__":
    unittest.main()
