import sys
from typing import Optional
from voice.speech_to_text import SpeechToText
from voice.text_to_speech import TextToSpeech
from agent.agent_core import AgentCore
from observability.logging import get_logger

logger = get_logger("CyberJarvis.VoiceLoop")

EXIT_PHRASES = {"exit", "quit", "goodbye", "bye", "stop", "terminate"}


class VoiceAssistantLoop:
    """Conversational voice interface loop coordinating STT, AgentCore, and TTS."""

    def __init__(self, agent: Optional[AgentCore] = None):
        self.stt = SpeechToText()
        self.tts = TextToSpeech()

        def on_agent_status(status_msg: str):
            print(status_msg)

        self.agent = agent or AgentCore(on_status_change=on_agent_status)

    def run(self):
        print("\n" + "=" * 55)
        print("🤖 CyberJarvis Voice Assistant Mode Active")
        print("🎤 Say 'CyberJarvis, diagnose my computer' or 'exit' to quit")
        print("=" * 55 + "\n")

        self.tts.speak("CyberJarvis is online and listening.", blocking=False)

        while True:
            try:
                user_command = self.stt.listen_and_transcribe()
                if not user_command:
                    continue

                print(f"\n👤 You (Voice): {user_command}")

                clean_cmd = user_command.lower().strip()
                if any(exit_word in clean_cmd for exit_word in EXIT_PHRASES):
                    farewell = "Goodbye! Shutting down CyberJarvis."
                    print(f"\n🤖 CyberJarvis: {farewell}")
                    self.tts.speak(farewell, blocking=True)
                    break

                # Process via Agent Core
                response = self.agent.process(user_command)

                # Print formatted detailed report
                print(f"\n🤖 CyberJarvis:\n{response.answer}\n")

                # Speak natural summary
                spoken_text = response.spoken_summary or response.answer[:200]
                self.tts.speak(spoken_text, blocking=True)

            except (KeyboardInterrupt, EOFError):
                print("\n\nShutting down CyberJarvis voice loop...")
                break
            except Exception as ex:
                logger.error(f"Voice loop error: {ex}", exc_info=True)
                print(f"\n❌ Error in voice processing: {ex}\n")
