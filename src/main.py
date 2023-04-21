from modules.stt import SpeechToText
from modules.tts import TextToSpeech


class Assistant:
    def __init__(self):
        self.speech_to_text = SpeechToText(model_path="models/vosk")
        self.text_to_speech = TextToSpeech(model_path="models/tts")

    def handle_cmd(self, cmd):
        print("cmd", cmd)
        self.text_to_speech.voice(cmd)

    def main(self):
        with self.speech_to_text as stt:
            stt.recognize(self.handle_cmd)


if __name__ == "__main__":
    assistant = Assistant()
    assistant.main()
