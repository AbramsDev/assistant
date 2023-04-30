from rasa.core.agent import Agent
from rasa.core.utils import read_endpoint_config
from modules.stt import SpeechToText
from modules.tts import TextToSpeech


import asyncio
import uuid


class Assistant:
    def __init__(self):
        endpoints = read_endpoint_config("endpoints.yml", "action_endpoint")

        self.agent = Agent.load(
            model_path="models/rasa/20230430-125833-deterministic-touch.tar.gz",
            action_endpoint=endpoints
        )

        self.speech_to_text = SpeechToText(model_path="models/vosk")
        self.text_to_speech = TextToSpeech(model_path="models/tts")

        self.sender_id = str(uuid.uuid4())

    async def handle_command(self, cmd):
        responses = await self.agent.handle_text(text_message=cmd, sender_id=self.sender_id)
        for response in responses:
            self.text_to_speech.voice(response["text"])

    async def main(self):
        with self.speech_to_text as stt:
            await stt.recognize(self.handle_command)


if __name__ == "__main__":
    assistant = Assistant()
    asyncio.run(assistant.main())
