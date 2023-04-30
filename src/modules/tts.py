
import os
import torch
import numpy as np
import pyaudio
import hashlib
import soundfile as sf


class TextToSpeech:
    def __init__(self, model_path=None, sample_rate=24000, speaker="xenia"):
        if model_path is None:
            raise ValueError("model_path is not specified")

        self.model_path = f"{model_path}/model.pt"
        self.sample_rate = sample_rate
        self.speaker = speaker
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = "tts_cache"

        torch.set_num_threads(6)

        if not os.path.isfile(self.model_path):
            if not os.path.exists(model_path):
                os.makedirs(os.path.dirname(self.model_path))

            torch.hub.download_url_to_file(
                url='https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                dst=self.model_path
            )

        # self.model = torch.package.PackageImporter(self.model_path).load_pickle(package="tts_models", resource="model")
        # self.model.to(torch.device(self.device))

        self.pa = pyaudio.PyAudio()

    def _cache_load(self, filename):
        cache_path = os.path.join(self.cache_dir, f"{filename}.wav")
        if os.path.exists(cache_path):
            print("loaded from cache")
            audio = sf.read(cache_path)
            return np.array(audio[0])

        return None

    def _cache_save(self, filename, audio):
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

        cache_path = os.path.join(self.cache_dir, f"{filename}.wav")
        sf.write(cache_path, audio, self.sample_rate)

    # TODO при каждом озвучивании модель сжирает память
    def _synthesize(self, text):
        print("no cache: synthesize start")
        model = torch.package.PackageImporter(self.model_path).load_pickle("tts_models", "model")
        model.to(torch.device(self.device))

        with torch.no_grad():
            audio = model.apply_tts(
                text=text,
                speaker=self.speaker,
                sample_rate=self.sample_rate,
                put_accent=True,
                put_yo=True
            )

        audio_array = np.array(audio)
        print("synthesize end")

        return audio_array

    def _play_audio(self, audio):
        stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True
        )

        stream.write(audio.astype(np.float32).tobytes())

        stream.stop_stream()
        stream.close()

    def voice(self, text):
        print("voice", text)

        if not text.strip():
            return

        if len(text) <= 100:
            audio_file = hashlib.md5(text.encode("utf-8")).hexdigest()
            audio = self._cache_load(filename=audio_file)

            if audio is None:
                audio = self._synthesize(text=text)
                self._cache_save(filename=audio_file, audio=audio)
        else:
            audio = self._synthesize(text=text)

        self._play_audio(audio=audio)

    def __del__(self):
        self.pa.terminate()


if __name__ == "__main__":
    tts = TextToSpeech(model_path="models/tts")
    tts.voice("Один!")
    tts.voice("Два!")
    tts.voice("Три!")
    tts.voice("Четыре!")
    tts.voice("Пять!")
