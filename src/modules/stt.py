# from tts import TextToSpeech

# import sounddevice as sd
# import vosk

# import json
# import queue


# class SpeechToText:
#     def __init__(self):
#         print("SpeechToText: initialization...")

#         self.q = queue.Queue()

#         self.vosk_model = vosk.Model(model_path='models/vosk')
#         self.tts_model = TextToSpeech(model_path="models/tts")

#         self.device = sd.default.device[0]
#         self.samplerate = int(sd.query_devices(self.device, 'input')['default_samplerate'])

#     def callback(self, indata, frames, time, status):
#         '''
#         Добавляет в очередь семплы из потока.
#         вызывается каждый раз при наполнении blocksize
#         в sd.RawInputStream
#         '''

#         self.q.put(bytes(indata))

#     def recognize(self, data):
#         print("SpeechToText: recognizing...")

#         '''
#         Анализ распознанной речи
#         '''

#         # проверяем есть ли имя бота в data, если нет, то return
#         trg = {"пятница"}.intersection(data.split())
#         if not trg:
#             return

#         text = data.replace(list(trg)[0], '').strip()
#         return text

#     def __enter__(self):
#         # постоянная прослушка микрофона
#         with sd.RawInputStream(
#             samplerate=self.samplerate,
#             blocksize=16000,
#             device=self.device,
#             dtype='int16',
#             channels=1,
#             callback=self.callback
#         ):
#             rec = vosk.KaldiRecognizer(self.vosk_model, self.samplerate)

#             while True:
#                 data = self.q.get()
#                 if rec.AcceptWaveform(data):
#                     data = json.loads(rec.Result())['text']
#                     result = self.recognize(data)
#                     print("result", result)
#                     return result
#                 # else:
#                 #     print(rec.PartialResult())

#     def __exit__(self, exc_type, exc_value, traceback):
#         print("Exiting the block")


# if __name__ == "__main__":
#     with SpeechToText() as text:
#         print(text)

# import os
# import json
# import vosk
# import pyaudio


# class MicrophoneSpeechToText:
#     def __init__(self, model_path, sample_rate=16000):
#         self.sample_rate = sample_rate
#         self.model = vosk.Model(model_path)
#         self.recognizer = vosk.KaldiRecognizer(self.model, sample_rate)
#         self.stream = pyaudio.PyAudio().open(
#             format=pyaudio.paInt16,
#             channels=1,
#             rate=self.sample_rate,
#             input=True,
#             frames_per_buffer=16000,
#         )

#     def transcribe(self):
#         while True:
#             data = self.stream.read(16000)
#             if len(data) == 0:
#                 break
#             if self.recognizer.AcceptWaveform(data):
#                 result = json.loads(self.recognizer.Result())
#                 print(result['text'])

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.stream.stop_stream()
#         self.stream.close()
#         pyaudio.PyAudio().terminate()

import json
import vosk
# import pvporcupine
import pvrecorder
import struct


class SpeechToText:
    def __init__(self, model_path=None, sample_rate=16000):
        if model_path is None:
            raise ValueError("model_path is not specified")

        self.sample_rate = sample_rate

        model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(model, sample_rate)

        # self.keyword_detector = pvporcupine.create(
        #     access_key="NjvbZkSQ/MQ8WGFNXi71elnRsI1KaaaN2jUa5AfCWy4enks4KfC0rQ==",
        #     keywords=['porcupine'],
        #     sensitivities=[1]
        # )

        self.recorder = pvrecorder.PvRecorder(
            device_index=0,
            frame_length=512  # self.keyword_detector.frame_length
        )

    async def recognize(self, callback):
        print("Говорите...")
        try:
            while True:
                pcm = self.recorder.read()
                sp = struct.pack("h" * len(pcm), *pcm)
                if self.recognizer.AcceptWaveform(sp):
                    result = json.loads(self.recognizer.Result())

                    text = result['text'].lower()
                    trg = {"сара"}.intersection(text.split())
                    if not trg:
                        continue

                    text = text.replace(list(trg)[0], '').strip()
                    if text == "пока":
                        break

                    self.recorder.stop()
                    await callback(text)
                    self.recorder.start()
        except Exception as _ex:
            print(_ex)
            self.recorder.stop()

    def __enter__(self):
        self.recorder.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.recorder.stop()
        # self.keyword_detector.delete()


if __name__ == "__main__":
    with SpeechToText("models/vosk") as stt:
        stt.recognize(lambda x: print("text:", x))
