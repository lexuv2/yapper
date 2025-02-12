from faster_whisper import WhisperModel
import time

import sounddevice as sd
import wavio
import pyaudio
import numpy 

from pynput.keyboard import Key, Listener
import os
from ctypes import cdll
# os.system("export LD_PRELOAD=./cudnn/lib/libcudnn.so.9")


## run this before export LD_PRELOAD=./cudnn/lib/libcudnn.so.9

# RATE=44100
RATE = 16000
RECORD_SECONDS = 5


# initialize portaudio
p = pyaudio.PyAudio()

# model_size = "distil-large-v3"
model_size = "medium"

model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# model = whisper.load_model("small.en")
while True:
    print("press to record")
    input()
    print("Speak now")
    recording = sd.rec(int(RATE * RECORD_SECONDS), samplerate=RATE, channels=1, dtype=numpy.float32)
    sd.wait()
    print("Recording done")
    # recording = recording.astype(numpy.int16).flatten().astype(numpy.float32) / 32768.0
    recording = recording.flatten()


    ## play the recording
    sd.play(recording, RATE)
    ## save the recording
    wavio.write("recording.wav", recording, RATE, sampwidth=2)

    print(numpy.min(recording))
    print(numpy.max(recording))
    start_time = time.time()
    print("Transcribing")
    segments, info = model.transcribe(recording, beam_size=5)

    
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    print("Time taken: ", time.time() - start_time)