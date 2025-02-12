import whisper
import time
import sounddevice as sd
import numpy
import torch
from pynput import keyboard
import pyperclip

# Constants
RATE = 16000

# Initialize Whisper model
model = whisper.load_model("small.en")

# Global variables
recording = numpy.array([])
is_recording = False

def on_press(key):
    global is_recording, recording
    if key == keyboard.Key.space and not is_recording:
        is_recording = True
        print("Recording started...")
        recording = numpy.array([])
        sd.InputStream(samplerate=RATE, channels=1, dtype=numpy.float32, callback=callback).start()

def on_release(key):
    global is_recording
    if key == keyboard.Key.space and is_recording:
        is_recording = False
        sd.stop()
        print("Recording stopped.")
        transcribe_and_copy(recording)
        return False  # Stop listener

def callback(indata, frames, time, status):
    global recording
    # recording.append(indata.copy())
    """
    The first and second argument are the input and output buffer, respectively, as two-dimensional numpy.ndarray with one column per channel (i.e. with a shape of (frames, channels)) 
    """    
    #https://python-sounddevice.readthedocs.io/en/0.5.1/api/streams.html#sounddevice.Stream
    for frame in indata:
        recording = numpy.append(recording, frame)

def transcribe_and_copy(recording):
    print("Transcribing...")
    start_time = time.time()
    ## convert to numpy ndarray

    recording = recording.flatten()
    sd.play(recording, RATE)
    result = model.transcribe(recording)
    print("Time taken: ", time.time() - start_time)
    transcribed_text = result["text"]
    print(transcribed_text)
    pyperclip.copy(transcribed_text)
    print("Transcribed text copied to clipboard.")

# Start listening for key presses
print("Started")
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
