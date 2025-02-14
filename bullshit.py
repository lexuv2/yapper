# import whisper
from faster_whisper import WhisperModel
import time
import sounddevice as sd
import numpy
import torch
from pynput import keyboard
import pyperclip
import threading
import simpleaudio as sa
from ollama import chat
from ollama import ChatResponse
from TTS.api import TTS


print(TTS().list_models())
# Constants
RATE = 16000
## run this before export LD_PRELOAD=./.venv/lib/python3.9/site-packages/nvidia/cudnn/lib/libcudnn_cnn.so.9

# Initialize Whisper model
# model_size = "medium"
model_size = "turbo"
# model_size = "distil-large-v3"
notification_file_name = "notification.wav"
wave = sa.WaveObject.from_wave_file(notification_file_name)

# model = whisper.load_model("small.en")
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

# Global variables
recording = numpy.array([],dtype=numpy.float32)
is_recording = False
pass_to_chat = False
global stream 



def on_press(key):
    global is_recording, recording, stream, pass_to_chat
    if  key == keyboard.Key.ctrl_r and not is_recording:
        start_listening()
    if key == keyboard.KeyCode.from_char('`') and not is_recording:
        pass_to_chat = True
        print("Chat mode activated.")
        x = threading.Thread(target=load_tts_model)
        x.start()
        start_listening()
        
def start_listening():
    global is_recording, recording, stream
    is_recording = True
    wave.play()
    print("Recording started...")
    recording = numpy.array([],dtype=numpy.float32)
    stream = sd.InputStream(samplerate=RATE, channels=1, dtype=numpy.float32, callback=callback)
    stream.start()

def on_release(key):
    global is_recording,recording, stream
    if (key == keyboard.KeyCode.from_char('`') or  key == keyboard.Key.ctrl_r) and is_recording:
        
        stream.stop()
        is_recording = False
        print("Recording stopped.")
        # transcribe_and_copy()
        x = threading.Thread(target=transcribe_and_copy, args=(recording.copy(), pass_to_chat))
        recording = numpy.array([],dtype=numpy.float32)
        x.start()
        x.join()                                    
        
        # return False  # Stop listener

def callback(indata, frames, time, status):
    global recording
    # recording.append(indata.copy())
    #https://python-sounddevice.readthedocs.io/en/0.5.1/api/streams.html#sounddevice.Stream
    # for frame in indata:
        # recording = numpy.append(recording, frame)
    recording = numpy.append(recording,indata)

tts = None
tts_ready = False
def load_tts_model():   
    global tts, tts_ready
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    tts_ready = True
    

def transcribe_and_copy(recording, pass_to_chat):
    global tts, tts_ready

    print("Transcribing...")                         
    start_time = time.time()
    ## convert to numpy ndarray
    recording = recording.flatten()                 
    print(numpy.info(recording))                     
    # sd.play(recording, RATE)
    # 
    # sd.wait()
    print(recording)
    print("dupatest")
    segments, info = model.transcribe(recording, beam_size=5)
    
    transcribed_text = ""
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        transcribed_text += segment.text
    pyperclip.copy(transcribed_text)
    wave.play()
    print("Time taken: ", time.time() - start_time)
    print("Transcribed text copied to clipboard.")
    recording = numpy.array([],dtype=numpy.float32)
    if pass_to_chat:
        print("Passing to chat...")
        stream = chat(
            model='deepseek-r1:1.5b',
            messages=[{'role': 'user', 'content': transcribed_text}],
            stream=True,
        )
        full_out = ""
        thinking = False
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
            ## ignore everything after </think>
            if "<think>" in chunk['message']['content']:
                thinking = True
            if "</think>" in chunk['message']['content']:
                thinking = False
            if not thinking:
                full_out += chunk['message']['content']

        print(("Starting tts"))
        while not tts_ready:
            time.sleep(0.05)
        tts.tts_to_file(text = full_out, speaker_wav="Electrochemistry_short.wav", language="en",file_path="output.wav")
        to_play = sa.WaveObject.from_wave_file("output.wav")
        to_play.play()
        del tts
        tts = None
        tts_ready = False


# Start listening for key presses
print("Started")
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()


                                                                                                                                                                  