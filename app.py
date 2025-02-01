import tkinter as tk
import pyaudio
import numpy as np
import tensorflow as tf
import threading
from PIL import Image, ImageTk

# ---------------------------
# Load model
# ---------------------------
model = tf.keras.models.load_model("speech_model.keras")

label_names = np.array([
    '_background_noise_', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on',
    'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
    'up', 'wow', 'yes', 'zero'
])

# ---------------------------
# Microphone
# ---------------------------
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 16000

# ---------------------------
# Load GIF Frames
# ---------------------------
def load_gif(gif_path):
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frames.append(ImageTk.PhotoImage(gif.copy()))
            gif.seek(len(frames))  # Move to next frame
    except EOFError:
        pass
    return frames

# ---------------------------
# Bikin Window
# ---------------------------
window = tk.Tk()
window.title("Speech Recognition DD")
window.geometry("1100x640")

# ---------------------------
# Load file gif
# ---------------------------
on_gif_path = "on.gif"
off_gif_path = "off.gif"

on_frames = load_gif(on_gif_path)
off_frames = load_gif(off_gif_path)

# ---------------------------
# Label
# ---------------------------
bg_label = tk.Label(window)
bg_label.place(relwidth=1, relheight=1)
state_label = tk.Label(window, text="ON", font=("Arial", 24, "bold"), fg="white", bg="black")
state_label.place(x=20, y=20)

# ---------------------------
# Animation loop
# ---------------------------
current_frames = on_frames
frame_index = 0

def animate_gif():
    global frame_index, current_frames

    frame = current_frames[frame_index]
    bg_label.config(image=frame)

    frame_index = (frame_index + 1) % len(current_frames)

    window.after(50, animate_gif)

animate_gif()

# ---------------------------
# Ganti Background
# ---------------------------
def change_background(predicted_label):
    global current_frames, frame_index

    # Switch ke on
    if predicted_label == "on" and current_frames is not on_frames:
        current_frames = on_frames
        frame_index = 0
        state_label.config(text="ON", fg="green")

    # Switch ke off
    elif predicted_label == "off" and current_frames is not off_frames:
        current_frames = off_frames
        frame_index = 0
        state_label.config(text="OFF", fg="red")

# ---------------------------
# Listen for Speech
# ---------------------------
def listen_for_speech():
    import pyaudio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

        max_amplitude = np.max(np.abs(data))
        if max_amplitude < 1000:
            continue  # skip silent

        data = data / max_amplitude

        # Make spectrogram
        spectrogram = tf.signal.stft(tf.convert_to_tensor(data, dtype=tf.float32), frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)[..., tf.newaxis]
        spectrogram = tf.expand_dims(spectrogram, axis=0)

        # Predict
        prediction = tf.nn.softmax(model.predict(spectrogram))
        predicted_label = label_names[np.argmax(prediction)]
        print("Predicted Label:", predicted_label)

        change_background(predicted_label)

# ---------------------------
# Thread
# ---------------------------
thread = threading.Thread(target=listen_for_speech, daemon=True)
thread.start()

window.mainloop()
