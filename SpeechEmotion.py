import soundfile # to read audio file
import numpy as np
import librosa # to extract speech features
import glob
import os
import pickle # to save model after training
from sklearn.model_selection import train_test_split # for splitting training and testing
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
from sklearn.metrics import accuracy_score # to measure how good we are

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result

# all emotions on RAVDESS dataset
int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# we allow only these emotions ( feel free to tune this on your need )
AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "neutral",
    "happy"
}

#Load the data and extract features for each sound file
def load_data(test_size = 0.2):
  x, y = [], []
  for folder in glob.glob('C:/Users/91779/Downloads/SRE/Ravdess/audio_speech_actors_01-24/Actor_*'):
    print(folder)
    for file in glob.glob(folder + '/*.wav'):
      file_name = os.path.basename(file)
      emotion = int2emotion[file_name.split('-')[2]]
      if emotion not in AVAILABLE_EMOTIONS:
        continue
      feature = extract_feature(file, mfcc = True, chroma = True, mel = True)
      x.append(feature)
      y.append(emotion)
    return train_test_split(np.array(x), y, test_size = test_size, random_state = 9)


x_train, x_test, y_train, y_test=load_data(test_size=0.1)
print(f'X_train:{x_train}')


# print some details
# number of samples in training data
print("[+] Number of training samples:", x_train.shape[0])#0
# number of samples in testing data
print("[+] Number of testing samples:", x_test.shape[0])#0
# number of features used
# this is a vector of features extracted 
# using extract_features() function
print("[+] Number of features:", x_train.shape[1])

# best model, determined by a grid search
model_params = {
    'alpha': 0.001,
    'batch_size':256,#256
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 700, 
}

# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
model.fit(x_train, y_train)

# predict 25% of data to measure how good we are
y_pred = model.predict(x_test)
# calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

filename="ser_model"
pickle.dump(model,open(filename,'wb'))
loaded_model=pickle.load(open(filename,'rb'))
result = loaded_model.score(x_train, y_train)
print(result)

import tkinter as tk
from tkinter import filedialog
import sounddevice as sd
import soundfile as sf

# Create a Tkinter window
root = tk.Tk()
root.title("Emotion Recognition")
root.configure(background='Gray')
root.geometry("500x500")

frame = tk.Frame(root, bg='black')
frame.pack(pady=20)
#root.geometry("500x500")

def upload_file():
    global audio_file_path
    audio_file_path = filedialog.askopenfilename()
    label_file_path.config(text="File Path: " + audio_file_path)

# Function to play audio file
def play_audio():
    if audio_file_path:
        data, samplerate = sf.read(audio_file_path)
        sd.play(data, samplerate)
        sd.wait()
    else:
        print("Please upload an audio file first.")    
# Function to predict emotion
def predict_emotion():
    feature = extract_feature(audio_file_path, mfcc=True, chroma=True, mel=True)
    feature = feature.reshape(1,-1)
    prediction = loaded_model.predict(feature)
    label_prediction.config(text="" + prediction[0])

# Button to upload file
btn_upload = tk.Button(frame, text="Upload File", font=("times new roman",30,"bold"),bd=9, command=upload_file, bg='blue', fg="green")
btn_upload.pack(pady=5)

# Label to show uploaded file path
label_file_path = tk.Label(frame, text="", font=("times new roman",20))
label_file_path.pack(pady=5)

# Button to play audio
btn_play = tk.Button(frame, text="Play Audio",font=("times new roman",25,'bold'), command=play_audio, bg="red", fg="white")
btn_play.pack(pady=5)

# Button to predict emotion
btn_predict = tk.Button(frame, text="Predict Emotion", font=("times new roman", 25, "bold"), command=predict_emotion, bg='red', fg='white')
btn_predict.pack(pady=5)

# Label to show predicted emotion
label_prediction = tk.Label(frame, text="", font=("times new roman", 20, "bold"))
label_prediction.pack(pady=5)
root.mainloop()