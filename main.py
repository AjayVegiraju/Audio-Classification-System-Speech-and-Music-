
from sklearn.metrics import classification_report, confusion_matrix
import tkinter as tk
from tkinter import messagebox
import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from playsound import playsound

class AudioFeatures:
    def __init__(self, file_name, file_type, data_set_type):
        self.file_name = file_name
        self.file_type = file_type
        self.data_set_type = data_set_type
        self.zero_crossing_rate = None
        self.spectral_centroid = None
        self.spectral_bandwidth = None
        self.mfccs = None
        self.rms_energy = None


    def print_features(self):
        print(f"File: {self.file_name}, ZCR: {self.zero_crossing_rate}, "
              f"Spectral Centroid: {self.spectral_centroid}, Bandwidth: {self.spectral_bandwidth}, "
              f"MFCCs: {self.mfccs}")

def process_audio_files(directory):
    features_list = []
    for file_name in os.listdir(directory):
        if file_name.lower().endswith('.wav'):
            file_type = get_file_type(file_name)
            data_set_type = os.path.basename(os.path.dirname(directory))  # 'Training' or 'Test'
            audio_path = os.path.join(directory, file_name)

            y, sr = librosa.load(audio_path)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            mfccs = librosa.feature.mfcc(y=y, sr=sr)
            rms = np.mean(librosa.feature.rms(y=y))
            avg_mfccs = np.mean(mfccs, axis=1)  # Averaging MFCCs across frames

            features = AudioFeatures(file_name, file_type, data_set_type)
            features.zero_crossing_rate = zcr
            features.spectral_centroid = centroid
            features.spectral_bandwidth = bandwidth
            features.mfccs = avg_mfccs.tolist()  # Convert to list for easier handling
            features.rms_energy = rms

            features_list.append(features)
    return features_list

def get_file_type(file_name):
    return "music" if file_name[0] == 'm' else "speech"

# Process training files
training_music_features = process_audio_files('audio/Training/Music')
training_speech_features = process_audio_files('audio/Training/Speech')
all_training_features = training_music_features + training_speech_features

# Process test files
test_music_features = process_audio_files('audio/Test/Music')
test_speech_features = process_audio_files('audio/Test/Speech')
all_test_features = test_music_features + test_speech_features

# Prepare training data
training_data = [[f.zero_crossing_rate, f.spectral_centroid, f.spectral_bandwidth, f.rms_energy, *f.mfccs] for f in all_training_features]
training_labels = [1 if f.file_type == 'music' else 0 for f in all_training_features]

# Prepare test data
test_data = [[f.zero_crossing_rate, f.spectral_centroid, f.spectral_bandwidth, f.rms_energy, *f.mfccs] for f in all_test_features]
test_labels = [1 if f.file_type == 'music' else 0 for f in all_test_features]

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(training_data)
X_test = scaler.transform(test_data)
y_train = training_labels
y_test = test_labels

# Train the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate the model
y_pred_test = model.predict(X_test)
print("Test Data Evaluation:")
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))


# Function to extract features from a single file for prediction
def extract_features_for_prediction(file_path):
    y, sr = librosa.load(file_path)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    avg_mfccs = np.mean(mfccs, axis=1)

    features = [zcr, centroid, bandwidth, rms, *avg_mfccs]
    return scaler.transform([features])

# GUI Code
def on_file_click(file_path):
    # Play the audio file
    playsound(file_path)

    # Extract features and predict
    features = extract_features_for_prediction(file_path)
    prediction = model.predict(features)[0]

    # Show prediction result
    result = "Music" if prediction == 1 else "Speech"
    messagebox.showinfo("Prediction", f"This file is a {result} file.")

# Setup the main window
root = tk.Tk()
root.title("Audio File Classifier")

# List audio files from the test dataset
test_music_dir = 'audio/Test/Music'
test_speech_dir = 'audio/Test/Speech'

for directory in [test_music_dir, test_speech_dir]:
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            file_path = os.path.join(directory, file)
            btn = tk.Button(root, text=file, command=lambda f=file_path: on_file_click(f))
            btn.pack()

root.mainloop()
