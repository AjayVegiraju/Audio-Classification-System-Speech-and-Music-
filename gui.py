import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Assuming 'model' and 'scaler' are your trained SVM model and scaler
# You should load or define your trained model and scaler here
# For example:
# model = load_model('your_model_path')
# scaler = load_scaler('your_scaler_path')

class AudioFeatures:
    def __init__(self):
        self.zero_crossing_rate = None
        self.spectral_centroid = None
        self.spectral_bandwidth = None
        self.mfccs = None
        self.rms_energy = None


def extract_features(file_path):
    y, sr = librosa.load(file_path)
    features = AudioFeatures()

    features.zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
    features.spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features.spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    features.mfccs = np.mean(mfccs, axis=1)
    features.rms_energy = np.mean(librosa.feature.rms(y=y))

    return [features.zero_crossing_rate, features.spectral_centroid, features.spectral_bandwidth, features.rms_energy,
            *features.mfccs]


class AudioClassifierApp:
    def __init__(self, root, model, scaler):
        self.root = root
        self.model = model
        self.scaler = scaler
        self.root.title("Audio File Classifier")

        self.label = tk.Label(root, text="Select an audio file to classify", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn_select = tk.Button(root, text="Select File", command=self.select_file)
        self.btn_select.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            prediction = self.predict(file_path)
            self.result_label.config(text=f"Prediction: {'Music' if prediction == 1 else 'Speech'}")

    def predict(self, file_path):
        features = extract_features(file_path)
        features_scaled = self.scaler.transform([features])
        return self.model.predict(features_scaled)[0]


# Initialize the Tkinter application
root = tk.Tk()
app = AudioClassifierApp(root, model, scaler)
root.mainloop()