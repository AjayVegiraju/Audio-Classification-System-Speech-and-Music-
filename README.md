
1. How to Run the Code and Use the System:
   
Prerequisites:
Python version <= 3.11.5 (In order accomodate sklearn ML model )

Dependency Install Instructions:

Enter the following commands in the terminal on the project directory:

pip install librosa numpy scikit-learn
pip install playsound==1.2.2

Running instructions:

Run the following command in the terminal on the project directory:

python main.py




3. Libraries/Tools/Techniques Used:
   
Libraries:

Librosa: For Audio Processing and Feature Extraction. 
Numpy: For Numerical Operations.
Scikit-Learn: For Machine Learning & Data Scaling, Model Training. 
Playsound: For Playing Audio Files in the GUI.
Tkinter: For Python GUI Development.
Techniques:
Feature Extraction: Extracts Audio Features like Zero-Crossing Rate, Spectral Centroid, Bandwidth, Energy, & MFCC’s.
MFCCs (Mel-Frequency Cepstral Coefficients) are coefficients that represent the characteristics of a sound signal, particularly its power spectrum. They are obtained by first converting the audio signal into a frequency domain using techniques like the Discrete Fourier Transform (DFT). Then, this frequency domain is adjusted using the mel-scale to mimic human auditory perception. From this mel-scaled spectrum, cepstral coefficients are extracted. MFCCs are especially valuable in emphasizing aspects of the sound that are crucial for understanding human speech, while filtering out less important information.
SVM Classifier: Uses a Support Vector Machine with a Linear Kernal for Classification. 


6. Features and Data Sets:

Features Extracted: Zero Crossing Rate, Spectral Centroid, Spectral Bandwidth, Root Mean Square Energy, and Mel-Frequency Cepstral Coefficients (MFCCs).
Training Data: Audio files in 'audio/Training/Music' and 'audio/Training/Speech'.
Testing Data: Audio files in 'audio/Test/Music' and 'audio/Test/Speech'.

4. Model Output vs. Ground Truth and Performance Metrics:

The script uses the confusion_matrix and classification_report from sklearn.metrics to evaluate the model on the test data.
Explanation of Precision, Recall, and Confusion Matrix:
Precision: The ratio of correctly predicted positive observations to the total predicted positives. High precision relates to a low false positive rate.
Recall : The ratio of correctly predicted positive observations to all observations in the actual class. High recall indicates most of the positive class is correctly recognized.
Confusion Matrix: A table used to describe the performance of a classification model. It contains four elements: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
True Positives (TP): Correctly predicted positive values.
True Negatives (TN): Correctly predicted negative values.
False Positives (FP): Incorrectly predicted positive values.
False Negatives (FN): Incorrectly predicted negative values.
According to our model, the underlying chart represents the 4 elements of the confusion matrix:

TN: 4
FP: 3
FN: 2
TP: 5






Precision and Recall: These are calculated as part of the classification report. Precision measures the accuracy of positive predictions, while recall measures the fraction of positives that were correctly identified.

Terminal Output for our model:
The ML model reports Speech and Music based on the labels, where 0 represents speech and 1 represents music.
For Speech (0):
Precision (0.67): Out of all the instances where the model predicted speech, 67% were actually speech.
Recall (0.57): Of all the actual speech instances, the model correctly identified 57% as speech.
F1-Score (0.62): A harmonic mean of precision and recall, giving a balance between the two. Higher is better.
Support (7): The number of actual occurrences of speech in the dataset.

For Music (1):
Precision (0.62): Out of all the instances where the model predicted music, 62% were actually music.
Recall (0.71): Of all the actual music instances, the model correctly identified 71% as music.
F1-Score (0.67): A harmonic mean of precision and recall.
Support (7): The number of actual occurrences of music in the dataset.

Overall Metrics:
Accuracy (0.64): Overall, the model correctly identified 64% of both music and speech correctly.


5. Sampling & Testing Model: 

Speech Files:
Training (training_speech_features):
Testing (test_speech_features):

Music Files:
Training (training_music_features): 
Testing (test_music_features):

