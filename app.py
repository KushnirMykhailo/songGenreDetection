from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import math
from keras.models import load_model
import librosa, librosa.display
from collections import Counter

model = load_model("genre_identification_model.keras")

song = ''
app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')

def predictGenre(model, X):
    X = X[np.newaxis, ...]
    # prediction - [ [0.1, 0.2, ...] ]
    prediction = model.predict(X)  # X -> (1, 130, 13, 1)
    print(prediction)
    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)  # [4]
    print(f"Predicted index: {int(predicted_index)}")
    return int(predicted_index)
def toMFCCs(song):
    SAMPLE_RATE = 22050
    DURATION = 30  # duration of every track is 30 seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    num_segments = 10
    signal, sr = librosa.load(song, sr=SAMPLE_RATE)
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / 512)  # 1.2 -> 2
    # process segments extracting mfcc and storing data
    data = list()
    for s in range(num_segments):
        start_sample = num_samples_per_segment * s  # s=0 -> 0
        finish_sample = start_sample + num_samples_per_segment  # s=0 -> num_samples_per_segment

        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                    sr=sr, n_fft=2048, hop_length=512, n_mfcc=13)
        mfcc = mfcc.T

        # store mfcc for segment if it has expected length
        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            data.append(mfcc.tolist())
    data = np.array(data)
    return data

def showResult(predicted_values : list):
    genre_dict = {0: "blues", 1: "classical", 2: "country", 3: "disco", 4: "hiphop", 5: "jazz", 6: "metal", 7: "pop",
                  8: "reggae",
                  9: "rock"}

    # Replace numerical values with genre names
    predicted_genres = [genre_dict[value] for value in predicted_values]

    # Count occurrences
    genre_counts = Counter(predicted_genres)

    # Calculate percentage
    total_values = len(predicted_values)
    percentage_genres = {genre: count / total_values * 100 for genre, count in genre_counts.items()}

    print("Count of each genre:", genre_counts)
    print("Percentage of each genre:", percentage_genres)
    return percentage_genres

@app.route('/predict', methods=['POST'])
def home():
    song = request.files['file']
    data = toMFCCs(song)
    predicted_values = list()
    for i, sample in enumerate(data[:10], start=0):
        sample_with_channel = sample[..., np.newaxis]
        print(f"Sample {i} shape: {sample_with_channel.shape}")
        predicted_values.append(predictGenre(model, sample_with_channel))
    print(predicted_values)
    prediction_result = showResult(predicted_values)
    return render_template('prediction.html', data=prediction_result)



if __name__ == "__main__":
    app.run(debug=True)