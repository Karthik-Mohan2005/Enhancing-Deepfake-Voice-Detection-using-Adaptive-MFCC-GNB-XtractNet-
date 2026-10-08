import os
import numpy as np
import librosa

SAMPLE_RATE = 16000
SEGMENT_DURATION = 1.0
N_MFCC = 40

SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)


def load_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Normalize amplitude
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    return audio, sr

def segment_audio(audio):

    segments = []

    total_samples = len(audio)

    for start in range(0, total_samples, SEGMENT_SAMPLES):

        end = start + SEGMENT_SAMPLES

        segment = audio[start:end]

        if len(segment) < SEGMENT_SAMPLES:
            continue

        segments.append(segment)

    return segments


def extract_mfcc(segment):

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate(
        [mfcc_mean, mfcc_std]
    )

    return features
def extract_features_from_file(file_path):

    audio, sr = load_audio(file_path)

    segments = segment_audio(audio)

    features = []

    for segment in segments:

        mfcc_features = extract_mfcc(segment)

        features.append(mfcc_features)

    return np.array(features)


def process_directory(directory_path, label):


    X = []
    y = []

    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return np.array(X), np.array(y)

    files = sorted(os.listdir(directory_path))

    for filename in files:

        if not filename.lower().endswith(".wav"):
            continue

        file_path = os.path.join(
            directory_path,
            filename
        )

        try:

            file_features = extract_features_from_file(
                file_path
            )

            for feature_vector in file_features:

                X.append(feature_vector)
                y.append(label)

            print(
                f"Processed: {filename} "
                f"-> {len(file_features)} segments"
            )

        except Exception as e:

            print(
                f"Error processing {filename}: {e}"
            )

    return np.array(X), np.array(y)


def build_dataset(dataset_path):

    real_directory = os.path.join(
        dataset_path,
        "REAL"
    )

    fake_directory = os.path.join(
        dataset_path,
        "FAKE"
    )

    print("\nProcessing REAL audio...")
    
    X_real, y_real = process_directory(
        real_directory,
        label=0
    )

    print("\nProcessing FAKE audio...")

    X_fake, y_fake = process_directory(
        fake_directory,
        label=1
    )

    X = np.vstack(
        [X_real, X_fake]
    )

    y = np.concatenate(
        [y_real, y_fake]
    )

    return X, y

if __name__ == "__main__":

    dataset_path = "dataset"

    X, y = build_dataset(dataset_path)

    print("\n==============================")
    print("MFCC EXTRACTION COMPLETED")
    print("==============================")

    print("Feature shape :", X.shape)
    print("Label shape   :", y.shape)

    print("\nREAL samples  :", np.sum(y == 0))
    print("FAKE samples  :", np.sum(y == 1))

    if len(X) > 0:
        print("\nNumber of features per sample:", X.shape[1])