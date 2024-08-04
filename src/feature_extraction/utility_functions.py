import numpy as np
from scipy.stats import zscore
import librosa
import os
import matplotlib.pyplot as plt


def extract_summary_statistics(feature_name, band_nr, feature):
    """ Helper function to calculate summary statistics for a given feature array. """
    return {
        f'{feature_name}_{band_nr}_mean': np.mean(feature),
        f'{feature_name}_{band_nr}_std_dev': np.std(feature),
        f'{feature_name}_{band_nr}_min': np.min(feature),
        f'{feature_name}_{band_nr}_max': np.max(feature),
        f'{feature_name}_{band_nr}_median': np.median(feature),
        f'{feature_name}_{band_nr}_q1': np.percentile(feature, 25),
        f'{feature_name}_{band_nr}_q3': np.percentile(feature, 75),
        f'{feature_name}_{band_nr}_iqr': np.percentile(feature, 75) - np.percentile(feature, 25),
        f'{feature_name}_{band_nr}_skewness': np.mean((feature - np.mean(feature))**3) / np.std(feature)**3,
        f'{feature_name}_{band_nr}_kurtosis': np.mean((feature - np.mean(feature))**4) / np.std(feature)**4
    }

def load_audio_mono(audio_file_path, trim_silence=True, sr=44100):
    try:
        y, _ = librosa.load(audio_file_path, sr=sr, mono=True)
        if trim_silence:
            y, _ = librosa.effects.trim(y)
        return y
    except Exception as e:
        raise RuntimeError(f"Error loading {audio_file_path}: {e}")
    
def load_audio_stereo(audio_file_path, sr=44100):
    try:
        y, _ = librosa.load(audio_file_path, sr=sr, mono=False)
        y_trimmed = trim_silence(y)
        if y_trimmed.ndim == 1:
            y_trimmed = y_trimmed[np.newaxis, :]
        else:
            y_trimmed = y_trimmed.T
        return y_trimmed
    except Exception as e:
        print(f"Error loading {audio_file_path}: {e}")
        return None
    
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def trim_silence(y, threshold=10):
    trimmed, _ = librosa.effects.trim(y, top_db=threshold)
    return trimmed

def visualize_trim(audio_path, thresholds):
    y, _ = librosa.load(audio_path)
    
    plt.figure(figsize=(12, 6))
    
    for i, t in enumerate(thresholds):
        y_trimmed = trim_silence(y, threshold=t)
        plt.subplot(len(thresholds), 1, i+1)
        plt.plot(y_trimmed)
        plt.title(f'Trimmed with threshold {t} dB')
    
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    audio_path = '/Volumes/Samsung T7/tracks/SE5IB2311119.wav'
    visualize_trim(audio_path, thresholds=[10, 20, 30])