import numpy as np
import librosa
import matplotlib.pyplot as plt
import traceback
from typing import Dict
from utility_functions import extract_summary_statistics, load_audio_mono

def calculate_time_axis(y: np.ndarray, sr: int = 44100) -> np.ndarray:
    return np.arange(0, len(y)) / sr

def calculate_time_axis_for_frames(y: np.ndarray, sr: int = 44100, hop_length: int = 1024) -> np.ndarray:
    n_frames = int(np.ceil(len(y) / hop_length))
    return np.arange(0, n_frames) * hop_length / sr

def bandwidth(y: np.ndarray, sr: int = 44100, hop_length: int = 1024, n_fft: int = 4096, plot: bool = False) -> Dict:
    try:
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        stats = extract_summary_statistics('bandwidth', 1, bw)
        if plot:
            plot_feature(bw, calculate_time_axis_for_frames(y, sr, hop_length), 'Bandwidth')
        return stats
    except Exception as e:
        print(f"Error in bandwidth: {e}")
        traceback.print_exc()
        return {}

def flatness(y: np.ndarray, hop_length: int = 1024, n_fft: int = 4096, plot: bool = False) -> Dict:
    try:
        fl = librosa.feature.spectral_flatness(y=y, hop_length=hop_length, n_fft=n_fft)
        stats = extract_summary_statistics('flatness', 1, fl)
        if plot:
            plot_feature(fl, calculate_time_axis_for_frames(y, sr, hop_length), 'Flatness')
        return stats
    except Exception as e:
        print(f"Error in flatness: {e}")
        traceback.print_exc()
        return {}

def centroid(y: np.ndarray, sr: int = 44100, hop_length: int = 1024, n_fft: int = 4096, plot: bool = False) -> Dict:
    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        stats = extract_summary_statistics('centroid', 1, cent)
        if plot:
            plot_feature(cent, calculate_time_axis_for_frames(y, sr, hop_length), 'Centroid')
        return stats
    except Exception as e:
        print(f"Error in centroid: {e}")
        traceback.print_exc()
        return {}

def mfcc(y: np.ndarray, sr: int = 44100, hop_length: int = 1024, n_fft: int = 4096, win_length: int = 4096, n_mfcc: int = 13, plot: bool = False) -> Dict:
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft, win_length=win_length)
        stats = {}
        for i, mfcc_band in enumerate(mfccs):
            band_stats = extract_summary_statistics('mfcc', i + 1, mfcc_band)
            stats.update(band_stats)
        if plot:
            # for i, mfcc_band in enumerate(mfccs):
            #     plot_feature(mfcc_band, calculate_time_axis_for_frames(y, sr, hop_length), f'MFCC {i + 1}')
            plot_mfcc(mfccs, sr, hop_length)
        return stats
    except Exception as e:
        print(f"Error in mfcc: {e}")
        traceback.print_exc()
        return {}

def zero_crossing_rate(y: np.ndarray, hop_length: int = 1024, plot: bool = False) -> Dict:
    try:
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
        stats = extract_summary_statistics('zero_crossing', 1, zcr)
        if plot:
            plot_feature(zcr, calculate_time_axis_for_frames(y, sr, hop_length), 'Zero Crossing Rate')
        return stats
    except Exception as e:
        print(f"Error in zero_crossing_rate: {e}")
        traceback.print_exc()
        return {}

def chroma(y: np.ndarray, sr: int = 44100, hop_length: int = 1024, n_fft: int = 4096, plot: bool = False) -> Dict:
    try:
        chr = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        stats = {}
        for i, chroma_band in enumerate(chr):
            band_stats = extract_summary_statistics('chroma', i + 1, chroma_band)
            stats.update(band_stats)
        if plot:
            # for i, chroma_band in enumerate(chr):
            #     plot_feature(chroma_band, calculate_time_axis_for_frames(y, sr, hop_length), f'Chroma {i + 1}')
            plot_chroma(chr, sr, hop_length)
        return stats
    except Exception as e:
        print(f"Error in chroma: {e}")
        traceback.print_exc()
        return {}

def spectral_contrast(y: np.ndarray, sr: int = 44100, hop_length: int = 1024, n_fft: int = 4096, plot: bool = False) -> Dict:
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        stats = {}
        for i, contrast_band in enumerate(contrast):
            band_stats = extract_summary_statistics('spectral_contrast', i + 1, contrast_band)
            stats.update(band_stats)
        if plot:
            # for i, contrast_band in enumerate(contrast):
            #     plot_feature(contrast_band, calculate_time_axis_for_frames(y, sr, hop_length), f'Spectral Contrast {i + 1}')
            plot_spectral_contrast(contrast, sr, hop_length)
        return stats
    except Exception as e:
        print(f"Error in spectral_contrast: {e}")
        traceback.print_exc()
        return {}

def rms(y: np.ndarray, hop_length: int = 1024, frame_length: int = 4096, plot: bool = False) -> Dict:
    try:
        rms_feature = librosa.feature.rms(y=y, hop_length=hop_length, frame_length=frame_length)
        stats = extract_summary_statistics('RMS Energy', 1, rms_feature)
        if plot:
            plot_feature(rms_feature, calculate_time_axis_for_frames(y, sr, hop_length), 'RMS Energy')
        return stats
    except Exception as e:
        print(f"Error in rms: {e}")
        traceback.print_exc()
        return {}

def plot_feature(feature: np.ndarray, time_axis: np.ndarray, feature_name: str):
    try:
        if feature.ndim == 2:
            feature = feature[0]  # In case feature is 2D, take the first row (common in librosa)
        plt.figure(figsize=(8, 2))
        plt.plot(time_axis[:len(feature)], feature)
        plt.xlabel('Time (s)')
        plt.ylabel('Feature Value')
        plt.title(f'{feature_name} Over Time')
        plt.show()
        plt.close()  # Close the figure to free up memory
    except Exception as e:
        print(f"Error in plot_feature: {e}")
        traceback.print_exc()

def plot_mfcc(mfcc: np.ndarray, sr: int, hop_length: int):
    plt.figure(figsize=(8, 2))
    librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length, x_axis='time', cmap='coolwarm')
    plt.colorbar(label='MFCC')
    plt.title('MFCC Over Time')
    plt.show()

def plot_chroma(chroma: np.ndarray, sr: int, hop_length: int):
    plt.figure(figsize=(8, 2))
    librosa.display.specshow(chroma, sr=sr, hop_length=hop_length, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.colorbar(label='Chroma')
    plt.title('Chroma Over Time')
    plt.show()

def plot_spectral_contrast(contrast: np.ndarray, sr: int, hop_length: int):
    plt.figure(figsize=(8, 2))
    librosa.display.specshow(contrast, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='coolwarm')
    plt.colorbar(label='Spectral Contrast')
    plt.title('Spectral Contrast Over Time')
    plt.show()

if __name__ == "__main__":
    try:
        audio_path = '/Volumes/Samsung T7/tracks/SE5IB2236769.wav'
        sr = 44100
        hop_length = 1024
        win_length = 4096
        n_fft = 4096

        y = load_audio_mono(audio_path, sr)
        time_axis = calculate_time_axis(y, sr)

        bw_stats = bandwidth(y, sr, hop_length, n_fft, plot=True)
        print(f"Bandwidth stats: {bw_stats}")

        fl_stats = flatness(y, hop_length, n_fft, plot=True)
        print(f"Flatness stats: {fl_stats}")

        cent_stats = centroid(y, sr, hop_length, n_fft, plot=True)
        print(f"Centroid stats: {cent_stats}")

        mfcc_stats = mfcc(y, sr, hop_length, n_fft, win_length, plot=True)
        print(f"MFCC stats: {mfcc_stats}")

        zcr_stats = zero_crossing_rate(y, hop_length, plot=True)
        print(f"Zero Crossing Rate stats: {zcr_stats}")

        chroma_stats = chroma(y, sr, hop_length, n_fft, plot=True)
        print(f"Chroma stats: {chroma_stats}")

        contrast_stats = spectral_contrast(y, sr, hop_length, n_fft, plot=True)
        print(f"Spectral Contrast stats: {contrast_stats}")

        rms_stats = rms(y, hop_length, win_length, plot=True)
        print(f"RMS stats: {rms_stats}")

    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()