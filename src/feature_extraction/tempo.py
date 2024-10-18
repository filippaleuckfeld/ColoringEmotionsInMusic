import os
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import traceback
from typing import Dict
from utility_functions import load_audio_mono, extract_summary_statistics, ensure_directory_exists

def extract_isrc(audio_path: str) -> str:
    return os.path.splitext(os.path.basename(audio_path))[0]

def audio_with_clicks(audio_path: str, y: np.ndarray, beat_times: np.ndarray, sr: int):
    try:
        isrc = extract_isrc(audio_path)
        clicks_plp = librosa.clicks(times=beat_times, sr=sr, click_freq=1000, length=len(y))
        audio_with_clicks_plp = y + clicks_plp

        output_dir = 'tempo_output_audio/beat'
        output_audio_path_plp = os.path.join(output_dir, f'{isrc}_click.wav')
        ensure_directory_exists(output_dir)

        sf.write(output_audio_path_plp, audio_with_clicks_plp, sr)

    except Exception as e:
        print(f"Error in audio_with_clicks for {audio_path}: {e}")
        traceback.print_exc()

def save_plp_function(audio_path: str, y: np.ndarray, pulse: np.ndarray, sr: int, hop_length: int):
    try:
        isrc = extract_isrc(audio_path)
        interp_times = np.arange(len(y)) / float(sr)
        plp_times = np.arange(len(pulse)) * hop_length / float(sr)
        plp_interpolated = np.interp(interp_times, plp_times, pulse)

        if np.ptp(plp_interpolated) == 0:
            print(f"Warning: No variation in PLP data for {isrc}. Normalization skipped.")
            return
        else:
            plp_normalized = (plp_interpolated - np.min(plp_interpolated)) / np.ptp(plp_interpolated)

        output_dir = 'output_audio/plp'
        output_path = os.path.join(output_dir, f'{isrc}_PLP.wav')
        ensure_directory_exists(output_dir)

        sf.write(output_path, plp_normalized, sr)

    except Exception as e:
        print(f"Error in save_plp_function for {audio_path}: {e}")
        traceback.print_exc()

def plot_plp_graph(pulse: np.ndarray, beat_times: np.ndarray, sr: int, hop_length: int):
    try:
        times = librosa.times_like(pulse, sr=sr, hop_length=hop_length)
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(times, librosa.util.normalize(pulse), label='PLP', color='b')
        ax.vlines(beat_times, 0, 1, alpha=0.5, color='r', linestyle='--', label='PLP Beats')
        ax.legend()
        ax.set_title('Tempo Over Time')
        ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in plot_plp_graph: {e}")
        traceback.print_exc()

def plp(y: np.ndarray, 
        audio_path: str, 
        average_bpm: float, 
        sr: int = 44100, 
        hop_length: int = 512, 
        win_length: int = 1024, 
        save_audio_with_clicks: bool = False, 
        save_plp: bool = False, 
        plot_graph: bool = False) -> Dict:
    if average_bpm == 0.0 or y is None:
        return {}

    try:
        tempo_min = average_bpm * 0.8
        tempo_max = average_bpm * 1.2

        pulse = librosa.beat.plp(y=y, sr=sr, hop_length=hop_length, win_length=win_length, tempo_min=tempo_min, tempo_max=tempo_max)
        beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
        beat_times = librosa.frames_to_time(beats_plp, sr=sr, hop_length=hop_length)

        summary_stats = {}
        if len(beat_times) > 1:
            intervals = np.diff(beat_times)
            bpm = 60.0 / intervals
            summary_stats = extract_summary_statistics('tempo', 1, bpm)

        if save_audio_with_clicks:
            audio_with_clicks(audio_path, y, beat_times, sr)

        if save_plp:
            save_plp_function(audio_path, y, pulse, sr, hop_length)

        if plot_graph:
            plot_plp_graph(pulse, beat_times, sr, hop_length)

        return summary_stats

    except Exception as e:
        print(f"Error in plp function for {audio_path}: {e}")
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    audio_path = '/Volumes/Samsung T7/tracks/SE5IB2236769.wav'
    average_bpm = 93  # Example BPM value

    mono_audio = load_audio_mono(audio_path)

    # Test PLP function
    summary_stats = plp(mono_audio, audio_path, average_bpm, save_audio_with_clicks=False, save_plp=False, plot_graph=True)
    print("Summary stats:", summary_stats)