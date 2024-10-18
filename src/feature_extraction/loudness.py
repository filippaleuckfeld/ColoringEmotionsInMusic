import numpy as np
import pyloudnorm as pyln
import matplotlib.pyplot as plt
import traceback
from typing import Dict
from utility_functions import load_audio_stereo, extract_summary_statistics

def analyze_loudness(data: np.ndarray, sr: int = 44100, block_size: float = 0.4, plot_graph: bool = False) -> Dict:
    if data is None:
        return {}

    try:
        block_samples = int(block_size * sr)
        meter = pyln.Meter(sr, block_size=block_size)
        
        segment_loudness = []
        for start in range(0, len(data), block_samples):
            end = start + block_samples
            segment = data[start:end]
            if len(segment) == block_samples:
                loudness = meter.integrated_loudness(segment)
                if loudness != float('-inf'):
                    segment_loudness.append(loudness)

        segment_loudness = np.array(segment_loudness)
        summary_stats = extract_summary_statistics('loudness', 1, segment_loudness)

        if plot_graph:
            num_segments = len(segment_loudness)
        plot_loudness_over_time(segment_loudness, block_size, sr, num_segments)

        return summary_stats

    except Exception as e:
        print(f"Error in analyze_loudness: {e}")
        traceback.print_exc()
        return {}

def plot_loudness_over_time(segment_loudness: np.ndarray, block_size: float, sr: int, num_segments: int):
    try:
        # Correctly calculate the time axis based on the number of segments
        time_axis = np.arange(0, num_segments * block_size, block_size)
        plt.figure(figsize=(8, 2))
        plt.plot(time_axis[:len(segment_loudness)], segment_loudness)
        plt.xlabel('Time (s)')
        plt.ylabel('Loudness (LUFS)')
        plt.title('Perceived Loudness Over Time')
        plt.show()

    except Exception as e:
        print(f"Error in plot_loudness_over_time: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        audio_path = '/Volumes/Samsung T7/tracks/SE5IB2236769.wav'
        y = load_audio_stereo(audio_path)
        summary_stats = analyze_loudness(y, plot_graph=True)
        print("Summary Statistics of Loudness:", summary_stats)

    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()