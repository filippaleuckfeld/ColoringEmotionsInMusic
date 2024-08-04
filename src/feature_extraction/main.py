import os
import pandas as pd
import traceback

from loudness import analyze_loudness
from librosa_features import bandwidth, centroid, flatness, mfcc, zero_crossing_rate, chroma, spectral_contrast, rms
from tempo import plp
from utility_functions import load_audio_mono, load_audio_stereo

def load_cache(output_csv_path=None):
    if output_csv_path and os.path.exists(output_csv_path):
        df = pd.read_csv(output_csv_path)
        return {row['isrc']: True for _, row in df.iterrows()}
    return {}

def process_audio_file(audio_path, tempo, isrc):
    try:
        mono_audio_not_trimmed = load_audio_mono(audio_path, False)
        mono_audio = load_audio_mono(audio_path)
        stereo_audio = load_audio_stereo(audio_path)

        # Extract features and their summary statistics
        pulse_summary_stats = plp(mono_audio_not_trimmed, audio_path, tempo, save_audio_with_clicks=True, save_plp=True)
        loudness_summary_stats = analyze_loudness(stereo_audio)
        centroid_summary_stats = centroid(mono_audio)
        bandwidth_summary_stats = bandwidth(mono_audio)
        flatness_summary_stats = flatness(mono_audio)
        mfcc_features = mfcc(mono_audio)
        zcr_summary_stats = zero_crossing_rate(mono_audio)
        chroma_features = chroma(mono_audio)
        contrast_features = spectral_contrast(mono_audio)
        rms_summary_stats = rms(mono_audio)

        # Initialize result dictionary with ISRC
        result = {'isrc': isrc}

        # Use update() to flatten and merge dictionaries
        result.update(pulse_summary_stats)
        result.update(loudness_summary_stats)
        result.update(centroid_summary_stats)
        result.update(bandwidth_summary_stats)
        result.update(flatness_summary_stats)
        result.update(zcr_summary_stats)
        result.update(rms_summary_stats)
        result.update(mfcc_features)
        result.update(chroma_features)
        result.update(contrast_features)
        return result

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        traceback.print_exc()
        return None

def append_to_csv(output_csv_path, data):
    df = pd.DataFrame([data])
    if not os.path.isfile(output_csv_path):
        df.to_csv(output_csv_path, index=False, mode='w')
    else:
        df.to_csv(output_csv_path, index=False, mode='a', header=False)

def process_csv(csv_path, output_csv_path):
    processed_cache = load_cache(output_csv_path)

    if not os.path.exists(csv_path):
        print(f"Input CSV file {csv_path} does not exist.")
        return

    try:
        with open(csv_path, 'r') as f:
            reader = pd.read_csv(f, chunksize=1)
            for chunk in reader:
                for row in process_rows(chunk, processed_cache):
                    append_to_csv(output_csv_path, row)
    except pd.errors.EmptyDataError:
        print(f"Input CSV file {csv_path} is empty.")
        return

def process_rows(df, processed_cache):
    for _, row in df.iterrows():
        isrc = str(row['isrc'])
        tempo = row['tempo']
        audio_path = "/Volumes/Samsung T7/tracks/" + isrc + ".wav"
        print(f'Processing audio {isrc}')
        if not os.path.exists(audio_path):
            continue

        if isrc in processed_cache:
            continue

        result = process_audio_file(audio_path, tempo, isrc)
        if result:
           processed_cache[isrc] = True
           yield result

if __name__ == "__main__":
    process_csv("tracks_cleaned_features.csv", "tracks_features.csv")