import csv
from time import sleep
import random
import requests
import os
from spotify_client import SpotifyClient
from dotenv import load_dotenv

def extract_track_id(spotify_uri):
    return spotify_uri.split(":")[-1]

def load_progress_from_csv(output_file):
    progress_dict = {}
    try:
        with open(output_file, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                progress_dict[row['isrc']] = row
        print(f"Loaded progress for {len(progress_dict)} tracks.")
    except FileNotFoundError:
        print("No existing progress file found. Starting fresh.")
    return progress_dict

def save_progress(output_file, output_columns, progress_dict):
    with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_columns)
        writer.writeheader()
        for features in progress_dict.values():
            writer.writerow({field: features.get(field, '') for field in output_columns})

def process_tracks(client_id, client_secret, input_file, output_file, output_columns):
    spotify_client = SpotifyClient(client_id, client_secret)
    progress_dict = load_progress_from_csv(output_file)
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            track_spotify_id = extract_track_id(row['spotify_uri'])
            if row['isrc'] in progress_dict:
                continue
            try:
                audio_features = spotify_client.get_audio_features(track_spotify_id)
                if audio_features:
                    progress_dict[track_spotify_id] = {
                        'isrc': row['isrc'],
                        'title': row['title'],
                        'artist': row['artist'],
                        'genre_id': row['genre_id'],
                        'genre_name': row['genre_name'],
                        'category_id': row['category_id'],
                        'category_name': row['category_name'],
                        **audio_features
                    }
                    print(f"Processed row: {row}.")
                sleep(random.uniform(0.2, 1))  # To avoid hitting API rate limits
            except requests.exceptions.RequestException as e:
                print(f"Error processing {row['isrc']}: {e}")
                break  # Break on error to save progress immediately

            save_progress(output_file, output_columns, progress_dict)

if __name__ == '__main__':
    load_dotenv("auth.env")
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')

    if client_id is None or client_secret is None:
        print("Failed to load client_id or client_secret from .env file")
        exit(1)

    input_file = 'src/spotify/tracks_cleaned.csv'
    output_file = 'src/spotify/tracks_cleaned_features.csv'
    output_columns = ['isrc', 'title', 'artist', 'genre_id', 'genre_name', 'category_id', 'category_name', 
                      'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                      'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

    process_tracks(client_id, client_secret, input_file, output_file, output_columns)