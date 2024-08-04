import csv
import os

def get_genres_and_categories(tracks_file, output_file):
    unique_genres_categories = set()

    with open(tracks_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            genre_id = row['genre_id']
            genre_name = row['genre_name']
            category_id = row['category_id']
            category_name = row['category_name']
            unique_genres_categories.add((genre_id, genre_name, category_id, category_name))

    sorted_genres_categories = sorted(unique_genres_categories, key=lambda x: x[3])

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['genre_id', 'genre_name', 'category_id', 'category_name'])
        writer.writerows(sorted_genres_categories)

    print(f'Data has been successfully extracted, sorted by category name, and saved to {output_file}')

def remove_tracks_by_genres_and_categories(tracks_file, genres_to_exclude_file, output_file):
    genres_to_exclude = set()

    with open(genres_to_exclude_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            genre_id = row['genre_id']
            genre_name = row['genre_name']
            category_id = row['category_id']
            category_name = row['category_name']
            genres_to_exclude.add((genre_id, genre_name, category_id, category_name))

    with open(tracks_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        tracks_to_keep = [row for row in reader if (row['genre_id'], row['genre_name'], row['category_id'], row['category_name']) not in genres_to_exclude]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['isrc', 'title', 'spotify_uri', 'artist', 'genre_id', 'genre_name', 'category_id', 'category_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tracks_to_keep)

    print(f'Tracks have been successfully cleaned and saved to {output_file}')

def remove_tracks_without_audiofile(input_file, output_file, audio_files_directory):
    def audio_file_exists(isrc):
        audio_file_path = os.path.join(audio_files_directory, f'{isrc}.wav')
        return os.path.isfile(audio_file_path)

    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows_with_audio = [row for row in reader if audio_file_exists(row['isrc'])]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_with_audio)

    print(f'Filtered tracks with existing audio files have been saved to {output_file}')

if __name__ == '__main__':
    tracks_file = "tracks.csv"
    audio_files_directory = "/Volumes/Samsung T7/tracks"

    get_genres_and_categories(tracks_file, "genres_categories.csv")
    remove_tracks_by_genres_and_categories(tracks_file, "genres_categories_to_exclude.csv", "tracks_genres_categories_cleaned.csv")
    remove_tracks_without_audiofile("tracks_genres_categories_cleaned.csv", "tracks_cleaned.csv", audio_files_directory)