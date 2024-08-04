import os
import pandas as pd
from tkinter import *
from tkinter import ttk
from pygame import mixer

mixer.init()

music_directory = "/Volumes/Samsung T7/tracks"
merged_table_csv = "src/feature_extraction/tracks_cleaned_features.csv"

df_songs = pd.read_csv(merged_table_csv)

def play_music(music_file_path):
    mixer.music.load(music_file_path)
    mixer.music.play()
    # Initial setting for the slider:
    seek_slider.set(0)

def update_position(value):
    mixer.music.set_pos(float(value))

def update_song_listbox(songs):
    listbox.delete(0, END)
    for song in songs:
        listbox.insert(END, song)

def play_selected_song(event):
    selection = event.widget.curselection()
    if selection:
        index = selection[0]
        song_info = event.widget.get(index)
        isrc = song_info.split(' - ')[0]
        song_path = os.path.join(music_directory, f"{isrc}.wav")
        play_music(song_path)

def filter_songs():
    valence_min = float(valence_min_var.get())
    valence_max = float(valence_max_var.get())
    energy_min = float(energy_min_var.get())
    energy_max = float(energy_max_var.get())
    filtered_songs = df_songs[(df_songs['valence'] >= valence_min) & (df_songs['valence'] <= valence_max) & (df_songs['energy'] >= energy_min) & (df_songs['energy'] <= energy_max)]
    song_list = [f"{row['isrc']} - {row['title']}" for index, row in filtered_songs.iterrows()]
    update_song_listbox(song_list)

def main():
    global root, listbox, valence_min_var, valence_max_var, energy_min_var, energy_max_var, seek_slider
    
    root = Tk()
    root.title("Music Player based on Valence and Energy")

    filter_frame = Frame(root)
    filter_frame.pack(fill=X, padx=5, pady=5)

    Label(filter_frame, text="Valence Min:").pack(side=LEFT)
    valence_min_var = StringVar(value="0.8")
    Entry(filter_frame, textvariable=valence_min_var, width=5).pack(side=LEFT, padx=5)

    Label(filter_frame, text="Valence Max:").pack(side=LEFT)
    valence_max_var = StringVar(value="0.9")
    Entry(filter_frame, textvariable=valence_max_var, width=5).pack(side=LEFT, padx=5)

    Label(filter_frame, text="Energy Min:").pack(side=LEFT)
    energy_min_var = StringVar(value="0.5") 
    Entry(filter_frame, textvariable=energy_min_var, width=5).pack(side=LEFT, padx=5)

    Label(filter_frame, text="Energy Max:").pack(side=LEFT)
    energy_max_var = StringVar(value="0.8")
    Entry(filter_frame, textvariable=energy_max_var, width=5).pack(side=LEFT, padx=5)

    Button(filter_frame, text="Filter", command=filter_songs).pack(side=LEFT, padx=5)

    listbox = Listbox(root)
    listbox.pack(fill=BOTH, expand=True)
    listbox.bind('<<ListboxSelect>>', play_selected_song)

    filter_songs()  # Initial filtering
    
    # Slider setup
    seek_slider = ttk.Scale(root, from_=0, to=100, orient=HORIZONTAL, command=update_position)  # Default to 100, will update on song play
    seek_slider.pack(fill=X, expand=True)

    root.mainloop()

if __name__ == "__main__":
    main()
