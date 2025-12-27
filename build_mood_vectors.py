import pandas as pd
import numpy as np
import os

print("--- STARTING 5-DIMENSIONAL MOOD VECTORIZATION ---")

# Format: [Valence, Energy, Danceability, Acousticness, Tempo]
# All values must be between 0.0 and 1.0

genre_mood_map = {
    # Genre        [Val,  Egy,  Dnc,  Aco,  Tmp]
    "Comedy":      [0.9, 0.7, 0.8, 0.2, 0.6], # Fun, bouncy, not too fast
    "Adventure":   [0.8, 0.8, 0.5, 0.1, 0.7], # Epic, grand, orchestral (low dance)
    "Action":      [0.7, 0.9, 0.4, 0.0, 0.9], # Fast, intense, loud (low acoustic)
    "Fantasy":     [0.7, 0.6, 0.4, 0.3, 0.5], # Magical, moderate pacing
    "Romance":     [0.8, 0.4, 0.6, 0.6, 0.4], # Slow, acoustic, swaying
    "Drama":       [0.4, 0.4, 0.3, 0.7, 0.3], # Serious, slow, dialogue-heavy
    "Thriller":    [0.3, 0.8, 0.4, 0.1, 0.7], # Tense, fast, electronic
    "Horror":      [0.2, 0.9, 0.3, 0.1, 0.6], # Scary, sharp spikes
    "Mystery":     [0.3, 0.5, 0.3, 0.4, 0.4], # Slow burn
    "Sci-Fi":      [0.6, 0.7, 0.5, 0.0, 0.6], # Synthetic, modern
    "Crime":       [0.3, 0.7, 0.4, 0.2, 0.5], # Gritty, moderate
    "Documentary": [0.5, 0.3, 0.2, 0.8, 0.3], # Real, slow, acoustic
    "War":         [0.2, 0.8, 0.2, 0.1, 0.7], # Chaotic, loud
    "Musical":     [0.9, 0.7, 0.9, 0.4, 0.6], # It's literally dancing
    "Children":    [0.9, 0.6, 0.8, 0.3, 0.6], # Bouncy, simple
    "Animation":   [0.8, 0.7, 0.7, 0.2, 0.6], # Energetic
    "Western":     [0.5, 0.6, 0.4, 0.7, 0.5], # Guitars, horses (mid-tempo)
    "Film-Noir":   [0.2, 0.3, 0.3, 0.5, 0.3], # Very slow, jazzy
    "IMAX":        [0.7, 0.8, 0.5, 0.1, 0.7]  # Spectacle
}


print("\n Loading Data...")
movies = pd.read_csv('data/movies.dat', sep='\t', encoding='latin-1')
genres_df = pd.read_csv('data/movie_genres.dat', sep='\t', encoding='latin-1')

movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
genres_df['movieID'] = pd.to_numeric(genres_df['movieID'], errors='coerce')

print("   Combining Genres...")
movies_genres_combined = genres_df.groupby('movieID')['genre'].apply('|'.join).reset_index()
movies = movies.merge(movies_genres_combined, left_on='id', right_on='movieID', how='left')
movies['genre'] = movies['genre'].fillna('Unknown')

print(f"   Merged! Now processing {len(movies)} movies.")

def calculate_movie_mood(genre_string):
    if genre_string == 'Unknown':
        return [0.5, 0.5, 0.5, 0.5, 0.5]
    
    genres = genre_string.split('|')
    vectors = []
    
    for g in genres:
        if g in genre_mood_map:
            vectors.append(genre_mood_map[g])
            
    if not vectors:
        return [0.5, 0.5, 0.5, 0.5, 0.5]
    
    # Average all 5 dimensions
    return np.mean(vectors, axis=0).tolist()

# Apply
mood_vectors = movies['genre'].apply(calculate_movie_mood)

# Expand the list into columns
mood_df = pd.DataFrame(mood_vectors.tolist(), columns=['valence', 'energy', 'danceability', 'acousticness', 'tempo'])
movies = pd.concat([movies, mood_df], axis=1)

# Save
movies_clean = movies[['id', 'title', 'genre', 'valence', 'energy', 'danceability', 'acousticness', 'tempo']]
movies_clean.to_csv('data/movie_moods.csv', index=False)
print(f" Generated 5D Moods for {len(movies_clean)} movies.")



print("\n Processing Artists...")
spotify = pd.read_csv('data/spotify_data.csv')

# Identify Artist Column
possible_artist_cols = ['artists', 'artist_name', 'artist', 'track_artist']
artist_col = next((c for c in possible_artist_cols if c in spotify.columns), None)

if not artist_col:
    raise ValueError("Could not find artist column in Spotify CSV.")

# Check for new columns
required_cols = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']
missing = [c for c in required_cols if c not in spotify.columns]
if missing:
    raise ValueError(f"Spotify CSV is missing columns: {missing}")

# NORMALIZE TEMPO
# Tempo is usually 50-200 BPM. We need it 0-1.
spotify['tempo'] = spotify['tempo'] / 200.0
spotify['tempo'] = spotify['tempo'].clip(0, 1)

# Group and Average
artist_moods = spotify.groupby(artist_col)[required_cols].mean().reset_index()
artist_moods.rename(columns={artist_col: 'artist_name'}, inplace=True)
artist_moods['artist_clean'] = artist_moods['artist_name'].astype(str).str.lower().str.strip()

artist_moods.to_csv('data/artist_moods.csv', index=False)
print(f" Generated 5D Moods for {len(artist_moods)} artists.")
print(artist_moods.head())

print("\n SUCCESS: 5-Dimensional Data is ready.")