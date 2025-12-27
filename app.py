import streamlit as st
import pandas as pd
import numpy as np
import NCF

st.set_page_config(
    page_title="InfyMix: Hybrid Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background: linear-gradient(to bottom, #0e1117, #161b22); }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00d26a; }
    div[data-testid="stVerticalBlock"] > div { border-radius: 10px; }
    div[data-testid="stMarkdownContainer"] p { font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)

MOOD_COLS = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']

@st.cache_data
def load_base_data():
    try:
        # Thumbnails
        movie_thumbnails = pd.read_csv('data/movies.dat', sep='\t', usecols=['id', 'imdbPictureURL'], encoding='latin-1')
        movie_thumbnails['imdbPictureURL'] = movie_thumbnails['imdbPictureURL'].str.replace('http', 'https', regex=False)

        # Mood Data
        mood_data = pd.read_csv('data/movie_moods.csv')
        
        # Artists
        artists = pd.read_csv('data/artist_moods.csv')

        # NCF Predictions
        ncf_model = NCF.build_NCF_model()
        all_user_recs = pd.DataFrame(
            ncf_model.recommendations, columns=['userID', 'id', 'predicted_rating'])
        
        # Ratings scaling (0-1 to 0-5)
        all_user_recs['predicted_rating'] = all_user_recs['predicted_rating'] * 5.0
        
        # Merge Thumbnails + Base Movie Data
        base_movies = pd.merge(mood_data, movie_thumbnails, on='id')
        
        return base_movies, artists, all_user_recs
        
    except FileNotFoundError:
        return None, None, None

# Load the data
base_movies_df, artists_df, all_user_recs_df = load_base_data()

if base_movies_df is None:
    st.error(" Critical Error: Data files not found. Run 'build_mood_vectors.py' first.")
    st.stop()

# User Logic 
if 'simulated_user_id' not in st.session_state:
    st.session_state['simulated_user_id'] = all_user_recs_df['userID'].sample(random_state=42).iloc[0]

# Filter ratings for specific user
current_user_ratings = all_user_recs_df[all_user_recs_df['userID'] == st.session_state['simulated_user_id']]

# Merge predictions into the main movie list
movies_df = pd.merge(base_movies_df, current_user_ratings[['id', 'predicted_rating']], on='id')

# Prepare Vectors
movie_vectors = movies_df[MOOD_COLS].values
movie_norms = np.linalg.norm(movie_vectors, axis=1)

# Get top 20 movies this user is predicted to like
user_top_picks = movies_df.sort_values(by='predicted_rating', ascending=False).head(20)
user_fav_genres = user_top_picks['genre'].str.split('|').explode().value_counts().head(3).index.tolist()
user_fav_genres_str = ", ".join(user_fav_genres)

def get_recommendations(artist_name):
    clean_name = artist_name.lower().strip()
    artist_row = artists_df[artists_df['artist_clean'] == clean_name]
    if artist_row.empty: return None, None
        
    artist_vector = artist_row[MOOD_COLS].values[0]
    
    # Cosine Similarity
    dot_products = np.dot(movie_vectors, artist_vector)
    artist_norm = np.linalg.norm(artist_vector)
    similarity_scores = dot_products / ((movie_norms * artist_norm) + 1e-9)
    
    # HYBRID SCORE: Mood (0-1) * Rating (Normalized 0-1)
    normalized_rating = movies_df['predicted_rating'] / 5.0
    movies_df['score'] = similarity_scores * normalized_rating
    movies_df['mood_pct'] = similarity_scores
    
    return movies_df.sort_values(by='score', ascending=False).head(10), artist_vector

with st.sidebar:
    st.title("🎧 InfyMix")
    st.markdown("---")
    
    st.subheader(" User Profile")
    col1, col2 = st.columns([3, 1])
    col1.markdown(f"**ID: #{st.session_state['simulated_user_id']}**")
    if col2.button("🔄", help="Shuffle User"):
        new_user = np.random.choice(all_user_recs_df['userID'].unique())
        st.session_state['simulated_user_id'] = new_user
        st.rerun()

    st.info(f"** Loves:** {user_fav_genres_str}")
    with st.expander(" User's Favorites (Top 3)"):
        for _, row in user_top_picks.head(3).iterrows():
            st.write(f"• **{row['title']}** ({row['predicted_rating']:.1f})")
    
    st.markdown("---")
    
    st.subheader(" Select Music")
    all_artists = sorted(artists_df['artist_name'].unique())
    default_idx = all_artists.index("The Weeknd") if "The Weeknd" in all_artists else 0
    selected_artist = st.selectbox("Artist:", all_artists, index=default_idx)
    
    if st.button(" Find Movies", use_container_width=True):
        st.session_state['run'] = True

st.title(" Music-to-Movie Bridge")

if st.session_state.get('run'):
    st.markdown(f"### Because User #{st.session_state['simulated_user_id']} likes **{selected_artist}**...")
    rec_movies, mood_vec = get_recommendations(selected_artist)
    
    if rec_movies is not None:
        # Mood DNA
        with st.container(border=True):
            st.markdown(f"** Mood DNA: {selected_artist}**")
            cols = st.columns(5)
            stats = zip(["Valence ", "Energy ", "Dance ", "Acoustic ", "Tempo ⏱"], mood_vec)
            for col, (label, val) in zip(cols, stats):
                col.metric(label, f"{val:.2f}")

        st.markdown("### ") 

        rec_movies_filtered = rec_movies[rec_movies['imdbPictureURL'].notna() & (rec_movies['imdbPictureURL'] != '')]
        
        for row_idx in range(0, len(rec_movies_filtered), 4):
            cols = st.columns(4)
            for col_idx, (index, row) in enumerate(rec_movies_filtered.iloc[row_idx:row_idx+4].iterrows()):
                with cols[col_idx]:
                    with st.container(border=True):
                        try:
                            st.image(row['imdbPictureURL'], width='stretch')
                        except:
                            st.write(" No Image")
                        
                        st.markdown(f"**{row['title']}**")
                        st.caption(f"_{row['genre']}_")
                        st.markdown("---")
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**{int(row['mood_pct']*100)}%**")
                            st.caption("Match")
                        with c2:
                            st.markdown(f"** {row['predicted_rating']:.1f}**")
                            st.caption("Rating")
                            
                        st.progress(int(row['mood_pct']*100))
    else:
        st.warning("Artist not found!")
else:
    st.markdown(" **Select an artist to start!**")
    st.info(" **Welcome!** This system combines **Music Mood Analysis** with **AI User Personalization**.")