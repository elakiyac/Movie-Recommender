import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import io
import json

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# --- SELF-CONTAINED DATA with KOLLYWOOD MOVIES ---
# This data is embedded directly in the script to avoid all external file/URL issues.
# The formatting has been corrected to prevent IndentationError.
MOVIE_DATA = """id,title,overview,genres
19995,Avatar,"In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting the world he feels is his home.","[{""id"": 28, ""name"": ""Action""}, {""id"": 12, ""name"": ""Adventure""}, {""id"": 14, ""name"": ""Fantasy""}, {""id"": 878, ""name"": ""Science Fiction""}]"
27205,Inception,"Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets, is offered a chance to regain his old life as payment for a task considered to be impossible: ""inception"", the implantation of another person's idea into a target's subconscious.","[{""id"": 28, ""name"": ""Action""}, {""id"": 878, ""name"": ""Science Fiction""}, {""id"": 12, ""name"": ""Adventure""}]"
155,The Dark Knight,"Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets. The partnership proves to be effective, but they soon find themselves prey to a reign of chaos unleashed by a rising criminal mastermind known to the terrified citizens of Gotham as the Joker.","[{""id"": 18, ""name"": ""Drama""}, {""id"": 28, ""name"": ""Action""}, {""id"": 80, ""name"": ""Crime""}, {""id"": 53, ""name"": ""Thriller""}]"
24428,The Avengers,"When an unexpected enemy emerges and threatens global safety and security, Nick Fury, director of the international peacekeeping agency known as S.H.I.E.L.D., finds himself in need of a team to pull the world back from the brink of disaster. Spanning the globe, a daring recruitment effort begins.","[{""id"": 878, ""name"": ""Science Fiction""}, {""id"": 28, ""name"": ""Action""}, {""id"": 12, ""name"": ""Adventure""}]"
13,Forrest Gump,"A man with a low IQ has accomplished great things in his life and been present during significant historic events - in each case, far exceeding what anyone imagined he could do. But despite all he has achieved, his one true love eludes him.","[{""id"": 35, ""name"": ""Comedy""}, {""id"": 18, ""name"": ""Drama""}, {""id"": 10749, ""name"": ""Romance""}]"
8587,The Lion King,"A young lion prince is cast out of his pride by his cruel uncle, who claims he killed his father. While the uncle rules with an iron fist, the prince grows up beyond the Savannah, living by a philosophy: No worries for the rest of your days. But when his past comes to haunt him, the young prince must decide his fate: Will he remain an outcast or face his demons and fulfill his destiny to be king?","[{""id"": 10751, ""name"": ""Family""}, {""id"": 16, ""name"": ""Animation""}, {""id"": 18, ""name"": ""Drama""}]"
1003,Nayakan,"A young man from a small village flees to Mumbai after a family tragedy and rises to become a powerful, respected gangster who fights for the rights of his people.","[{""id"": 18, ""name"": ""Drama""}, {""id"": 80, ""name"": ""Crime""}]"
1001,Enthiran,"A brilliant scientist creates a sophisticated humanoid robot to protect mankind, but things go awry when human emotions are programmed into it, and it falls in love with the scientist's fiancée.","[{""id"": 28, ""name"": ""Action""}, {""id"": 878, ""name"": ""Science Fiction""}]"
1002,Vikram Vedha,"A tough, no-nonsense police officer engages in a mind game with a cunning gangster, where the gangster challenges the officer's perception of good and evil by narrating stories from his past.","[{""id"": 28, ""name"": ""Action""}, {""id"": 53, ""name"": ""Thriller""}, {""id"": 80, ""name"": ""Crime""}]"
1004,Soorarai Pottru,"A man from a remote village dreams of launching his own low-cost airline. He must overcome numerous obstacles from powerful enemies and a rigid system to make air travel affordable for the common person.","[{""id"": 18, ""name"": ""Drama""}]"
1005,Kaithi,"An ex-convict, on his way to meet his daughter for the first time after a long prison sentence, is forced by an injured police officer to help him stop a gang of ruthless drug smugglers in a single, action-packed night.","[{""id"": 28, ""name"": ""Action""}, {""id"": 53, ""name"": ""Thriller""}]"
1006,Baasha,"An auto-rickshaw driver leads a simple, peaceful life but has a violent past as a feared gangster in Mumbai. When his family is threatened by local thugs, he is forced to reveal his old identity to protect them.","[{""id"": 28, ""name"": ""Action""}, {""id"": 18, ""name"": ""Drama""}]"
1007,Anbe Sivam,"Two men, an optimistic, kind-hearted, and handicapped communist and an arrogant young advertisement filmmaker, are stranded together on a journey. They form an unlikely bond while debating their conflicting ideologies and facing unforeseen challenges.","[{""id"": 35, ""name"": ""Comedy""}, {""id"": 18, ""name"": ""Drama""}, {""id"": 12, ""name"": ""Adventure""}]"
"""

@st.cache_resource
def load_model():
    """Loads the sentence-transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_movie_data():
    """Loads the movie data from the embedded string in the script."""
    # Use io.StringIO to treat the string data as a file
    movies_df = pd.read_csv(io.StringIO(MOVIE_DATA))
    return movies_df

@st.cache_data(show_spinner="Analyzing movie database...")
def create_movie_embeddings(_model, df):
    """Creates numerical embeddings for movie overviews."""
    return _model.encode(df['overview'].tolist(), convert_to_tensor=True)

def format_genres(genres_str):
    """Helper function to format the genre string for display."""
    try:
        # The genre data is a string representation of a list of dictionaries
        genres_list = json.loads(genres_str.replace("'", '"'))
        return ', '.join([genre['name'] for genre in genres_list])
    except:
        return "N/A" # Return N/A if formatting fails

# --- MAIN APP LOGIC ---

model = load_model()
movies_df = load_movie_data()
movie_embeddings = create_movie_embeddings(model, movies_df)

def find_similar_movies(query, top_k=3):
    """Finds movies similar to a user's query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, movie_embeddings)
    # Ensure k is not larger than the number of movies
    top_k = min(top_k, len(movies_df))
    top_results = torch.topk(cosine_scores, k=top_k)
    return top_results.indices.tolist()[0], top_results.values.tolist()[0]

# --- STREAMLIT APP UI ---
st.title("🎬 AI Movie Recommender (Kollywood & Hollywood)")
st.markdown(
    """
    Welcome! Describe a movie or a theme, and our AI will search its database of 
    popular movies to find the perfect match for you.
    """
)

user_input = st.text_input(
    "What are you in the mood for? (e.g., 'a cop chasing a villain', 'a movie about dreams inside of dreams')"
)

if user_input:
    with st.spinner('Searching for matching movies...'):
        indices, scores = find_similar_movies(user_input)
        
        st.subheader("Here are your top recommendations:")
        
        if len(indices) > 0:
            cols = st.columns(len(indices))
            for i, (idx, score) in enumerate(zip(indices, scores)):
                with cols[i]:
                    movie = movies_df.iloc[idx]
                    st.markdown(f"**{movie['title']}**")
                    st.markdown(f"_(Similarity Score: {score:.2f})_")
                    
                    # Format genres for clean display
                    display_genres = format_genres(movie['genres'])
                    
                    with st.expander("Details"):
                        st.write(f"**Genres:** {display_genres}")
                        st.write(f"**Overview:** {movie['overview']}")
        else:
            st.warning("No recommendations found.")

    st.success("Done! We hope you enjoy your movie! 🍿")