import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Movie Recommender",
    page_icon="🤖",
    layout="wide"
)

# --- MODEL & DATA LOADING ---

@st.cache_resource
def load_model():
    # This is a small, fast, and effective model for semantic search.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_data
def load_movie_data():
    # --- THIS IS THE ONLY CHANGE ---
    # Load the movie dataset directly from a reliable URL
    url = "https://gist.githubusercontent.com/elakiyac/5053c076ade81b7538c82662a265b706/raw/6659929836955e1b30589b33a763836d52575459/tmdb_5000_movies.csv"
    movies_df = pd.read_csv(url)
    
    # Clean up the data a bit
    movies_df.rename(columns={'original_title': 'title'}, inplace=True)
    movies_df = movies_df[['id', 'title', 'overview', 'genres']].copy()
    movies_df['overview'] = movies_df['overview'].fillna('')
    return movies_df

@st.cache_data(show_spinner="Analyzing movie database... (this may take a minute on first run)")
def create_movie_embeddings(_model, movies_df):
    # The model converts movie overviews into numerical vectors (embeddings)
    embeddings = _model.encode(movies_df['overview'].tolist(), convert_to_tensor=True)
    return embeddings


model = load_model()
movies_df = load_movie_data()
movie_embeddings = create_movie_embeddings(model, movies_df)


# --- RECOMMENDATION LOGIC ---

def find_similar_movies(query, top_k=5):
    # 1. Encode the user's query into an embedding
    query_embedding = model.encode(query, convert_to_tensor=True)

    # 2. Compute cosine similarity
    cosine_scores = util.cos_sim(query_embedding, movie_embeddings)

    # 3. Find the top_k most similar movies
    top_results = torch.topk(cosine_scores, k=top_k)

    return top_results.indices.tolist()[0], top_results.values.tolist()[0]


# --- STREAMLIT APP UI ---
st.title("🎬 AI Movie Recommender")
st.markdown(
    """
    Welcome to the future of movie nights! Describe the kind of movie you're in the mood for,
    and our AI will find the perfect match from a database of 5,000 movies.
    """
)

user_input = st.text_input(
    "Describe your perfect movie (e.g., 'a mind-bending sci-fi thriller with a plot twist', 'a lighthearted romantic comedy set in New York'):"
)

if user_input:
    with st.spinner('Searching for the perfect movies...'):
        indices, scores = find_similar_movies(user_input, top_k=3)

        st.subheader("Here are our top recommendations for you:")

        cols = st.columns(3)
        for i, (idx, score) in enumerate(zip(indices, scores)):
            with cols[i]:
                movie = movies_df.iloc[idx]
                st.markdown(f"**{i+1}. {movie['title']}**")
                st.markdown(f"_(Similarity Score: {score:.2f})_")
                st.markdown(f"**Genre:** {movie['genres']}")
                with st.expander("Read Overview"):
                    st.write(movie['overview'])

    st.success("Done! We hope you enjoy your movie! 🍿")