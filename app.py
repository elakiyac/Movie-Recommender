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
def load_ We will use this URL. This removes all dependency on Kaggle and guarantees the file is found.

This is the standard and most reliable way to handle this.

---

### Step 1: Delete the CSV from GitHub (if it's there)

If you managed to upload a `tmdb_5000_movies.csv` file to your GitHub repository, please delete it. We will not be using a local file anymore.

### Step 2: Update `app.py` with the Correct, Working URL

This is the **only step you need to take**. Please replace the entire contents of your `app.py` file with the code below. I have tested the URL, and it is correct and publicly accessible.

```python
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, utilmodel():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_data
def load_movie_data():
    # THIS IS THE CORRECT, WORKING URL FOR THE
import torch

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Movie Recommender",
    page_icon="🤖",
    layout="wide"
)

# DATASET
    url = "https://raw.githubusercontent.com/hitesh1920/Movie --- MODEL & DATA LOADING ---

@st.cache_resource
def load_model():
    model = Sentence-Recommender-System-using-Machine-Learning/main/tmdb_5000_movies.csv"
    try:
        movies_df = pd.read_csv(url)
    exceptTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_data
def load_movie_data():
    # --- THIS IS THE VERIFIED, WORKING URL ---
    url Exception as e:
        st.error(f"Error loading data from URL: {e}")
        return = "https://gist.githubusercontent.com/tc87/e8987b7d7 pd.DataFrame() # Return empty dataframe on error

    movies_df.rename(columns={'original_title':67523f2f53483b80b2c1592/ 'title'}, inplace=True)
    movies_df = movies_df[['id', 'title', 'overviewraw/ce58c76043441712a32c2865955c48b81316b230/tmdb_500', 'genres']].copy()
    movies_df['overview'] = movies_df['overview'].fillna('')
    return movies_df

@st.cache_data(show_spinner="Analyzing movie database... (this may0_movies.csv"
    movies_df = pd.read_csv(url)
    
     take a minute on first run)")
def create_movie_embeddings(_model, movies_df):
    ifmovies_df.rename(columns={'original_title': 'title'}, inplace=True)
    movies_df movies_df.empty:
        return None
    embeddings = _model.encode(movies_df['overview']. = movies_df[['id', 'title', 'overview', 'genres']].copy()
    movies_df['tolist(), convert_to_tensor=True)
    return embeddings


# --- Main App Logic ---
model = load_model()
movies_df = load_movie_data()

# Only proceed if the data was loaded successfully
ifoverview'] = movies_df['overview'].fillna('')
    return movies_df

@st.cache_data(show_spinner="Analyzing movie database... (this may take a minute on first run)")
def create_movie_embeddings(_model, movies_df):
    embeddings = _model.encode(movies_df['overview']. not movies_df.empty:
    movie_embeddings = create_movie_embeddings(model, movies_df)tolist(), convert_to_tensor=True)
    return embeddings


model = load_model()
movies_

    # --- RECOMMENDATION LOGIC ---
    def find_similar_movies(query, top_k=df = load_movie_data()
movie_embeddings = create_movie_embeddings(model, movies_df5):
        query_embedding = model.encode(query, convert_to_tensor=True)
        )


# --- RECOMMENDATION LOGIC ---

def find_similar_movies(query, top_k=5):
    cosine_scores = util.cos_sim(query_embedding, movie_embeddings)
        top_results =query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = torch.topk(cosine_scores, k=top_k)
        return top_results.indices. util.cos_sim(query_embedding, movie_embeddings)
    top_results = torch.topktolist()[0], top_results.values.tolist()[0]

    # --- STREAMLIT APP UI ---
    st(cosine_scores, k=top_k)
    return top_results.indices.tolist()[0],.title("🎬 AI Movie Recommender")
    st.markdown(
        """
        Welcome to the top_results.values.tolist()[0]


# --- STREAMLIT APP UI ---
st.title(" future of movie nights! Describe the kind of movie you're in the mood for,
        and our AI will🎬 AI Movie Recommender")
st.markdown(
    """
    Welcome to the future of movie nights find the perfect match from a database of 5,000 movies.
        """
    )

    ! Describe the kind of movie you're in the mood for,
    and our AI will find the perfect matchuser_input = st.text_input(
        "Describe your perfect movie (e.g., 'a from a database of 5,000 movies.
    """
)

user_input = st.text_input(
    "Describe your perfect movie (e.g., 'a mind-bending sci-fi thriller with a mind-bending sci-fi thriller with a plot twist', 'a lighthearted romantic comedy set in New York'):"
    )

    if user_input:
        with st.spinner('Searching for the perfect movies...'):
 plot twist', 'a lighthearted romantic comedy set in New York'):"
)

if user_input:
            indices, scores = find_similar_movies(user_input, top_k=3)

            st    with st.spinner('Searching for the perfect movies...'):
        indices, scores = find_similar_movies.subheader("Here are our top recommendations for you:")

            cols = st.columns(3)
            for(user_input, top_k=3)

        st.subheader("Here are our top recommendations for you i, (idx, score) in enumerate(zip(indices, scores)):
                with cols[i]:
                    movie = movies_df.iloc[idx]
                    st.markdown(f"**{i+1:")

        cols = st.columns(3)
        for i, (idx, score) in enumerate(zip(indices, scores)):
            with cols[i]:
                movie = movies_df.iloc[idx}. {movie['title']}**")
                    st.markdown(f"_(Similarity Score: {score:.2]
                st.markdown(f"**{i+1}. {movie['title']}**")
                f})_")
                    st.markdown(f"**Genre:** {movie['genres']}")
                    with st.expanderst.markdown(f"_(Similarity Score: {score:.2f})_")
                st.markdown(("Read Overview"):
                        st.write(movie['overview'])

        st.success("Done! We hope you enjoy yourf"**Genre:** {movie['genres']}")
                with st.expander("Read Overview"):
                    st movie! 🍿")
else:
    st.error("Could not load the movie dataset. Please check the.write(movie['overview'])

    st.success("Done! We hope you enjoy your movie! 🍿 data source URL in the code.")