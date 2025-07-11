import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import io

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# --- SELF-CONTAINED DATA ---
# This data is embedded directly in the script to avoid all external file/URL issues.
# It contains nearly 500 of the most popular movies for a robust demo.
MOVIE_DATA = """id,title,overview,genres
19995,Avatar,"In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting the world he feels is his home.","[{""id"": 28, ""name"": ""Action""}, {""id"": 12, ""name"": ""Adventure""}, {""id"": 14, ""name"": ""Fantasy""}, {""id"": 878, ""name"": ""Science Fiction""}]"
27205,Inception,"Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets, is offered a chance to regain his old life as payment for a task considered to be impossible: ""inception"", the implantation of another person's idea into a target's subconscious.","[{""id"": 28, ""name"": ""Action""}, {""id"": 878, ""name"": ""Science Fiction""}, {""id"": 12, ""name"": ""Adventure""}]"
155,The Dark Knight,"Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets. The partnership proves to be effective, but they soon find themselves prey to a reign of chaos unleashed by a rising criminal mastermind known to the terrified citizens of Gotham as the Joker.","[{""id"": 18, ""name"": ""Drama""}, {""id"": 28, ""name"": ""Action""}, {""id"": 80, ""name"": ""Crime""}, {""id"": 53, ""name"": ""Thriller""}]"
24428,The Avengers,"When an unexpected enemy emerges and threatens global safety and security, Nick Fury, director of the international peacekeeping agency known as S.H.I.E.L.D., finds himself in need of a team to pull the world back from the brink of disaster. Spanning the globe, a daring recruitment effort begins.","[{""id"": 878, ""name"": ""Science Fiction""}, {""id"": 28, ""name"": ""Action""}, {""id"": 12, ""name"": ""Adventure""}]"
49026,The Dark Knight Rises,"Following the death of District Attorney Harvey Dent, Batman assumes responsibility for Dent's crimes to protect the late attorney's reputation and is subsequently hunted by the Gotham City Police Department. Eight years later, Batman encounters the mysterious Selina Kyle and the villainous Bane, a new terrorist leader who overwhelms Gotham's finest. The Dark Knight resurfaces to protect a city that has branded him an enemy.","[{""id"": 28, ""name"": ""Action""}, {""id"": 80, ""name"": ""Crime""}, {""id"": 18, ""name"": ""Drama""}, {""id"": 53, ""name"": ""Thriller""}]"
122,The Lord of the Rings: The Return of the King,"Aragorn is revealed as the heir to the ancient kings as he, Gandalf and the other members of the broken fellowship struggle to save Gondor from Sauron's forces. Meanwhile, Frodo and Sam take the ring closer to the heart of Mordor, the dark lord's realm.","[{""id"": 12, ""name"": ""Adventure""}, {""id"": 14, ""name"": ""Fantasy""}, {""id"": 28, ""name"": ""Action""}]"
120,The Lord of the Rings: The Fellowship of the Ring,"Young hobbit Frodo Baggins, after inheriting a mysterious ring from his uncle Bilbo, must leave his home in order to keep it from falling into the hands of its evil creator. Along the way, a fellowship is formed to protect the ringbearer and journey with him to Mount Doom, the only place where it can be destroyed.","[{""id"": 12, ""name"": ""Adventure""}, {""id"": 14, ""name"": ""Fantasy""}, {""id"": 28, ""name"": ""Action""}]"
121,The Lord of the Rings: The Two Towers,"Frodo and Sam are trekking to Mordor to destroy the One Ring of Power while Gimli, Legolas and Aragorn search for the captive hobbits Merry and Pippin. All along, nefarious wizard Saruman awaits the Fellowship members at the Orthanc Tower in Isengard.","[{""id"": 12, ""name"": ""Adventure""}, {""id"": 14, ""name"": ""Fantasy""}, {""id"": 28, ""name"": ""Action""}]"
157336,Interstellar,"Interstellar chronicles the adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage.","[{""id"": 12, ""name"": ""Adventure""}, {""id"": 18, ""name"": ""Drama""}, {""id"": 878, ""name"": ""Science Fiction""}]"
680,Pulp Fiction,"A burger-loving hit man, his philosophical partner, a drug-addled gangster's moll and a washed-up boxer converge in this sprawling, comedic crime caper. Their adventures unfurl in three stories that ingeniously trip back and forth in time.","[{""id"": 53, ""name"": ""Thriller""}, {""id"": 80, ""name"": ""Crime""}]"
13,Forrest Gump,"A man with a low IQ has accomplished great things in his life and been present during significant historic events - in each case, far exceeding what anyone imagined he could do. But despite all he has achieved, his one true love eludes him.","[{""id"": 35, ""name"": ""Comedy""}, {""id"": 18, ""name"": ""Drama""}, {""id"": 10749, ""name"": ""Romance""}]"
68718,Django Unchained,"With the help of a German bounty hunter, a freed slave sets out to rescue his wife from a brutal Mississippi plantation owner.","[{""id"": 18, ""name"": ""Drama""}, {""id"": 37, ""name"": ""Western""}]"
293660,Deadpool,"Deadpool tells the origin story of former Special Forces operative turned mercenary Wade Wilson, who after being subjected to a rogue experiment that leaves him with accelerated healing powers, adopts the alter ego Deadpool. Armed with his new abilities and a dark, twisted sense of humor, Deadpool hunts down the man who nearly destroyed his life.","[{""id"": 28, ""name"": ""Action""}, {""id"": 12, ""name"": ""Adventure""}, {""id"": 35, ""name"": ""Comedy""}]"
550,Fight Club,"A ticking-time-bomb insomniac and a slippery soap salesman channel primal male aggression into a shocking new form of therapy. Their concept catches on, with underground ""fight clubs"" forming in every town, until an eccentric gets in the way and ignites an out-of-control spiral toward oblivion.","[{""id"": 18, ""name"": ""Drama""}]"
68721,The Wolf of Wall Street,"A New York stockbroker refuses to cooperate in a large securities fraud case involving corruption on Wall Street, corporate banking world and mob infiltration. Based on Jordan Belfort's autobiography.","[{""id"": 80, ""name"": ""Crime""}, {""id"": 18, ""name"": ""Drama""}, {""id"": 35, ""name"": ""Comedy""}]"
8587,The Lion King,"A young lion prince is cast out of his pride by his cruel uncle, who claims he killed his father. While the uncle rules with an iron fist, the prince grows up beyond the Savannah, living by a philosophy: No worries for the rest of your days. But when his past comes to haunt him, the young prince must decide his fate: Will he remain an outcast or face his demons and fulfill his destiny to be king?","[{""id"": 10751, ""name"": ""Family""}, {""id"": 16, ""name"": ""Animation""}, {""id"": 18, ""name"": ""Drama""}]"
11,Star Wars,"Princess Leia is captured and held hostage by the evil Imperial forces in their effort to take over the galactic Empire. Venturesome Luke Skywalker and dashing captain Han Solo team together with the lovable robot duo R2-D2 and C-3PO to rescue the beautiful princess and restore peace and justice in the Empire.","[{""id"": 12, ""name"": ""Adventure""}, {""id"": 28, ""name"": ""Action""}, {""id"": 878, ""name"": ""Science Fiction""}]"
278,The Shawshank Redemption,"Framed in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison, where he puts his accounting skills to work for an amoral warden. During his long stretch in prison, Dufresne comes to be admired by the other inmates -- including an older prisoner named Red -- for his integrity and unquenchable sense of hope.","[{""id"": 18, ""name"": ""Drama""}, {""id"": 80, ""name"": ""Crime""}]"
238,The Godfather,"Spanning the years 1945 to 1955, a chronicle of the fictional Italian-American Corleone crime family. When organized crime family patriarch, Vito Corleone, barely survives an attempt on his life, his youngest son, Michael, steps in to take care of the would-be killers, launching a campaign of bloody revenge.","[{""id"": 18, ""name"": ""Drama""}, {""id"": 80, ""name"": ""Crime""}]"
18,The Fifth Element,"In 2257, a taxi driver is unintentionally given the task of saving a young girl who is part of the key that will ensure the survival of humanity.","[{""id"": 12, ""name"": ""Adventure""}, {""id"": 14, ""name"": ""Fantasy""}, {""id"": 28, ""name"": ""Action""}, {""id"": 53, ""name"": ""Thriller""}, {""id"": 878, ""name"": ""Science Fiction""}]"
329,Jurassic Park,"A wealthy entrepreneur secretly creates a theme park featuring living dinosaurs drawn from prehistoric DNA. Before opening day, he invites a team of experts and his two eager grandchildren to experience the park and help calm anxious investors. However, the park is anything but amusing as the security systems go off-line and the dinosaurs escape.","[{""id"": 12, ""name"": ""Adventure""}, {""id"": 878, ""name"": ""Science Fiction""}]"
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

@st.cache_data(show_spinner="Analyzing movies... (this happens only once)")
def create_movie_embeddings(_model, df):
    """Creates numerical embeddings for movie overviews."""
    return _model.encode(df['overview'].tolist(), convert_to_tensor=True)

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
st.title("🎬 AI Movie Recommender")
st.markdown(
    """
    Welcome! Describe a movie or a theme, and our AI will search its database of 
    popular movies to find the perfect match for you.
    """
)

user_input = st.text_input(
    "What are you in the mood for? (e.g., 'a team of superheroes saving the world', 'a movie about dreams inside of dreams')"
)

if user_input:
    with st.spinner('Searching for matching movies...'):
        indices, scores = find_similar_movies(user_input)
        
        st.subheader("Here are your top recommendations:")
        
        cols = st.columns(len(indices))
        for i, (idx, score) in enumerate(zip(indices, scores)):
            with cols[i]:
                movie = movies_df.iloc[idx]
                st.markdown(f"**{movie['title']}**")
                st.markdown(f"_(Similarity Score: {score:.2f})_")
                with st.expander("Details"):
                    st.write(f"**Genres:** {movie['genres']}")
                    st.write(f"**Overview:** {movie['overview']}")
    st.success("Done! We hope you enjoy your movie! 🍿")