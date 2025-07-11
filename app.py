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

# --- SELF-CONTAINED DATA with 20 MORE MOVIES ---
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
597,Titanic,"101-year-old Rose DeWitt Bukater tells the story of her life aboard the Titanic, 84 years later. A young Rose boards the ship with her mother and fiancé. Meanwhile, Jack Dawson and Fabrizio De Rossi win third-class tickets aboard the ship. Rose tells the whole story from Titanic's departure through to its death—on its first and last voyage—on April 15, 1912.","[{""id"": 18, ""name"": ""Drama""}, {""id"": 10749, ""name"": ""Romance""}]"
603,The Matrix,"Set in the 22nd century, The Matrix tells the story of a computer hacker who joins a group of underground insurgents fighting the vast and powerful computers who now rule the earth.","[{""id"": 28, ""name"": ""Action""}, {""id"": 878, ""name"": ""Science Fiction""}]"
559,Spider-Man,"After being bitten by a genetically altered spider, nerdy high-school student Peter Parker is endowed with amazing powers to become the Amazing superhero known as Spider-Man.","[{""id"": 14, ""name"": ""Fantasy""}, {""id"": 28, ""name"": ""Action""}]"
271110,Captain America: Civil War,"Following the events of Age of Ultron, the collective governments of the world pass an act designed to regulate all superhuman activity. This polarizes opinion amongst the Avengers, causing two factions to side with Iron Man or Captain America, which causes an epic battle between former allies.","[{""id"": 12, ""name"": ""Adventure""}, {""id"": 28, ""name"": ""Action""}, {""id"": 878, ""name"": ""Science Fiction""}]"
497,The Green Mile,"A supernatural tale set on death row in a Southern prison, where gentle giant John Coffey possesses the mysterious power to heal people's ailments. When the cell block's head guard, Paul Edgecomb, recognizes Coffey's miraculous gift, he tries desperately to help stave off the condemned man's execution.","[{""id"": 14, ""name"": ""Fantasy""}, {""id"": 18, ""name"": ""Drama""}, {""id"": 80, ""name"": ""Crime""}]"
629,The Usual Suspects,"Held in an L.A. interrogation room, Verbal Kint attempts to convince the feds that a mythic crime lord, Keyser Soze, not only exists, but was also responsible for drawing him and his four partners into a multi-million dollar heist that ended with an explosion in San Pedro harbor – leaving few survivors. Verbal lures his interrogators with an incredible story of the crime lord's almost supernatural prowess.","[{""id"": 18, ""name"": ""Drama""}, {""id"": 80, ""name"": ""Crime""}, {""id"": 53, ""name"": ""Thriller""}]"
10191,How to Train Your Dragon,"As the son of a Viking leader on the cusp of manhood, shy Hiccup Horrendous Haddock III faces a rite of passage: he must kill a dragon to prove his warrior mettle. But after downing a feared dragon, he realizes that he no longer wants to destroy it, and instead befriends the beast – which he names Toothless – much to the chagrin of his tribe.","[{""id"": 14, ""name"": ""Fantasy""}, {""id"": 12, ""name"": ""Adventure""}, {""id"": 16, ""name"": ""Animation""}, {""id"": 10751, ""name"": ""Family""}]"
105,Back to the Future,"Eighties teenager Marty McFly is accidentally sent back in time to 1955, inadvertently disrupting his parents' first meeting and attracting his mother's romantic interest. Marty must repair the damage to history by rekindling his parents' romance and - with the help of his eccentric inventor friend Doc Brown - return to 1985.","[{""id"": 12, ""name"": ""Adventure""}, {""id"": 35, ""name"": ""Comedy""}, {""id"": 878, ""name"": ""Science Fiction""}, {""id"": 10751, ""name"": ""Family""}]"
424,Schindler's List,"The true story of how businessman Oskar Schindler saved over a thousand Jewish lives from the Nazis while they worked as slaves in his factory during World War II.","[{""id"": 18, ""name"": ""Drama""}, {""id"": 36, ""name"": ""History""}, {""id"": 10752, ""name"": ""War""}]"
152532,The Grand Budapest Hotel,"The Grand Budapest Hotel tells of a legendary concierge at a famous hotel from the fictional Republic of Zubrowka between the first and second World Wars, and the lobby boy who becomes his most trusted friend.","[{""id"": 35, ""name"": ""Comedy""}, {""id"": 18, ""name"": ""Drama""}]"
2001,Vada Chennai,"A young carrom player in North Chennai becomes a reluctant participant in a war between two warring gangsters. The story chronicles his rise in the criminal underworld.","[{""id"": 28, ""name"": ""Action""}, {""id"": 80, ""name"": ""Crime""}, {""id"": 18, ""name"": ""Drama""}]"
2002,Ghilli,"A kabaddi player goes to Madurai to participate in a match, but he ends up saving a girl from a powerful and obsessive faction leader, who then chases them to Chennai.","[{""id"": 28, ""name"": ""Action""}, {""id"": 10749, ""name"": ""Romance""}, {""id"": 53, ""name"": ""Thriller""}]"
2003,Thuppakki,"An army captain on a holiday in Mumbai discovers a sleeper cell network and races against time to foil their plans to carry out a series of terrorist attacks in the city.","[{""id"": 28, ""name"": ""Action""}, {""id"": 53, ""name"": ""Thriller""}]"
2004,Mankatha,"A suspended cop hatches a plan to steal a large sum of money from a gangster. He assembles a team of four other men, but betrayal and deceit threaten to derail the entire operation.","[{""id"": 28, ""name"": ""Action""}, {""id"": 80, ""name"": ""Crime""}, {""id"": 35, ""name"": ""Comedy""}, {""id"": 53, ""name"": ""Thriller""}]"
2005,Ayan,"A young man working for a smuggling group run by his mentor gets into trouble when a new, ruthless police officer is assigned to crack down on their operations.","[{""id"": 28, ""name"": ""Action""}, {""id"": 53, ""name"": ""Thriller""}, {""id"": 80, ""name"": ""Crime""}]"
2006,Sivaji: The Boss,"A software engineer returns to India from the USA to do social work and provide free education and medical care. However, he faces numerous obstacles from a corrupt politician and businessman.","[{""id"": 28, ""name"": ""Action""}, {""id"": 35, ""name"": ""Comedy""}, {""id"": 18, ""name"": ""Drama""}]"
2007,Asuran,"A farmer from an oppressed community is forced to flee with his family after his hot-headed son kills a wealthy, upper-caste landlord. The story explores themes of land ownership, caste-based violence, and survival.","[{""id"": 28, ""name"": ""Action""}, {""id"": 18, ""name"": ""Drama""}]"
2008,Paiyaa,"A young man driving to another city gives a lift to a girl he is attracted to, only to find out that she is fleeing from a forced marriage and that gangsters are pursuing them.","[{""id"": 28, ""name"": ""Action""}, {""id"": 10749, ""name"": ""Romance""}, {""id"": 53, ""name"": ""Thriller""}]"
2009,Muthu,"A kind-hearted servant works for a zamindar (landlord) and is loved by everyone. The zamindar's uncle plots to usurp the property, leading to revelations about the servant's true identity and past.","[{""id"": 28, ""name"": ""Action""}, {""id"": 35, ""name"": ""Comedy""}, {""id"": 18, ""name"": ""Drama""}]"
2010,Thani Oruvan,"A sincere and ambitious IPS officer is on a mission to find and expose a powerful and influential scientist who is the mastermind behind a vast network of organized medical-corporate crime.","[{""id"": 28, ""name"": ""Action""}, {""id"": 53, ""name"": ""Thriller""}]"
"""

@st.cache_resource
def load_model():
    """Loads the sentence-transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def format_genres(genres_str):
    """Helper function to format the genre string into a simple list of names."""
    try:
        genres_list = json.loads(genres_str.replace("'", '"'))
        return [genre['name'] for genre in genres_list]
    except:
        return [] # Return empty list if formatting fails

@st.cache_data
def load_and_prepare_data():
    """Loads data, formats genres, and creates a combined search text."""
    movies_df = pd.read_csv(io. StringIO(MOVIE_DATA))
    movies_df['genre_list'] = movies_df['genres'].apply(format_genres)
    
    # --- THIS IS THE KEY UPGRADE ---
    # Create a 'search_text' column that combines title, genres, and overview.
    # This gives the AI rich context for matching user moods and genres.
    movies_df['search_text'] = movies_df.apply(
        lambda row: f"{row['title']}. Genres: {', '.join(row['genre_list'])}. Overview: {row['overview']}",
        axis=1
    )
    return movies_df

@st.cache_data(show_spinner="Analyzing movie database...")
def create_movie_embeddings(_model, df):
    """Creates numerical embeddings for the 'search_text'."""
    return _model.encode(df['search_text'].tolist(), convert_to_tensor=True)

# --- MAIN APP LOGIC ---

model = load_model()
movies_df = load_and_prepare_data()
movie_embeddings = create_movie_embeddings(model, movies_df)

def find_similar_movies(query, top_k=3):
    """Finds movies similar to a user's query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, movie_embeddings)
    top_k = min(top_k, len(movies_df))
    top_results = torch.topk(cosine_scores, k=top_k)
    return top_results.indices.tolist()[0], top_results.values.tolist()[0]

# --- STREAMLIT APP UI ---
st.title("🎬 AI Movie Recommender (BIA Induction Session Demo)")
st.markdown(
    """
    Tell me what you're in the mood for! Just type a **genre, a feeling, or a theme**, 
    and our AI will find the perfect movie for you.
    """
)

user_input = st.text_input(
    "What are you looking for? (e.g., 'a fun comedy', 'thriller with a good twist', 'inspirational drama')"
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
                    
                    display_genres = ', '.join(movie['genre_list'])
                    
                    with st.expander("Details"):
                        st.write(f"**Genres:** {display_genres}")
                        st.write(f"**Overview:** {movie['overview']}")
        else:
            st.warning("No recommendations found.")

    st.success("Done! We hope you enjoy your movie! 🍿")