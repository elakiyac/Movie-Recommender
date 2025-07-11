import streamlit as st
from transformers import pipeline

# --- MODEL LOADING ---
# We are using a smaller, more efficient model that fits in memory.
@st.cache_resource
def load_classifier():
    return pipeline('zero-shot-classification', model="MoritzLaurer/deberta-v3-small-zeroshot-v1")

classifier = load_classifier()


# --- APP UI ---
st.title("🤖 AI Movie Recommender")
st.markdown(
    """
    Welcome! This demo uses a Zero-Shot Classification model to recommend a movie genre 
    based on how you're feeling or what you want to watch. 
    Describe a theme, a mood, or even another movie.
    """
)

# A more diverse list of genres/themes
genres = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "Historical", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western", "Superhero", "Mind-bending Plot Twist"
]

user_input = st.text_input(
    "Describe your mood or what you want to watch (e.g., 'a funny movie with friends', 'something with spaceships and aliens'):"
)

if user_input:
    with st.spinner('Analyzing your request...'):
        # Get the model's predictions
        # Set multi_label to True for better results with multiple potential genres
        recommendations = classifier(user_input, genres, multi_label=True)

        st.subheader("Here are our top recommendations for you:")

        # Display the top 3 recommendations
        top_3_labels = recommendations['labels'][:3]
        top_3_scores = recommendations['scores'][:3]

        for i in range(3):
            st.write(f"**{i+1}. {top_3_labels[i]}** (Confidence: {top_3_scores[i]:.2%})")

    st.success("Done! Enjoy your movie night! 🍿")