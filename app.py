import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background:
      radial-gradient(circle at 10% 0%, rgba(111,66,193,.22), transparent 28%),
      radial-gradient(circle at 90% 8%, rgba(20,184,166,.13), transparent 25%),
      #0b1020;
    color: #f7f7fb;
}
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1,h2,h3 { font-family: 'Outfit', sans-serif !important; }
.hero {
    padding: 30px 32px; border: 1px solid rgba(255,255,255,.10);
    border-radius: 24px; background: linear-gradient(120deg, rgba(97,70,180,.30), rgba(20,184,166,.12));
    margin-bottom: 24px;
}
.eyebrow { color:#a9a1ff; font-size:12px; letter-spacing:2px; font-weight:700; }
.hero h1 { font-size: clamp(34px, 5vw, 54px); margin: 8px 0 6px; color:#fff; }
.hero p { color:#c5c9d8; font-size:16px; max-width:680px; }
.movie-card {
    background: linear-gradient(160deg, #202943, #141b2d);
    border: 1px solid rgba(255,255,255,.09); border-radius:18px;
    overflow:hidden; min-height:245px; margin-bottom:14px;
}
.poster {
    height:118px; display:flex; align-items:center; justify-content:center;
    font-size:46px; background:linear-gradient(135deg, var(--c1), var(--c2));
}
.movie-info { padding:14px 15px 16px; }
.movie-title { font-family:'Outfit'; font-size:17px; font-weight:700; color:#fff; margin-bottom:5px; }
.movie-meta { color:#aeb6ca; font-size:12px; }
.movie-desc { color:#cbd1df; font-size:12px; margin-top:9px; line-height:1.45; }
.pill {
    display:inline-block; padding:5px 10px; border-radius:30px;
    background:rgba(167,139,250,.15); color:#c9baff; font-size:11px; margin:3px 4px 0 0;
}
div[data-testid="stSelectbox"] > label, div[data-testid="stTextInput"] > label { color:#dfe4f2; font-weight:600; }
.stButton > button {
    background:linear-gradient(90deg,#8b5cf6,#5b8def); color:white; border:0;
    border-radius:12px; font-weight:700; padding:.65rem 1.2rem; width:100%;
}
.stButton > button:hover { border:0; color:white; filter:brightness(1.12); }
section[data-testid="stSidebar"] { background:#11182a; border-right:1px solid rgba(255,255,255,.08); }
.metric-box { background:#151e32; border:1px solid rgba(255,255,255,.08); border-radius:15px; padding:16px; }
.small-muted { color:#9ba6bd; font-size:13px; }
hr { border-color:rgba(255,255,255,.10); }
</style>
""", unsafe_allow_html=True)

EMOJIS = {
    "Action":"⚡", "Adventure":"🧭", "Animation":"✨", "Comedy":"😂",
    "Drama":"🎭", "Fantasy":"🪄", "Science Fiction":"🚀", "Sci-Fi":"🚀",
    "Thriller":"🕵️", "Romance":"💗", "Crime":"🔎", "Family":"👨‍👩‍👧",
    "Mystery":"🧩", "Horror":"🌙"
}
COLORS = [
    ("#6d4aff","#1d9bf0"),("#e15b75","#6c42b8"),("#0f9b8e","#1b4965"),
    ("#e09f3e","#9e2a2b"),("#536dfe","#7c4dff"),("#b33951","#4b134f")
]

@st.cache_data
def load_movies():
    path = Path(__file__).parent / "movies.csv"
    df = pd.read_csv(path)
    df["combined"] = (df["genres"].fillna("") + " " + df["overview"].fillna("") + " " + df["keywords"].fillna(""))
    return df

@st.cache_resource
def build_model(text_data):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    matrix = vectorizer.fit_transform(text_data)
    similarity = cosine_similarity(matrix)
    return vectorizer, similarity

movies = load_movies()
_, similarity = build_model(tuple(movies["combined"].tolist()))

def recommend(title, count=5):
    idx = movies.index[movies["title"] == title][0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    results = []
    for movie_idx, score in scores:
        if movie_idx != idx:
            results.append((movies.iloc[movie_idx], float(score)))
        if len(results) == count:
            break
    return results

def movie_card(movie, score=None, index=0):
    genre_list = str(movie["genres"]).split("|")
    emoji = EMOJIS.get(genre_list[0], "🎬")
    c1, c2 = COLORS[index % len(COLORS)]
    score_text = f"<span class='pill'>Match score: {score*100:.0f}%</span>" if score is not None else ""
    tags = "".join(f"<span class='pill'>{g}</span>" for g in genre_list[:2])
    st.markdown(f"""
    <div class="movie-card">
      <div class="poster" style="--c1:{c1};--c2:{c2}">{emoji}</div>
      <div class="movie-info">
        <div class="movie-title">{movie['title']}</div>
        <div class="movie-meta">{int(movie['year'])} &nbsp;•&nbsp; {movie['rating']}/10</div>
        {tags}{score_text}
        <div class="movie-desc">{movie['overview']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("Your personal movie discovery assistant")
    st.markdown("---")
    st.markdown("### About this project")
    st.write("CineMatch uses **TF-IDF** and **Cosine Similarity** to find movies with similar genres, descriptions and keywords.")
    st.markdown("---")
    st.markdown("**Project type**")
    st.caption("Content-Based Filtering • Machine Learning")
    st.markdown("**Built with**")
    st.caption("Python • Pandas • Scikit-learn • Streamlit")

# ---------- Main page ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">DISCOVER YOUR NEXT FAVOURITE</div>
  <h1>Find movies you'll love.</h1>
  <p>Choose a movie you already enjoy. CineMatch analyses its genres, story and keywords to recommend similar films.</p>
</div>
""", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
for col, number, label in [
    (col_a, f"{len(movies)}", "MOVIES IN THIS DEMO DATASET"),
    (col_b, "TF-IDF", "TEXT FEATURE EXTRACTION"),
    (col_c, "COSINE", "SIMILARITY METHOD")
]:
    with col:
        st.markdown(f"<div class='metric-box'><div style='font-size:25px;font-weight:800;color:#fff'>{number}</div><div class='small-muted'>{label}</div></div>", unsafe_allow_html=True)

st.markdown("### 🔎 Choose a movie")
left, right = st.columns([3,1])
with left:
    selected = st.selectbox("Select a movie", movies["title"].tolist(), label_visibility="collapsed")
with right:
    count = st.selectbox("Recommendations", [5, 6, 8], index=0, label_visibility="collapsed")

selected_movie = movies[movies["title"] == selected].iloc[0]
st.markdown("#### Your selected movie")
movie_card(selected_movie, index=0)

if st.button("✨ Recommend similar movies"):
    results = recommend(selected, count)
    st.markdown("---")
    st.markdown(f"### 🎯 Because you liked {selected}")
    st.caption("Ranked by content similarity. A higher match score means the movie description is more similar to your selection.")
    cols = st.columns(3)
    for i, (movie, score) in enumerate(results):
        with cols[i % 3]:
            movie_card(movie, score, i+1)
else:
    st.info("Select a movie above and press **Recommend similar movies** to see your results.")

with st.expander("📘 How does the ML model work?"):
    st.markdown("""
    1. **Dataset:** Each movie has a title, genres, overview and keywords.
    2. **Text preparation:** These fields are combined into one text description.
    3. **TF-IDF:** Converts movie text into numerical feature vectors. Words that are more useful for distinguishing a movie receive more weight.
    4. **Cosine Similarity:** Compares the vectors and calculates how similar two movies are.
    5. **Recommendation:** The system sorts movies by similarity and displays the closest matches.

    This is a **content-based recommendation system**. It recommends films based on their content, not on users' viewing histories.
    """)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#8792aa;font-size:12px'>CineMatch • AI & ML Mini Project • Educational demonstration</div>", unsafe_allow_html=True)
