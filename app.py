
import os
import html
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import streamlit as st


st.set_page_config(
    page_title="CineMatch | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_API_URL = "https://api.themoviedb.org/3/movie/{}"


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #09090c;
    --panel: #15151d;
    --line: #2d2d39;
    --red: #f04455;
    --text: #f8f7fa;
    --muted: #a4a3b1;
}

.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, #35111b 0%, transparent 35%),
        radial-gradient(ellipse at 0% 45%, #1d101b 0%, transparent 40%),
        var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1320px;
    padding: 1.6rem 2rem 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.hero {
    padding: 35px 0 32px;
}

.hero-label {
    display: inline-block;
    padding: 8px 13px;
    border: 1px solid #71313c;
    border-radius: 100px;
    background: #2b1118;
    color: #ff7885;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero-title {
    margin: 22px 0 14px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(42px, 6vw, 76px);
    font-weight: 700;
    letter-spacing: -3px;
    line-height: 1.05;
    color: #ffffff;
}

.hero-title span {
    color: var(--red);
}

.hero-description {
    max-width: 570px;
    color: var(--muted);
    font-size: 16px;
    line-height: 1.8;
}

.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -.7px;
    color: #fff;
}

.section-caption {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(22, 22, 30, .88);
    border: 1px solid var(--line);
    border-radius: 18px;
}

div[data-testid="stSelectbox"] label {
    color: #e7e6ee;
    font-weight: 600;
}

div[data-testid="stSelectbox"] > div > div {
    background: #191920;
    border: 1px solid #383844;
    border-radius: 12px;
    min-height: 48px;
}

.stButton > button {
    min-height: 49px;
    border: 1px solid #ff5868;
    border-radius: 12px;
    background: linear-gradient(115deg, #ed354d, #b91f39);
    color: #ffffff;
    font-weight: 700;
    transition: transform .2s ease, filter .2s ease;
}

.stButton > button:hover {
    color: white;
    border-color: #ff7885;
    filter: brightness(1.12);
    transform: translateY(-2px);
}

div[data-testid="stImage"] img {
    width: 100%;
    aspect-ratio: 2 / 3;
    object-fit: cover;
    border: 1px solid #34313b;
    border-radius: 13px;
}

.movie-rank {
    margin: 0 0 10px;
    color: #ff6979;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.movie-name {
    margin-top: 12px;
    min-height: 50px;
    color: #f8f7fa;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 600;
    line-height: 1.5;
    overflow-wrap: anywhere;
}

.poster-placeholder {
    aspect-ratio: 2 / 3;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 12px;
    background: #191920;
    border: 1px solid #34313b;
    border-radius: 13px;
    color: #aaa8b6;
}

.footer {
    margin-top: 60px;
    padding-top: 22px;
    border-top: 1px solid #292932;
    text-align: center;
    color: #777789;
    font-size: 11px;
    letter-spacing: 2px;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1rem;
    }

    .hero-title {
        letter-spacing: -1.5px;
    }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open(MODEL_DIR / "movie_list.pkl", "rb") as file:
        movies = pickle.load(file)

    with open(MODEL_DIR / "similarity.pkl", "rb") as file:
        similarity = pickle.load(file)

    # Keep the first match when titles repeat.
    title_to_index = {}
    for index, title in enumerate(movies["title"]):
        title_to_index.setdefault(title, index)

    return movies, similarity, title_to_index


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster(movie_id):
    api_key = os.getenv("TMDB_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("TMDB_API_KEY", "")
        except Exception:
            api_key = ""

    if not api_key:
        return None

    try:
        response = requests.get(
            TMDB_API_URL.format(int(movie_id)),
            params={
                "api_key": api_key,
                "language": "en-US",
            },
            timeout=4,
        )
        response.raise_for_status()

        poster_path = response.json().get("poster_path")

        if poster_path:
            return POSTER_BASE_URL + poster_path

    except (requests.RequestException, ValueError, TypeError):
        return None

    return None


@st.cache_data(show_spinner=False)
def get_recommendations(movie_title):
    movies, similarity, title_to_index = load_model()

    index = title_to_index.get(movie_title)

    if index is None:
        return []

    scores = np.asarray(similarity[index])
    count = min(6, len(scores))

    candidates = np.argpartition(scores, -count)[-count:]
    ranked = sorted(
        candidates,
        key=lambda i: scores[i],
        reverse=True,
    )

    results = []

    for movie_index in ranked:
        if movie_index == index:
            continue

        movie = movies.iloc[int(movie_index)]

        results.append({
            "title": str(movie["title"]),
            "movie_id": int(movie["movie_id"]),
        })

        if len(results) == 5:
            break

    return results


st.markdown("""
<div class="hero">
    <div class="hero-label">✦ YOUR PERSONAL CINEMA</div>
    <div class="hero-title">
        Find your next<br>
        <span>obsession.</span>
    </div>
    <div class="hero-description">
        Great movies lead to more great movies.
        Choose a favorite and discover five films
        that match your vibe.
    </div>
</div>
""", unsafe_allow_html=True)


try:
    movies, similarity, title_to_index = load_model()
except Exception:
    st.error(
        "Unable to load the recommendation model. Please check that "
        "model/movie_list.pkl and model/similarity.pkl exist."
    )
    st.stop()


movie_list = movies["title"].dropna().tolist()

default_index = (
    movie_list.index("The Avengers")
    if "The Avengers" in movie_list
    else 0
)

with st.container(border=True):
    st.markdown(
        '<div class="section-heading">🎬 Choose your movie</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">'
        'Search the collection and choose a movie you love.'
        '</div>',
        unsafe_allow_html=True,
    )

    selected_movie = st.selectbox(
        "Search or select a movie",
        movie_list,
        index=default_index,
        label_visibility="collapsed",
        placeholder="Type a movie title...",
    )

    recommend_clicked = st.button(
        "✦  Find Similar Movies",
        use_container_width=True,
        type="primary",
    )


if recommend_clicked:
    recommendations = get_recommendations(selected_movie)

    if not recommendations:
        st.warning("No recommendations found for this movie.")

    else:
        st.markdown("")
        st.markdown(
            '<div class="section-heading">Made for your watchlist<span '
            'style="color:#f04455">.</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="section-caption">Inspired by '
            f'<b>{html.escape(selected_movie)}</b> · '
            f'Your next five picks</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        # Fetch posters together so the page loads faster.
        with st.spinner("Finding your next favorites..."):
            with ThreadPoolExecutor(max_workers=5) as executor:
                poster_urls = list(
                    executor.map(
                        lambda item: fetch_poster(item["movie_id"]),
                        recommendations,
                    )
                )

        columns = st.columns(5, gap="medium")

        for position, (column, movie, poster_url) in enumerate(
            zip(columns, recommendations, poster_urls),
            start=1,
        ):
            with column:
                st.markdown(
                    f'<div class="movie-rank">'
                    f'PICK {position:02d} / 05</div>',
                    unsafe_allow_html=True,
                )

                if poster_url:
                    st.image(
                        poster_url,
                        use_container_width=True,
                    )
                else:
                    st.markdown(
                        '<div class="poster-placeholder">'
                        '🎞️<br>Poster unavailable</div>',
                        unsafe_allow_html=True,
                    )

                # Keep the title safe for the HTML below.
                safe_title = html.escape(movie["title"])

                st.markdown(
                    f'<div class="movie-name">{safe_title}</div>',
                    unsafe_allow_html=True,
                )


st.markdown("""
<div class="footer">
    CINEMATCH &nbsp;·&nbsp; DISCOVER SOMETHING INCREDIBLE
</div>
""", unsafe_allow_html=True)
