from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import difflib
import requests
import re

app = Flask(__name__)
CORS(app)

TMDB_API_KEY = '0ffe789e840303831098b2e536957fd3'
DEFAULT_POSTER_URL = 'https://via.placeholder.com/500x750'

# ✅ Manual fallback posters for specific titles
manual_posters = {
    # Example entries:
    # "Movie Title in CSV": "https://link_to_poster.jpg",
    # Add your own below if you find missing ones
}

poster_cache = {}

# ------------------------------------------
# TMDb search
# ------------------------------------------
def try_tmdb_search(query: str) -> str:
    """Search TMDb for a poster."""
    try:
        r = requests.get(
            'https://api.themoviedb.org/3/search/movie',
            params={'api_key': TMDB_API_KEY, 'query': query},
            timeout=5
        )
        data = r.json()
        for result in data.get('results', []):
            poster_path = result.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"TMDb error for {query}: {e}")
    return None

# ------------------------------------------
# Poster getter with manual map & strategies
# ------------------------------------------
def get_poster_url(title: str) -> str:
    # Check manual map first
    if title in manual_posters:
        return manual_posters[title]

    if title in poster_cache:
        return poster_cache[title]

    strategies = [
        title,
        re.sub(r"\([^)]*\)", "", title).strip(),              # remove (year)
        title.split(":")[0].strip(),                          # before colon
        re.sub(r"\([^)]*\)", "", title.split(":")[0]).strip() # both
    ]

    poster_url = None
    for q in strategies:
        if not q or q.lower() in ["nan", "none"]:
            continue
        poster_url = try_tmdb_search(q)
        if poster_url:
            break

    if not poster_url:
        print(f"[WARN] No poster found for: {title}")
        poster_url = DEFAULT_POSTER_URL

    poster_cache[title] = poster_url
    return poster_url

# ------------------------------------------
# Load movie data & similarity
# ------------------------------------------
MOVIES_CSV = 'Copy of Copy of movies.csv'
SIMILARITY_MODEL = 'film_recommendation_model.pkl'

movies = pd.read_csv(MOVIES_CSV)

# ✅ Clean titles in CSV for better matching
def clean_title_for_csv(t):
    t = str(t)
    t = re.sub(r'\([^)]*\)', '', t)  # remove (year)
    t = t.strip()
    return t

movies['title'] = movies['title'].apply(clean_title_for_csv)
movies['index'] = movies['index'].astype(int)
movies = movies.set_index('index', drop=False)

similarity = joblib.load(SIMILARITY_MODEL)

# ------------------------------------------
# Routes
# ------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend')
def recommend():
    movie_name = request.args.get('movie', '')
    if not movie_name:
        return jsonify({'error': 'No movie name provided'}), 400

    all_titles = movies['title'].tolist()
    close_matches = difflib.get_close_matches(movie_name, all_titles, n=1, cutoff=0.5)
    if not close_matches:
        return jsonify({'error': 'Movie not found'}), 404

    close_match = close_matches[0]
    movie_idx = movies[movies['title'] == close_match]['index'].values[0]

    sim_scores = list(enumerate(similarity[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for idx, score in sim_scores[1:11]:
        if idx in movies.index:
            movie = movies.loc[idx]
            poster_url = get_poster_url(movie['title'])

            recommendations.append({
                'title': movie.get('title', '-'),
                'year': str(movie.get('year', '-')),
                'genre': str(movie.get('genres', movie.get('genre', '-'))),
                'rating': str(movie.get('vote_average', movie.get('rating', '-'))),
                'description': str(movie.get('overview', '-')),
                'similarity_score': f"{score * 100:.1f}",
                'poster_url': poster_url
            })

    return jsonify({'query': close_match, 'recommendations': recommendations})

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

# ------------------------------------------
# Run
# ------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
