from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ── 1. Load CSVs ──────────────────────────────────────────────
movies  = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
tags    = pd.read_csv("tags.csv")

# ── 2. Build tags column ──────────────────────────────────────
tags = tags.groupby('movieId')['tag'].apply(lambda x: " ".join(x)).reset_index()
movies = movies.merge(tags, on='movieId', how='left')
movies['tag']    = movies['tag'].fillna('')
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
movies['tag']    = movies['tag'].apply(lambda x: str(x).split())
movies['tags']   = movies['genres'] + movies['tag']

new_df = movies[['movieId', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# ── 3. Simple stemmer (pure Python, no nltk needed) ───────────
def simple_stem(word):
    """Basic suffix stripping — good enough for genre/tag words."""
    suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 'ers', 'er', 's']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word

def stem(text):
    return " ".join(simple_stem(w) for w in text.split())

new_df['tags'] = new_df['tags'].apply(stem)

# ── 4. Content-based similarity ───────────────────────────────
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
content_similarity = cosine_similarity(vectors)

# ── 5. Collaborative similarity ───────────────────────────────
movie_ratings     = ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
collab_sim_matrix = cosine_similarity(movie_ratings)
collab_similarity = pd.DataFrame(collab_sim_matrix,
                                  index=movie_ratings.index,
                                  columns=movie_ratings.index)

print("✅ Models loaded successfully")

# ── 6. Recommend function ─────────────────────────────────────
def recommend(movie):
    movie = movie.lower().strip()

    match = new_df[new_df['title'].str.lower() == movie]
    if match.empty:
        return None

    movie_index = match.index[0]
    movie_id    = new_df.iloc[movie_index].movieId

    # Content scores
    content_scores = list(enumerate(content_similarity[movie_index]))
    content_vals   = normalize([s[1] for s in content_scores])

    # Collaborative scores
    collab_sim = collab_similarity[movie_id] if movie_id in collab_similarity.columns else None
    collab_scores = []
    for i, row in new_df.iterrows():
        mid = row.movieId
        if collab_sim is not None and mid in collab_sim.index:
            collab_scores.append(float(collab_sim.loc[mid]))
        else:
            collab_scores.append(0)
    collab_vals = normalize(collab_scores)

    # Weighted hybrid
    hybrid_scores = [
        (i, (content_vals[i] * CONTENT_WEIGHT + collab_vals[i] * COLLAB_WEIGHT))
        for i in range(len(content_scores))
    ]
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[1:20]

    # Filter unpopular movies
    result = []
    for i, score in hybrid_scores:
        mid = new_df.iloc[i].movieId
        if mid in popular_movies:
            result.append(new_df.iloc[i].title)
        if len(result) == 10:
            break

    return result if result else None


# ── 7. Routes ─────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend")
def get_recommendations():
    movie = request.args.get("movie", "").strip()
    if not movie:
        return jsonify({"error": "Please provide a movie name"}), 400

    result = recommend(movie)
    if result is None:
        return jsonify({"error": f"Movie '{movie}' not found in database"}), 404

    return jsonify({"recommendations": result})


@app.route("/search")
def search_movies():
    query = request.args.get("q", "").strip().lower()
    if not query:
        return jsonify([])
    matches = new_df[new_df['title'].str.lower().str.contains(query, na=False)]['title'].head(10).tolist()
    return jsonify(matches)


if __name__ == "__main__":
    app.run(debug=True)
