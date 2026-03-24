import os
import requests
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ── CONFIG ──────────────────────────────────────────────────────────────────
from config import RAWG_API_KEY, RAWG_BASE

# ── RAWG HELPERS ─────────────────────────────────────────────────────────────
def rawg_get(path: str, params: dict = None) -> dict:
    params = params or {}
    params["key"] = RAWG_API_KEY
    resp = requests.get(f"{RAWG_BASE}{path}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def search_games(query: str, page_size: int = 8) -> list:
    """Autocomplete / search endpoint for the search bar."""
    data = rawg_get("/games", {"search": query, "page_size": page_size,
                               "search_precise": True})
    results = []
    for g in data.get("results", []):
        results.append({
            "id":         g["id"],
            "name":       g["name"],
            "background": g.get("background_image") or "",
            "rating":     g.get("rating", 0),
            "released":   g.get("released", ""),
            "genres":     [x["name"] for x in g.get("genres", [])],
        })
    return results

def get_game_detail(game_id: int) -> dict:
    """Fetch full game details including tags and metacritic."""
    g = rawg_get(f"/games/{game_id}")
    return {
        "id":         g["id"],
        "name":       g["name"],
        "background": g.get("background_image") or "",
        "rating":     g.get("rating", 0),
        "metacritic": g.get("metacritic") or 0,
        "playtime":   g.get("playtime") or 0,
        "released":   g.get("released", ""),
        "description": _clean_html(g.get("description_raw") or g.get("description", "")),
        "genres":     [x["name"] for x in g.get("genres", [])],
        "genre_ids":  [x["id"]   for x in g.get("genres", [])],
        "tags":       [x["id"]   for x in g.get("tags", [])[:30]],
        "tag_names":  [x["name"] for x in g.get("tags", [])[:30]],
        "platforms":  [x["platform"]["id"] for x in g.get("platforms", [])],
        "developers": [x["name"] for x in g.get("developers", [])],
    }

def _clean_html(text: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", text)[:280]

# ── CANDIDATE FETCHER ────────────────────────────────────────────────────────
def fetch_candidates(genre_ids: list, tag_ids: list, exclude_id: int, count: int = 60) -> list:
    """
    Pull games that share genres with the seed game.
    Two pages of results; exclude the seed game.
    """
    genres_str = ",".join(str(x) for x in genre_ids[:3])
    candidates = []
    for page in range(1, 3):
        try:
            data = rawg_get("/games", {
                "genres":    genres_str,
                "ordering":  "-rating",
                "page_size": 30,
                "page":      page,
            })
            for g in data.get("results", []):
                if g["id"] == exclude_id:
                    continue
                candidates.append({
                    "id":         g["id"],
                    "name":       g["name"],
                    "background": g.get("background_image") or "",
                    "rating":     g.get("rating", 0),
                    "metacritic": g.get("metacritic") or 0,
                    "playtime":   g.get("playtime") or 0,
                    "released":   g.get("released", ""),
                    "genres":     [x["name"] for x in g.get("genres", [])],
                    "genre_ids":  [x["id"]   for x in g.get("genres", [])],
                    "tags":       [x["id"]   for x in g.get("tags", [])[:30]],
                    "platforms":  [x["platform"]["id"] for x in g.get("platforms", [])],
                })
        except Exception as exc:
            logging.warning("Candidate fetch page %d failed: %s", page, exc)
    return candidates[:count]

# ── ML RECOMMENDER ───────────────────────────────────────────────────────────
def build_feature_matrix(seed: dict, candidates: list):
    """
    Content-based feature engineering:
      • Genre multi-hot (weight ×3)
      • Tag multi-hot (weight ×2) – top 30 tags from seed+candidates
      • Platform multi-hot
      • Normalised metacritic score
      • Normalised log(playtime+1)
    Returns (seed_vec, candidate_vecs, candidates).
    """
    all_games   = [seed] + candidates
    all_genres  = sorted({g for game in all_games for g in game.get("genre_ids", [])})
    all_tags    = sorted({t for game in all_games for t in game.get("tags", [])})
    all_plats   = sorted({p for game in all_games for p in game.get("platforms", [])})

    def encode(game):
        genre_vec = [3.0 if g in game.get("genre_ids", []) else 0.0 for g in all_genres]
        tag_vec   = [2.0 if t in game.get("tags",     []) else 0.0 for t in all_tags]
        plat_vec  = [1.0 if p in game.get("platforms", []) else 0.0 for p in all_plats]
        meta      = [game.get("metacritic", 0) / 100.0]
        play      = [np.log1p(game.get("playtime", 0)) / 5.0]
        return np.array(genre_vec + tag_vec + plat_vec + meta + play, dtype=np.float32)

    seed_vec  = encode(seed)
    cand_vecs = np.array([encode(c) for c in candidates])
    return seed_vec, cand_vecs

def recommend(seed: dict, candidates: list, top_n: int = 5) -> list:
    if not candidates:
        return []
    seed_vec, cand_vecs = build_feature_matrix(seed, candidates)

    # Cosine similarity between seed and every candidate
    seed_2d = seed_vec.reshape(1, -1)
    sims     = cosine_similarity(seed_2d, cand_vecs)[0]

    top_idx = np.argsort(sims)[::-1][:top_n]
    results = []
    for i in top_idx:
        c = candidates[i]
        results.append({
            "id":         c["id"],
            "name":       c["name"],
            "background": c["background"],
            "rating":     round(c["rating"], 2),
            "released":   c.get("released", ""),
            "genres":     c["genres"],
            "similarity": round(float(sims[i]) * 100, 1),
        })
    return results

# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])
    try:
        return jsonify(search_games(q))
    except Exception as exc:
        logging.error("Search error: %s", exc)
        return jsonify({"error": str(exc)}), 500

@app.route("/api/recommend/<int:game_id>")
def api_recommend(game_id: int):
    try:
        seed       = get_game_detail(game_id)
        candidates = fetch_candidates(seed["genre_ids"], seed["tags"], exclude_id=game_id)
        recs       = recommend(seed, candidates)
        return jsonify({"seed": seed, "recommendations": recs})
    except Exception as exc:
        logging.error("Recommend error: %s", exc)
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
