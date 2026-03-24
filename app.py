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
def _parse_candidate(g: dict) -> dict:
    """Parse a RAWG game result dict into our candidate schema."""
    return {
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
    }

def fetch_candidates(genre_ids: list, tag_ids: list, exclude_id: int, count: int = 80) -> list:
    """
    Pull a rich candidate pool using two complementary strategies:
      1. Genre-based  — 2 pages ordered by rating (broad genre match)
      2. Tag-based    — 1 page using seed's top 4 tags (thematic/mechanic match)
    Results are deduplicated by ID. Low-quality games are filtered out.
    """
    genres_str = ",".join(str(x) for x in genre_ids[:3])
    top_tags   = ",".join(str(x) for x in tag_ids[:4])

    seen: set = set()
    candidates: list = []

    # ── Batch 1: genre-based, two pages ──────────────────────────────────────
    for page in range(1, 3):
        try:
            data = rawg_get("/games", {
                "genres":    genres_str,
                "ordering":  "-rating",
                "page_size": 30,
                "page":      page,
            })
            for g in data.get("results", []):
                if g["id"] == exclude_id or g["id"] in seen:
                    continue
                seen.add(g["id"])
                candidates.append(_parse_candidate(g))
        except Exception as exc:
            logging.warning("Genre fetch page %d failed: %s", page, exc)

    # ── Batch 2: tag-based, one page (thematic reinforcement) ─────────────────
    if top_tags:
        try:
            data = rawg_get("/games", {
                "tags":      top_tags,
                "ordering":  "-rating",
                "page_size": 30,
                "page":      1,
            })
            for g in data.get("results", []):
                if g["id"] == exclude_id or g["id"] in seen:
                    continue
                seen.add(g["id"])
                candidates.append(_parse_candidate(g))
        except Exception as exc:
            logging.warning("Tag fetch failed: %s", exc)

    # ── Quality filter: drop very low-rated games ─────────────────────────────
    candidates = [
        c for c in candidates
        if c["rating"] >= 1.5 or c["metacritic"] >= 40
    ]

    return candidates[:count]

# ── ML RECOMMENDER ───────────────────────────────────────────────────────────
def build_feature_matrix(seed: dict, candidates: list):
    """
    Seed-overlap-aware content-based feature engineering.

    Weight rationale
    ─────────────────────────────────────────────────────────────────────────
    • Primary genre (×14): The single most important signal. A visual-novel
      must recommend visual-novels; an RPG must recommend RPGs.
    • Other genres  (×10): Strong genre overlap still matters a lot.
    • Seed-matched tags (×6): Tags the seed game has are the game's "DNA"
      (mechanics, themes, mood). A candidate sharing those tags is a very
      strong thematic match.
    • Other tags (×0.8): Tags present in a candidate but absent from the seed
      are near-noise — we keep a tiny weight to avoid zero-vectors but they
      should not inflate similarity.
    • Platforms (×0.2): Cross-platform era; irrelevant for theme/genre match.
    • Metacritic / Playtime: Very low weights — quality is a secondary signal
      and must not override genre/theme alignment.
    ─────────────────────────────────────────────────────────────────────────
    """
    all_games      = [seed] + candidates
    all_genres     = sorted({g for game in all_games for g in game.get("genre_ids", [])})
    all_tags       = sorted({t for game in all_games for t in game.get("tags", [])})
    all_plats      = sorted({p for game in all_games for p in game.get("platforms", [])})

    seed_tag_set   = set(seed.get("tags", []))
    seed_genres    = seed.get("genre_ids", [])
    primary_genre  = seed_genres[0] if seed_genres else None

    def encode(game: dict) -> np.ndarray:
        game_genre_set = set(game.get("genre_ids", []))
        game_tag_set   = set(game.get("tags", []))
        game_plat_set  = set(game.get("platforms", []))

        # Genre vector — primary genre gets extra weight
        genre_vec = [
            (14.0 if g == primary_genre else 10.0) if g in game_genre_set else 0.0
            for g in all_genres
        ]

        # Tag vector — seed-overlap aware
        tag_vec = [
            (6.0 if t in seed_tag_set else 0.8) if t in game_tag_set else 0.0
            for t in all_tags
        ]

        # Platform vector — minimal contribution
        plat_vec = [0.2 if p in game_plat_set else 0.0 for p in all_plats]

        # Quality scalars — kept low so they don't override theme alignment
        meta = [game.get("metacritic", 0) / 100.0 * 0.4]
        play = [np.log1p(game.get("playtime", 0)) / 5.0 * 0.2]

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
