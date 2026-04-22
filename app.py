import os
import requests
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime, timedelta

# Configure Flask to look for templates in the current directory
app = Flask(__name__, template_folder='.')
logging.basicConfig(level=logging.INFO)

# ── CONFIG ──────────────────────────────────────────────────────────────────
from config import STEAM_API_KEY, STEAM_BASE, STEAM_STORE_BASE

# ── STEAM HELPERS ────────────────────────────────────────────────────────────
STEAM_APP_CACHE = {}
CACHE_DURATION = timedelta(hours=24)

def steam_store_get(path: str, params: dict = None) -> dict:
    """Fetch from Steam Store API (public, no key required)."""
    params = params or {}
    resp = requests.get(f"{STEAM_STORE_BASE}{path}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def steam_web_get(interface: str, method: str, params: dict = None) -> dict:
    """Fetch from Steam Web API (requires API key for some endpoints)."""
    params = params or {}
    params["key"] = STEAM_API_KEY
    url = f"{STEAM_BASE}/{interface}/{method}/v1"
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def search_games(query: str, page_size: int = 8) -> list:
    """
    Search for games by fetching popular games and filtering by name.
    SteamSpy doesn't have a search API, so we pool top games and filter locally.
    """
    try:
        query_lower = query.lower().strip()
        results = []
        seen = set()
        
        # ── Batch 1: Top 100 games from SteamSpy ──────────────────────────────
        try:
            spy_resp = requests.get(
                "https://www.steamspy.com/api.php",
                params={"request": "top100forever"},
                timeout=10
            ).json()
            
            for app_id_str in spy_resp.keys():
                if app_id_str == "error" or app_id_str in seen:
                    continue
                try:
                    app_id = int(app_id_str)
                    if app_id in seen or len(results) >= page_size:
                        continue
                    
                    game_data = get_game_detail(app_id)
                    if game_data is None:  # Skip games that failed to load
                        continue
                    
                    game_name = game_data.get("name", "").lower()
                    
                    # Simple substring match
                    if query_lower in game_name:
                        results.append({
                            "id":         game_data["id"],
                            "name":       game_data["name"],
                            "background": game_data.get("background", ""),
                            "rating":     game_data.get("rating", 0),
                            "released":   game_data.get("released", ""),
                            "genres":     game_data.get("genres", []),
                        })
                        seen.add(app_id)
                except Exception as e:
                    logging.debug(f"Failed to process game {app_id_str}: {e}")
                    continue
        except Exception as e:
            logging.warning(f"SteamSpy top100 fetch failed: {e}")
        
        # ── Batch 2: Featured games from Steam Store ──────────────────────────
        if len(results) < page_size:
            try:
                featured = steam_store_get("/featured", {})
                for game in featured.get("featured_win", [])[:50]:
                    if len(results) >= page_size:
                        break
                    
                    app_id = game.get("id")
                    if app_id in seen:
                        continue
                    
                    try:
                        game_data = get_game_detail(app_id)
                        if game_data is None:  # Skip games that failed to load
                            continue
                        
                        game_name = game_data.get("name", "").lower()
                        
                        if query_lower in game_name:
                            results.append({
                                "id":         game_data["id"],
                                "name":       game_data["name"],
                                "background": game_data.get("background", ""),
                                "rating":     game_data.get("rating", 0),
                                "released":   game_data.get("released", ""),
                                "genres":     game_data.get("genres", []),
                            })
                            seen.add(app_id)
                    except Exception as e:
                        logging.debug(f"Failed to process featured game {app_id}: {e}")
                        continue
            except Exception as e:
                logging.warning(f"Featured games fetch failed: {e}")
        
        return results[:page_size]
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return []

def get_game_detail(app_id: int) -> dict:
    """Fetch full game details from Steam Store API."""
    try:
        store_data = steam_store_get(f"/appdetails", {"appids": app_id})
        
        if not store_data.get(str(app_id), {}).get("success"):
            return None  # Game not found, return None instead of raising
        
        app_data = store_data[str(app_id)]["data"]
        
        # Extract categories (genres in Steam)
        categories = [cat["description"] for cat in app_data.get("categories", [])]
        genre_list = [g["description"] for g in app_data.get("genres", [])]
        
        # Get tags from SteamSpy if available
        try:
            spy_data = requests.get(
                f"https://www.steamspy.com/api.php",
                params={"request": "appdetails", "appid": app_id},
                timeout=5
            ).json()
            user_score = spy_data.get("score_rank", 0)
            tags = list(spy_data.get("tags", {}).keys())[:30]
        except:
            user_score = 0
            tags = []
        
        release_date = app_data.get("release_date", {}).get("date", "")
        
        return {
            "id":          app_id,
            "name":        app_data.get("name", ""),
            "background":  app_data.get("header_image", ""),
            "rating":      user_score,
            "released":    release_date,
            "description": _clean_html(app_data.get("short_description", ""))[:280],
            "genres":      genre_list,
            "genre_ids":   [i for i in range(len(genre_list))],  # Use indices as IDs
            "tags":        tags,
            "tag_names":   tags,
            "categories":  categories,
            "platforms":   _get_platforms(app_data),
            "developers":  [d for d in app_data.get("developers", [])],
            "price":       app_data.get("price_overview", {}).get("final", 0) / 100,
        }
    except Exception as e:
        logging.warning(f"Error fetching game details for {app_id}: {e}")
        return None  # Return None on error, don't raise

def _get_platforms(app_data: dict) -> list:
    """Extract platform info from Steam app data."""
    platforms = []
    if app_data.get("platforms", {}).get("windows"):
        platforms.append("Windows")
    if app_data.get("platforms", {}).get("mac"):
        platforms.append("macOS")
    if app_data.get("platforms", {}).get("linux"):
        platforms.append("Linux")
    return platforms

def _clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    import re
    return re.sub(r"<[^>]+>", "", text)

# ── CANDIDATE FETCHER ───────────────────────────────────────────────────────
def _parse_candidate(app_data: dict) -> dict:
    """Parse a Steam game dict into our candidate schema."""
    try:
        return {
            "id":         app_data["id"],
            "name":       app_data["name"],
            "background": app_data.get("background", ""),
            "rating":     app_data.get("rating", 0),
            "released":   app_data.get("released", ""),
            "genres":     app_data.get("genres", []),
            "genre_ids":  app_data.get("genre_ids", []),
            "tags":       app_data.get("tags", []),
            "categories": app_data.get("categories", []),
        }
    except Exception as e:
        logging.error(f"Error parsing candidate: {e}")
        return {}

def fetch_candidates(genres: list, tags: list, exclude_id: int, count: int = 80) -> list:
    """
    Fetch candidate games from Steam using multiple strategies:
      1. Genre-based — Pull popular games in similar genres
      2. Tag-based — Pull games with similar tags/categories
    
    Since Steam API doesn't have rich filtering, we use SteamSpy's popularity data
    and filter by genres/tags locally.
    """
    candidates = []
    seen = {exclude_id}
    
    try:
        # Get popular games from SteamSpy and filter by genre/tag overlap
        spy_resp = requests.get(
            "https://www.steamspy.com/api.php",
            params={"request": "top100forever"},
            timeout=10
        ).json()
        
        for app_id_str, spy_data in spy_resp.items():
            if app_id_str == "error":
                continue
            
            try:
                app_id = int(app_id_str)
                if app_id in seen or app_id == exclude_id:
                    continue
                
                game_detail = get_game_detail(app_id)
                if game_detail is None:  # Skip failed loads
                    continue
                
                game_genres = game_detail.get("genres", [])
                game_tags = game_detail.get("tags", [])
                
                # Check for genre or tag overlap
                genre_overlap = any(g in game_genres for g in genres)
                tag_overlap = any(t in game_tags for t in tags)
                
                if genre_overlap or tag_overlap or not genres:  # Include if any overlap or no genre filter
                    candidate = _parse_candidate(game_detail)
                    if candidate:
                        candidates.append(candidate)
                        seen.add(app_id)
                    
                    if len(candidates) >= count:
                        break
            except Exception as e:
                logging.warning(f"Failed to load candidate {app_id_str}: {e}")
                continue
        
        # If we don't have enough, fetch more from featured games
        if len(candidates) < count // 2:
            try:
                featured = steam_store_get("/featured", {})
                for game in featured.get("featured_win", [])[:30]:
                    app_id = game.get("id")
                    if app_id in seen or app_id == exclude_id:
                        continue
                    try:
                        game_detail = get_game_detail(app_id)
                        if game_detail is None:  # Skip failed loads
                            continue
                        candidate = _parse_candidate(game_detail)
                        if candidate:
                            candidates.append(candidate)
                            seen.add(app_id)
                        if len(candidates) >= count:
                            break
                    except Exception as e:
                        logging.warning(f"Failed to load featured game {app_id}: {e}")
                        continue
            except Exception as e:
                logging.warning(f"Featured games fetch failed: {e}")
        
        return candidates[:count]
    except Exception as e:
        logging.error(f"Candidate fetch failed: {e}")
        return []

# ── ML RECOMMENDER ──────────────────────────────────────────────────────────
def build_feature_matrix(seed: dict, candidates: list):
    """
    Seed-overlap-aware content-based feature engineering for Steam games.

    Weight rationale
    ─────────────────────────────────────────────────────────────────────────
    • Primary genre (×14): The single most important signal. A tactical game
      must recommend tactical games; an RPG must recommend RPGs.
    • Other genres  (×10): Strong genre overlap still matters a lot.
    • Seed-matched tags (×6): Tags are Steam community tags representing the
      game's core mechanics, themes, and mood. High overlap = strong match.
    • Other tags (×0.8): Tags not in seed but in candidate; background signal.
    • Categories (×4): Steam-defined categories (single-player, multiplayer, etc)
    • Price similarity (×0.5): Games in similar price ranges may appeal similarly
    ─────────────────────────────────────────────────────────────────────────
    """
    all_games      = [seed] + candidates
    all_genres     = sorted({g for game in all_games for g in game.get("genres", [])})
    all_tags       = sorted({t for game in all_games for t in game.get("tags", [])})
    all_categories = sorted({c for game in all_games for c in game.get("categories", [])})

    seed_tag_set   = set(seed.get("tags", []))
    seed_genres    = seed.get("genres", [])
    seed_categories = set(seed.get("categories", []))
    seed_price     = seed.get("price", 0)
    primary_genre  = seed_genres[0] if seed_genres else None

    def encode(game: dict) -> np.ndarray:
        game_genre_set = set(game.get("genres", []))
        game_tag_set   = set(game.get("tags", []))
        game_category_set = set(game.get("categories", []))
        game_price = game.get("price", 0)

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

        # Category vector
        cat_vec = [
            4.0 if c in game_category_set and c in seed_categories else
            (2.0 if c in game_category_set else 0.0)
            for c in all_categories
        ]

        # Price similarity — games at similar price points
        price_diff = abs(game_price - seed_price) / max(seed_price, 1.0)
        price_sim = [(1.0 - min(price_diff, 1.0)) * 0.5]

        # Rating as secondary signal
        rating = [game.get("rating", 0) / 100.0 * 0.3]

        return np.array(genre_vec + tag_vec + cat_vec + price_sim + rating, dtype=np.float32)

    seed_vec  = encode(seed)
    cand_vecs = np.array([encode(c) for c in candidates])
    return seed_vec, cand_vecs

def recommend(seed: dict, candidates: list, top_n: int = 5) -> list:
    """Generate recommendations using cosine similarity."""
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

# ── ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])
    try:
        results = search_games(q)
        return jsonify(results)
    except Exception as exc:
        logging.error("Search error: %s", exc)
        return jsonify({"error": str(exc)}), 500

@app.route("/api/recommend/<int:game_id>")
def api_recommend(game_id: int):
    try:
        seed = get_game_detail(game_id)
        if seed is None:
            return jsonify({"error": f"Game {game_id} not found on Steam"}), 404
        
        # Extract genres and tags for candidate fetching
        seed_genres = seed.get("genres", [])
        seed_tags = seed.get("tags", [])
        
        candidates = fetch_candidates(seed_genres, seed_tags, exclude_id=game_id)
        recs = recommend(seed, candidates)
        
        return jsonify({"seed": seed, "recommendations": recs})
    except Exception as exc:
        logging.error("Recommend error: %s", exc)
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
