# NEXUS — AI-Powered Game Recommender

A content-based ML game recommendation engine using the Steam Store API with a cyberpunk-themed frontend.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Steam API key (Optional)

The app uses **public Steam Store API** by default (no key required), but for enhanced features you can optionally set a Steam Web API key for your `config.py`:

**Linux / macOS:**
```bash
export STEAM_API_KEY="your_steam_api_key"
```

**Windows CMD:**
```cmd
set STEAM_API_KEY=your_steam_api_key
```

**Or edit `config.py` directly** (currently not required).

### 3. Run the app

```bash
python app.py
```

Open your browser at [http://localhost:5000](http://localhost:5000)

---

## How it works

### Data Sources

1. **Steam Store API** — Game metadata, screenshots, prices, release dates, genres, categories
2. **SteamSpy API** — Community tags, user ratings, playtime statistics (public, no key required)
3. **Steam Web API** — Additional game details and user-generated tags (optional)

### ML Algorithm (Content-Based Filtering)

1. **Feature Engineering** — For the selected game and ~60 candidate games (pulled from Steam), we build a multi-dimensional feature vector:
   - **Genres** — text-based genre names, weighted ×14 for primary genre and ×10 for others
   - **Community Tags** — Steam community tags representing gameplay themes, mechanics, and mood, weighted by seed overlap
   - **Categories** — Steam-defined categories (single-player, multiplayer, etc), weighted ×4
   - **Price** — normalized price similarity to find games in similar price ranges
   - **Rating** — community rating (SteamSpy score)

2. **Cosine Similarity** — We compute the cosine similarity between the seed game's vector and every candidate's vector. Cosine similarity measures the *angle* between vectors (genre/tag overlap) regardless of magnitude, making it perfect for content-based recommendations.

3. **Ranking** — The top 5 candidates by similarity score are returned as recommendations.

### Why Steam?

- **Accurate Game Data** — Authoritative source for all Steam games
- **Community-Driven Tags** — Real player feedback via SteamSpy's community tagging system
- **Rich Metadata** — Genres, categories, platforms, release dates, prices, screenshots
- **No Gatekeeping** — Public APIs, no artificial rate limits for reasonable use
- **Active Ecosystem** — Thousands of indie and AAA games continuously updated

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/search?q=<query>` | Search games by name (uses SteamSpy), returns up to 8 results |
| `GET /api/recommend/<app_id>` | Returns seed game details + 5 AI recommendations |

---

## Project Structure

```
game-recommender/
├── app.py              # Flask server + ML engine + Steam API integration
├── config.py           # Steam API configuration
├── requirements.txt    # Python dependencies
├── README.md
└── templates/
    └── index.html      # Single-page frontend
```

---

## Stack

- **Backend**: Flask + NumPy + Scikit-learn
- **Frontend**: Vanilla JS + Canvas + CSS animations
- **APIs**: Steam Store API, SteamSpy API, optional Steam Web API
- **ML**: Cosine Similarity (Content-Based Filtering)

---

## Troubleshooting

**"No results found"** — Search queries need at least 2 characters and must match Steam game names.

**Slow recommendations** — The first recommendation may take 5-10 seconds as we fetch game data from Steam. Subsequent searches are faster due to caching.

**Invalid App ID** — Make sure you're using valid Steam app IDs. You can find them at `steampowered.com` or via SteamSpy.

---

## License

Open source. Use and modify freely.
