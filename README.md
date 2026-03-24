# NEXUS — AI-Powered Game Recommender

A content-based ML game recommendation engine using the RAWG API with a cyberpunk-themed frontend.

---

## Quick Start

### 1. Get a free RAWG API key

1. Go to [https://rawg.io/apidocs](https://rawg.io/apidocs)
2. Click **"Get API key"** and sign up (free)
3. Copy your API key

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

**Linux / macOS:**
```bash
export RAWG_API_KEY="your_api_key_here"
```

**Windows CMD:**
```cmd
set RAWG_API_KEY=your_api_key_here
```

**Or edit `config.py` directly**.

### 4. Run the app

```bash
python app.py
```

Open your browser at [http://localhost:5000](http://localhost:5000)

---

## How it works

### ML Algorithm (Content-Based Filtering)

1. **Feature Engineering** — For the selected game and ~60 candidate games (pulled from RAWG by genre), we build a multi-dimensional feature vector:
   - **Genre IDs** — multi-hot encoded, weighted ×3 (most important signal)
   - **Tag IDs** — multi-hot encoded, weighted ×2 (gameplay themes, settings, mechanics)
   - **Platform IDs** — multi-hot encoded
   - **Metacritic score** — normalised to [0, 1]
   - **Playtime** — log-normalised to reduce outlier influence

2. **Cosine Similarity** — We compute the cosine similarity between the seed game's vector and every candidate's vector. Cosine similarity is ideal here because it measures the *angle* between vectors (genre/tag overlap) regardless of magnitude.

3. **Ranking** — The top 5 candidates by similarity score are returned as recommendations.

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/search?q=<query>` | Autocomplete search, returns up to 8 games |
| `GET /api/recommend/<game_id>` | Returns seed game details + 5 recommendations |

---

## Project Structure

```
game-recommender/
├── app.py              # Flask server + ML engine
├── requirements.txt    # Python dependencies
├── README.md
└── templates/
    └── index.html      # Single-page frontend
```
