import requests 
import pandas as pd
from magic_strings import API_KEY

API_KEY = API_KEY

def fetch_games(query):
    url = f"https://api.rawg.io/api/games"
    params = {
        "key": API_KEY,
        "search": query,
        "page_size": 40
    }

    response = requests.get(url, params=params)
    data = response.json()["results"]
    
    games = []
    
    for game in data:
        games.append({
            "name": game["name"],
            "genres": [g["name"] for g in game["genres"]],
            "platforms": [p["platform"]["name"] for p in game["platforms"]],
            "rating": game["rating"],
            "released": game["released"],
            "description": game.get("slug", ""),  # We'll improve this later
            "image": game["background_image"]
        })
    
    return pd.DataFrame(games)

