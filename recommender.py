from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_similarity(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["description"])
    
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

def recommend(game_name, df, similarity, top_n=5):
    idx = df[df["name"] == game_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    indices = [i[0] for i in scores]
    return df.iloc[indices]