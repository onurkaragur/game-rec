import pandas as pd

def prepare_features(df):
    df = df.copy()

    df["genres"] = df["genres"].apply(lambda x: " ".join(x))
    df["platforms"] = df["platforms"].apply(lambda x: " ".join(x))

    df["combined"] = (
        df["genres"] + " " +
        df["platforms"] + " " +
        df["description"]
    )

    return df