import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Descarreguem el diccionari de paraules (només ho fa el primer cop)
nltk.download('vader_lexicon', quiet=True)

print("🚀 Iniciant l'anàlisi de sentiment dels tuits...")

# 1. Carregar el dataset de Kaggle (2021-2022)
# (Canvia "Tweet" pel nom real de la columna del text si es diu diferent, ex: "text" o "Message")
df_tweets = pd.read_csv("stock_tweets.csv")

# 2. Convertir la data per agrupar correctament (eliminem les hores)
df_tweets['Date'] = pd.to_datetime(df_tweets['Date'], errors='coerce').dt.normalize()

# 3. Inicialitzar l'analitzador de sentiment VADER
sia = SentimentIntensityAnalyzer()
print("🧠 Llegint milers de tuits... (això pot trigar uns segons o minuts depenent de la mida del CSV)")

# 4. Funció per extreure només el "compound" (el sentiment global del tuit)
def obtenir_sentiment(text):
    try:
        return sia.polarity_scores(str(text))['compound']
    except:
        return 0.0

# Apliquem la funció a la columna dels tuits
# ATENCIÓ: Si la teva columna de text es diu 'text', canvia 'Tweet' per 'text' a la línia de sota
df_tweets['Sentiment_Score'] = df_tweets['Tweet'].apply(obtenir_sentiment)

# 5. Agrupar per dia per fer la mitjana (Tableau necessita 1 dada diària)
df_diari = df_tweets.groupby('Date')['Sentiment_Score'].mean().reset_index()

# 6. Guardar el resultat
nom_sortida = "sentiment_diari_2021_2022.csv"
df_diari.to_csv(nom_sortida, index=False)

print(f"✨ Procés completat! S'ha generat el fitxer: {nom_sortida}")
print("--- Mostra del resultat ---")
print(df_diari.head())