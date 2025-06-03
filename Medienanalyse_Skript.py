import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
import pandas as pd
from nltk import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Text bereinigen

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)


def clean_text(text):
    if not isinstance(text, str) or pd.isna(text):
        return []
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'#\w+|\@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and not word.isdigit()]
    return tokens


try:
    df = pd.read_csv('posts.csv', sep=';', encoding='utf-8')
except FileNotFoundError:
    print("Datei 'posts.csv' nicht gefunden. Bitte den Dateipfad überprüfen.")
    exit()

print("Spaltennamen:", df.columns.tolist())

df.columns = df.columns.str.strip()

if 'Text' not in df.columns:
    print("Fehler: Spalte 'Text' nicht gefunden. Verfügbare Spalten:", df.columns.tolist())
    possible_text_columns = [col for col in df.columns if 'text' in col.lower()]
    if possible_text_columns:
        print(f"Verwende stattdessen die Spalte: {possible_text_columns[0]}")
        text_column = possible_text_columns[0]
    else:
        print("Keine passende Spalte gefunden. Bitte die CSV-Datei überprüfen.")
        exit()
else:
    text_column = 'Text'

df['Cleaned_Text'] = df[text_column].apply(clean_text)
print(df[['Post-ID', 'Cleaned_Text']].head())

# Häufigkeiten ermitteln
all_tokens = [token for tokens in df['Cleaned_Text'] for token in tokens]

freq_dist = FreqDist(all_tokens)
print(freq_dist.most_common(10))  # Top 10 Wörter

# WordCloud erstellen
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(freq_dist))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Sentiment-Analyse
analyzer = SentimentIntensityAnalyzer()

df['Sentiment'] = df['Text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
print(df[['Post-ID', 'Text', 'Sentiment']].head())

avg_sentiment = df['Sentiment'].mean()
print(f"Durchschnittliches Sentiment: {avg_sentiment:.2f}")

# Fear words identifizieren
fear_words = ['danger', 'threat', 'crisis', 'scandal', 'shocking']
df['Fear_Words'] = df['Text'].apply(lambda x: sum(1 for word in fear_words if word in x.lower()))
print(df[['Post-ID', 'Text', 'Fear_Words']].head())

# Korrelation berechnen
correlation_sentiment = df['Sentiment'].corr(df['Gesamt'])
correlation_fear = df['Fear_Words'].corr(df['Gesamt'])
print(f"Korrelation Sentiment-Engagement: {correlation_sentiment:.2f}")
print(f"Korrelation Fear_Words-Engagement: {correlation_fear:.2f}")

# Excel erstellen
df.to_excel('medienanalyse_results.xlsx', index=False)
print("Daten wurden erfolgreich in 'medienanalyse_results.xlsx' exportiert.")
