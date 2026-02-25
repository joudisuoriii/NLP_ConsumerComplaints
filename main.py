# main.py
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Einmaliges Herunterladen der grundlegenden NLTK-Dateien
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

# 1️⃣ Daten laden (Stichprobe von 5000 Zeilen entnehmen, um den Prozess zu beschleunigen)
print("Loading data...")
df = pd.read_csv('consumer_complaints.csv')
df = df[['Consumer complaint narrative']].dropna().sample(5000, random_state=42)

# 2️⃣ Datenvorverarbeitung (Preprocessing)
def clean_text(text):
    text = str(text).lower()                  # In Kleinbuchstaben umwandeln
    text = re.sub(r'[^\w\s]', '', text)       # Satzzeichen entfernen
    text = re.sub(r'\d+', '', text)           # Zahlen entfernen
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2 and w.isalpha()]
    # len(w) > 2 : Ignoriert sehr kurze Wörter
    # w.isalpha() : Ignoriert Symbole und Sonderzeichen
    return ' '.join(words)

print("Cleaning text...")
df['clean_text'] = df['Consumer complaint narrative'].apply(clean_text)

# 3️⃣ Vektorisierung (Umwandlung in numerische Vektoren)
print("Vectorizing...")
# Methode 1: Bag of Words
count_vect = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
bow_matrix = count_vect.fit_transform(df['clean_text'])

# Methode 2: TF-IDF
tfidf_vect = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
tfidf_matrix = tfidf_vect.fit_transform(df['clean_text'])

# 4️⃣ Themenextraktion (Topic Extraction)
n_topics = 5

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-top_n - 1:-1]]
        print(f"Topic {idx + 1}: {', '.join(top_words)}")

# Anwendung von LDA mit Bag of Words
print("\n--- LDA Topics (using Bag of Words) ---")
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(bow_matrix)
print_topics(lda, count_vect)

# Anwendung von NMF mit TF-IDF
print("\n--- NMF Topics (using TF-IDF) ---")
nmf = NMF(n_components=n_topics, random_state=42)
nmf.fit(tfidf_matrix)
print_topics(nmf, tfidf_vect)

