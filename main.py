import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

# Einmaliges Herunterladen der grundlegenden NLTK-Dateien
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text, lemmatizer, stop_words):
    text = str(text).lower() 
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\d+', '', text) 
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

def get_topic_words(model, vectorizer, top_n=10):
    topic_words = []
    for topic in model.components_:
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-top_n - 1:-1]]
        topic_words.append(top_words)
    return topic_words

def print_topics(topic_words):
    for idx, words in enumerate(topic_words):
        print(f"Topic {idx + 1}: {', '.join(words)}")

# Der Hauptteil des Programms muss unter diesem Block stehen (Wichtig für macOS/Multiprocessing)
if __name__ == '__main__':
    # 1. Daten laden
    print("Loading data...")
    try:
        df = pd.read_csv('consumer_complaints.csv', low_memory=False)
        df = df[['Consumer complaint narrative']].dropna().sample(5000, random_state=42)
    except FileNotFoundError:
        print("Error: Die Datei 'consumer_complaints.csv' wurde nicht gefunden.")
        exit()

    # 2. Datenvorverarbeitung (Preprocessing)
    print("Cleaning text...")
    stop_words = set(stopwords.words('english'))
    stop_words.update(['xxxx', 'xxxxxxxx', 'xxxxxxxxxxxx'])
    lemmatizer = WordNetLemmatizer()
    
    df['clean_text'] = df['Consumer complaint narrative'].apply(lambda x: clean_text(x, lemmatizer, stop_words))

    # 3. Vektorisierung
    print("Vectorizing...")
    count_vect = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
    bow_matrix = count_vect.fit_transform(df['clean_text'])

    tfidf_vect = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
    tfidf_matrix = tfidf_vect.fit_transform(df['clean_text'])

    # 4. Themenextraktion (Topic Extraction)
    n_topics = 5

    print("\n--- LDA Topics (using Bag of Words) ---")
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(bow_matrix)
    lda_topics = get_topic_words(lda, count_vect)
    print_topics(lda_topics)

    print("\n--- NMF Topics (using TF-IDF) ---")
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(tfidf_matrix)
    nmf_topics = get_topic_words(nmf, tfidf_vect)
    print_topics(nmf_topics)

    # 5. Coherence Score Implementierung (Tutor Feedback)
    print("\n--- Calculating Coherence Scores ---")
    texts = [text.split() for text in df['clean_text']]
    dictionary = Dictionary(texts)

    # LDA Coherence
    coherence_model_lda = CoherenceModel(topics=lda_topics, texts=texts, dictionary=dictionary, coherence='c_v')
    print(f"LDA Coherence Score: {coherence_model_lda.get_coherence():.4f}")

    # NMF Coherence
    coherence_model_nmf = CoherenceModel(topics=nmf_topics, texts=texts, dictionary=dictionary, coherence='c_v')
    print(f"NMF Coherence Score: {coherence_model_nmf.get_coherence():.4f}")

