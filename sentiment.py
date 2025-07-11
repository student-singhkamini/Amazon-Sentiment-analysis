import pandas as pd, re, joblib, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# nltk.download('stopwords')

df = pd.read_csv('reviews_dataset.csv')   
print("Shape:", df.shape)    

label_map = {'good': 'Positive', 'bad': 'Negative', 'worst': 'Negative'}
df['Sentiment'] = df['review_label'].map(label_map)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub(r'\W', ' ', text)       
    text = text.lower()
    tokens = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)



df['clean'] = df['review_text'].apply(preprocess)

# ---------- 5. TF‑IDF Vectorization ----------
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['clean'])
y = df['Sentiment']

# ---------- 6. Train–Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- 7. Model Training ----------
model = MultinomialNB()
model.fit(X_train, y_train)

# ---------- 8. Evaluation ----------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------- 9. Save Model + Vectorizer ----------
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model & vectorizer saved!")