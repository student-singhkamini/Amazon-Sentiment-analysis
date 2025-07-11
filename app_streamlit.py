import streamlit as st, joblib, re, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

stop = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub(r'\W', ' ', text).lower()
    return ' '.join(stemmer.stem(w) for w in text.split() if w not in stop)

st.set_page_config(page_title="Amazon Sentiment App")
st.title("🛒 Amazon Review Sentiment Analyzer")

review = st.text_area("Paste any Amazon review here:")

if st.button("Analyze Sentiment"):
    clean_text = preprocess(review)
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]

    if prediction == 'Positive':
        st.success("✅ Positive Review")
    elif prediction == 'Negative':
        st.error("❌ Negative Review")
    else:
        st.warning("⚠️ Neutral Review")
