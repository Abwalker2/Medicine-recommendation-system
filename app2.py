from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Flask app setup
app = Flask(__name__)
CORS(app)

# Load data and vectorizer
with open("medicine_dict.pkl", "rb") as f:
    medicine_dict = pickle.load(f)

with open("similarity.pkl", "rb") as f:
    similarity = pickle.load(f)

# Recreate dataframe
import pandas as pd
new_df = pd.DataFrame(medicine_dict)
new_df = pd.DataFrame.from_dict(new_df)

# Load CountVectorizer
cv = CountVectorizer(stop_words='english', max_features=5000)
vectors = cv.fit_transform(new_df['tags']).toarray()

# NLP processing
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        reason = data.get("reason", "")
        description = data.get("description", "")
        input_text = (reason + " " + description).strip().lower()
        if not input_text:
            return jsonify({"error": "Please enter a reason or description."}), 400

        input_stemmed = stem(input_text)
        input_vector = cv.transform([input_stemmed]).toarray()
        similarity_scores = cosine_similarity(input_vector, vectors)

        top_index = np.argmax(similarity_scores)
        recommendation = new_df.iloc[top_index]['Drug_Name']
        return jsonify({"recommended_medicine": recommendation})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
