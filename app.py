from flask import Flask, render_template, request
import os
import joblib
from gensim.models import Word2Vec
from src.utils import preprocess_text, vectorize_text

app = Flask(__name__)

model = joblib.load("Outputs/Models/suicide_detection_model.pkl")
w2v = Word2Vec.load("Outputs/Models/Vectorizer.model")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    text = ""

    if request.method == "POST":
        text = request.form["user_input"]
        tokens = preprocess_text(text)
        vector = vectorize_text(tokens, w2v)
        prediction = model.predict([vector])[0]
        probability = model.predict_proba([vector])[0][prediction]

    return render_template("index.html", 
                           prediction=prediction, 
                           probability=probability,
                           text=text)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",  # Escucha en todas las interfaces
        port=int(os.environ.get("PORT", 5000)),  # Usa el puerto que te da Railway
        debug=True
    )
