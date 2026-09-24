from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("Model/fake_news_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("Model/tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None

    if request.method == "POST":

        news = request.form["news"].strip()

        # Check if news is too short
        if len(news.split()) < 10:

            prediction = "INSUFFICIENT TEXT"
            probability = None

        else:

            # Convert news into numbers
            news_tfidf = vectorizer.transform([news])

            # Make prediction
            result = model.predict(news_tfidf)[0]

            # Get probabilities
            probabilities = model.predict_proba(news_tfidf)[0]

            fake_probability = probabilities[0] * 100
            real_probability = probabilities[1] * 100

            # Check confidence
            if max(fake_probability, real_probability) < 60:

                prediction = "UNCERTAIN"
                probability = round(
                    max(fake_probability, real_probability), 2
                )

            elif result == 0:

                prediction = "FAKE NEWS"
                probability = round(fake_probability, 2)

            else:

                prediction = "REAL NEWS"
                probability = round(real_probability, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)