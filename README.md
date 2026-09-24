# 📰 Fake News Detection

A Machine Learning-based web application that analyzes news text and predicts whether the news is **Real, Fake, or Uncertain**.

The application uses **TF-IDF Vectorization** and a trained Machine Learning classification model to analyze the text entered by the user.

---

## 🚀 Features

- 📰 Enter news text through a simple web interface
- 🤖 Machine Learning-based prediction
- ✅ Detects **Real News**
- ❌ Detects **Fake News**
- ⚠️ Returns **Uncertain** when the model is not confident enough
- 📊 Displays prediction confidence
- 🌐 Flask-based web application
- 💻 Simple and responsive user interface

---

## 🛠️ Technologies Used

- **Python**
- **Flask**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Joblib**
- **HTML**
- **CSS**
- **JavaScript**
- **TF-IDF Vectorization**
- **Machine Learning**

---

## 📂 Project Structure

```text
Fake-News-Detection/
│
├── Model/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── predict.py
├── test.py
├── train_model.py
├── requirements.txt
└── .gitignore
