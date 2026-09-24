import joblib

# Load model and vectorizer
model = joblib.load("Model/fake_news_model.pkl")
vectorizer = joblib.load("Model/tfidf_vectorizer.pkl")

# Get news from user
news = input("\nEnter news article: ").strip()

# Check input length
if len(news.split()) < 10:
    print("\n⚠️ Please enter a longer news article.")
    print("For better results, enter at least 10 words.")
    exit()

# Convert text into numbers
news_tfidf = vectorizer.transform([news])

# Prediction
prediction = model.predict(news_tfidf)[0]

# Probability
probability = model.predict_proba(news_tfidf)[0]

fake_probability = probability[0] * 100
real_probability = probability[1] * 100

# Difference between probabilities
confidence = max(fake_probability, real_probability)

print("\n-----------------------------")

if confidence < 60:
    print("⚠️ Prediction: UNCERTAIN")
    print("The model is not confident enough.")

elif prediction == 0:
    print("⚠️ Prediction: FAKE NEWS")
    print(f"Fake probability: {fake_probability:.2f}%")
    
else:
    print("✅ Prediction: REAL NEWS")
    print(f"Real probability: {real_probability:.2f}%")

print("-----------------------------")
print(f"Fake probability: {fake_probability:.2f}%")
print(f"Real probability: {real_probability:.2f}%")