import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")


# Add labels
fake["label"] = 0
true["label"] = 1


# Combine datasets
data = pd.concat([fake, true], ignore_index=True)


# Keep required columns
data = data[["text", "label"]]


# Remove empty values
data = data.dropna()


# Remove duplicates
data = data.drop_duplicates()


# Separate text and labels
X = data["text"]
y = data["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Convert text into numbers
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1,2),
    min_df=2,
    sublinear_tf=True

)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("Text converted into numbers!")
print("Training shape:", X_train_tfidf.shape)
print("Testing shape:", X_test_tfidf.shape)


# Create model
model = LogisticRegression(
    max_iter=1000
)


# Train model
model.fit(X_train_tfidf, y_train)

print("Model trained successfully!")


# Make predictions
y_pred = model.predict(X_test_tfidf)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save the model
joblib.dump(model, "Model/fake_news_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "Model/tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")