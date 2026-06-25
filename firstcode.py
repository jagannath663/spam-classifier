import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Dynamically get the absolute path to your spam_or_not_spam.csv file
# This prevents FileNotFoundError regardless of how you run the script!
BASE_DIR = r"C:\Users\Ethnotech\Desktop\proj\Spam_SMS"
CSV_PATH = os.path.join(BASE_DIR, "spam_or_not_spam.csv")

try:
    df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: Could not find the file at {CSV_PATH}")
    print("Please double check your file name inside the Spam_SMS folder.")
    exit()

# Print the columns so we can see how the dataset is structured
print("Your Dataset Columns are:", df.columns.tolist())
print(df.head(3))

# Automatically picks the 1st column for text and 2nd for the spam/ham label
TEXT_COLUMN = df.columns[0]   
LABEL_COLUMN = df.columns[1]  

# Clean missing data
df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])

# 2. Split into features and targets
X = df[TEXT_COLUMN].astype(str)
y = df[LABEL_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Vectorize text data
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train the Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_tfidf)
print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Test with a custom input
def test_new_message(msg):
    vectorized = vectorizer.transform([msg])
    prediction = model.predict(vectorized)
    print(f"\nMessage: '{msg}'\nPrediction: {prediction[0]}")

test_new_message("Claim your free gift card now by clicking this link!")
import joblib

# Save the trained model and the vectorizer to disk
joblib.dump(model, os.path.join(BASE_DIR, 'spam_model.pkl'))
joblib.dump(vectorizer, os.path.join(BASE_DIR, 'vectorizer.pkl'))
print("\nModel and Vectorizer saved successfully as .pkl files!")hn
