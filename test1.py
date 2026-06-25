import os
import joblib

BASE_DIR = r"C:\Users\Ethnotech\Desktop\proj\Spam_SMS"

# Load your saved model artifacts
try:
    model = joblib.load(os.path.join(BASE_DIR, 'spam_model.pkl'))
    vectorizer = joblib.load(os.path.join(BASE_DIR, 'vectorizer.pkl'))
except FileNotFoundError:
    print("Error: Could not find the saved .pkl files.")
    exit()

def classify_message(msg):
    vectorized_msg = vectorizer.transform([msg])
    prediction = model.predict(vectorized_msg)[0]
    
    if prediction == 1:
        return "SPAM DETECTED"
    else:
        return "NOT SPAM (HAM)"

# Continuous loop to let you type whatever you want
print("--- Interactive Spam Testing Panel ---")
print("Type your message below and press Enter. (Type 'exit' to quit)\n")

while True:
    user_input = input("Enter message to test: ")
    if user_input.lower() == 'exit':
        print("Exiting testing panel.")
        break
    
    if user_input.strip() == "":
        continue
        
    result = classify_message(user_input)
    print(f"Result: {result}\n")