import os
import joblib
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\Ethnotech\Desktop\proj\Spam_SMS"

# 1. Load your saved model artifacts safely
try:
    model = joblib.load(os.path.join(BASE_DIR, 'spam_model.pkl'))
    vectorizer = joblib.load(os.path.join(BASE_DIR, 'vectorizer.pkl'))
except FileNotFoundError:
    print("Error: Could not find the saved .pkl files. Please run your training script first.")
    exit()

def classify_and_plot(msg):
    # Transform raw input text to numerical features
    vectorized_msg = vectorizer.transform([msg])
    
    # Extract prediction and underlying probability array [prob_of_0, prob_of_1]
    prediction = model.predict(vectorized_msg)[0]
    probabilities = model.predict_proba(vectorized_msg)[0]
    
    prob_ham = probabilities[0] * 100
    prob_spam = probabilities[1] * 100
    
    # Calculate performance metrics based on the winning class
    if prediction == 1:
        label = "SPAM DETECTED"
        confidence = prob_spam
    else:
        label = "NOT SPAM (HAM)"
        confidence = prob_ham
        
    error_margin = 100 - confidence
    
    # Print numerical results to the terminal window
    print("\n--- Decision Analysis ---")
    print(f"Result: {label}")
    print(f"Model Confidence: {confidence:.2f}%")
    print(f"Estimated Error Margin: ±{error_margin:.2f}%")
    print(f"Detailed Breakdown -> Ham: {prob_ham:.1f}%, Spam: {prob_spam:.1f}%")
    print("Close the chart window to input another message.\n")
    
    # 2. GENERATE LIVE PROBABILITY DISTRIBUTION CHART
    plt.figure(figsize=(6, 4))
    classes = ['Not Spam (Ham)', 'Spam']
    scores = [prob_ham, prob_spam]
    colors = ['#2ecc71', '#e74c3c'] # Green for clean, Red for spam
    
    bars = plt.bar(classes, scores, color=colors, width=0.5)
    plt.ylim(0, 100)
    plt.title(f"Model Probability Breakdown\nInput: \"{msg[:40]}...\"")
    plt.ylabel("Probability Percentage (%)")
    
    # Add percentage labels directly on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
                 
    plt.tight_layout()
    plt.show() # Code execution pauses here until the window is closed

# --- Interactive Terminal Loop ---
print("--- Advanced Interactive Spam Testing Panel ---")
print("Type your message below. (Type 'exit' to close the program)\n")

while True:
    user_input = input("Enter message to test: ")
    if user_input.lower() == 'exit':
        print("Exiting testing panel.")
        break
    
    if user_input.strip() == "":
        continue
        
    classify_and_plot(user_input)