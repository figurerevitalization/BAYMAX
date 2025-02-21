import ollama
import requests
import base64
from playsound import playsound
import os
import pandas as pd
import joblib
import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def speak(text: str, model: str = "aura-arcas-en", filename: str = "ASSETS/output_audio.mp3"):
    """
    Converts text to speech using the Deepgram API and plays the audio.
    """
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    url = "https://deepgram.com/api/ttsAudioGeneration"
    payload = {"text": text, "model": model}

    response = requests.post(url, json=payload)
    response.raise_for_status()  # Ensure the request was successful

    with open(filename, 'wb') as audio_file:
        audio_file.write(base64.b64decode(response.json()['data']))

    playsound(filename)
    os.remove(filename)


def load_chat_history():
    """
    Loads the chat history from the 'chat_history.json' file.
    If the file does not exist or is invalid, returns an empty list.
    """
    if os.path.exists("chat_history.json"):
        try:
            with open("chat_history.json", "r") as file:
                data = file.read().strip()  # Remove any leading/trailing whitespace
                return json.loads(data) if data else []  # Return an empty list if the file is empty
        except json.JSONDecodeError:
            print("Error: chat_history.json is invalid. Loading empty chat history.")
            return []
    return []

def save_chat_history():
    """Saves the chat history to the 'chat_history.json' file."""
    try:
        # Ensure only serializable data (strings, numbers, etc.) are stored
        serializable_history = [{'role': entry['role'], 'content': entry['content']} for entry in chat_history]
        with open("chat_history.json", "w") as file:
            json.dump(serializable_history, file)
    except Exception as e:
        print(f"Error saving chat history: {e}")

# Load chat history on startup
chat_history = load_chat_history()

def greet():
    """Greets the user by asking their name."""
    try:
        greet_response = ollama.chat(model='Baymax:latest', messages=[{'role': 'user', 'content': 'Greet me asking my name! and in the whole chat do not use any emojies!'}])
        
        # Extract only the relevant content from the response
        greet_res = greet_response.get('message', {}).get('content', 'Hello! What is your name?')
        
        speak(greet_res)
        print(f"BAYMAX: {greet_res}")
        chat_history.append({'role': 'assistant', 'content': greet_res})
        save_chat_history()
    except Exception as e:
        print(f"BAYMAX: Error generating response: {str(e)}")

def send_message():
    """Handles the main message flow with the user."""
    user_message = input("You: ").lower()
    if user_message in ['exit', 'quit', 'bye']:
        bye_message = "It was a pleasure assisting you. Take care!"
        speak(bye_message)
        print(f"BAYMAX: {bye_message}")
        time.sleep(2)
        exit()

    chat_history.append({'role': 'user', 'content': user_message})
    save_chat_history()

    if user_message in ['survey', 'start survey', 'survay', 'start survay']:
        Survey()
        return

    try:
        # Use the existing chat history to generate a response
        response = ollama.chat(model='Baymax:latest', messages=chat_history)
        
        # Extract just the content of the message
        if 'message' in response:
            baymax_response = response['message']['content']
            chat_history.append({'role': 'assistant', 'content': baymax_response})
            save_chat_history()
            speak(baymax_response)
            print(f"BAYMAX: {baymax_response}")
        else:
            print("Unexpected response format:", response)
    except Exception as e:
        print(f"BAYMAX: Error generating response: {str(e)}")

def Survey():
    """Handles the survey process for heart disease or female diabetes."""
    survey_prompts = {
        'heart': [
            "Enter age:", "Enter sex (1=male, 0=female):", "Enter chest pain type (0-3):",
            "Enter resting blood pressure:", "Enter cholesterol level:", "Fasting blood sugar > 120 mg/dl (1=True, 0=False):",
            "Enter resting ECG results (0-2):", "Enter max heart rate:", "Exercise-induced angina (1=True, 0=False):",
            "ST depression induced by exercise:", "Slope of peak exercise ST segment (0-2):",
            "Number of major vessels colored by fluoroscopy (0-3):", "Thalassemia (0=normal; 1=fixed defect; 2=reversible defect):"
        ],
        'female diabetes': [
            "Enter pregnancies:", "Enter glucose level:", "Enter blood pressure:", "Enter skin thickness:", "Enter insulin level:",
            "Enter BMI:", "Enter diabetes pedigree function:", "Enter age:"
        ]
    }

    # Load models only if not already cached
    if not os.path.exists('heart_model.pkl') or not os.path.exists('diabetes_model.pkl'):
        print("Training models... Please wait.")
        train_and_save_models()

    heart_model = joblib.load('heart_model.pkl')
    diabetes_model = joblib.load('diabetes_model.pkl')

    choice = 'Please specify the kind of survey you want to take: heart or female diabetes.'
    speak(choice)

    survey_choice = input(f"BAYMAX: {choice}\nYou: ").lower()
    if survey_choice == 'heart':
        process_heart_survey(heart_model)
    elif survey_choice == 'female diabetes':
        process_diabetes_survey(diabetes_model)
    else:
        unknown_response = "Sorry, I can only assist with heart or female diabetes surveys."
        print(f"BAYMAX: {unknown_response}")
        chat_history.append({'role': 'assistant', 'content': unknown_response})
        speak(unknown_response)

def process_heart_survey(heart_model):
    """Processes the heart disease survey and makes a prediction."""
    print("Please provide the following information for heart disease assessment.")
    patient_data = get_survey_data('heart')
    prediction = predict_health_condition(patient_data, 'heart', heart_model)
    respond_with_prediction('heart', prediction)

def process_diabetes_survey(diabetes_model):
    """Processes the female diabetes survey and makes a prediction."""
    print("Please provide the following information for female diabetes assessment.")
    patient_data = get_survey_data('female diabetes')
    prediction = predict_health_condition(patient_data, 'female diabetes', diabetes_model)
    respond_with_prediction('female diabetes', prediction)

def get_survey_data(survey_type):
    """Prompts the user to enter data based on the survey type."""
    survey_prompts = {
        'heart': [
            "Enter age:", "Enter sex (1=male, 0=female):", "Enter chest pain type (0-3):", 
            "Enter resting blood pressure:", "Enter cholesterol level:", "Fasting blood sugar > 120 mg/dl (1=True, 0=False):",
            "Enter resting ECG results (0-2):", "Enter max heart rate:", "Exercise-induced angina (1=True, 0=False):",
            "ST depression induced by exercise:", "Slope of peak exercise ST segment (0-2):",
            "Number of major vessels colored by fluoroscopy (0-3):", "Thalassemia (0=normal; 1=fixed defect; 2=reversible defect):"
        ],
        'female diabetes': [
            "Enter pregnancies:", "Enter glucose level:", "Enter blood pressure:", "Enter skin thickness:", 
            "Enter insulin level:", "Enter BMI:", "Enter diabetes pedigree function:", "Enter age:"
        ]
    }

    data = []
    for prompt in survey_prompts[survey_type]:
        speak(prompt)
        data.append(float(input(prompt)))
    return data

def train_and_save_models():
    """Trains the heart disease and female diabetes models and saves them."""
    heart_data = pd.read_csv('Heart_disease_cleveland_new.csv')
    diabetes_data = pd.read_csv('diabetes.csv')

    def train_model(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
        return model

    heart_model = train_model(heart_data.drop(columns='target'), heart_data['target'])
    diabetes_model = train_model(diabetes_data.drop(columns='Outcome'), diabetes_data['Outcome'])

    joblib.dump(heart_model, 'heart_model.pkl')
    joblib.dump(diabetes_model, 'diabetes_model.pkl')

def predict_health_condition(patient_data, survey_type, model):
    """Makes a prediction based on the model and patient data."""
    prediction = model.predict([patient_data])[0]
    return prediction

def respond_with_prediction(survey_type, prediction):
    """Responds to the user based on the model's prediction."""
    if survey_type == 'heart':
        condition = "heart disease" if prediction == 1 else "no heart disease"
    elif survey_type == 'female diabetes':
        condition = "diabetes" if prediction == 1 else "no diabetes"

    response = f"Based on your responses, you are predicted to have {condition}."
    print(f"BAYMAX: {response}")
    chat_history.append({'role': 'assistant', 'content': response})
    speak(response)

def main():
    """Main function to run the program."""
    greet()

    while True:
        send_message()

if __name__ == "__main__":
    main()