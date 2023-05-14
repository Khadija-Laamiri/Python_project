from flask import Flask, render_template, request, jsonify
from Speech_Model import load_model
from preprocess import preprocess_audio

app = Flask(__name__)

# Load the speech recognition model
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize_speech():
    # Get the audio file from the request
    audio_file = request.files['audio']

    # Preprocess the audio data
    preprocessed_data = preprocess_audio(audio_file)

    # Perform speech recognition using the model
    transcription = model.predict(preprocessed_data)

    # Return the transcription as a JSON response
    return jsonify({'transcription': transcription})

if __name__ == '__main__':
    app.run()
