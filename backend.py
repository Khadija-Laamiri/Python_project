import speech_recognition as sr
from flask import jsonify,request,Flask

app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def recognize_speech():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(file)
    with audio_file as source:
        audio = recognizer.record(source)

    text = recognizer.recognize_google(audio)
    return jsonify({'transcription': text})

if __name__ == '_main_':
    app.run()