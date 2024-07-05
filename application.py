from flask import Flask, request, jsonify
import os
import librosa
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def say_hello(username = "World"):
    return '<p>Hello %s!</p>\n' % username

# add a rule for the index page.
app.add_url_rule('/', 'index', (lambda:
    say_hello()))

def analyze_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Use librosa.pyin to extract fundamental frequencies
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # Filter out unvoiced frequencies
        voiced_f0 = f0[voiced_flag]
        
        min_freq = np.min(voiced_f0)
        max_freq = np.max(voiced_f0)
        
        return min_freq, max_freq
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return None, None

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        min_freq, max_freq = analyze_audio(file_path)
        if min_freq is None or max_freq is None:
            return jsonify({"error": "Failed to analyze audio"}), 500
        return jsonify({
            "min_frequency": min_freq,
            "max_frequency": max_freq
        }), 200

if __name__ == '__main__':
    app.run(debug=True)
