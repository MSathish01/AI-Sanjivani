"""
AI-Sanjivani Flask API
RESTful API for health risk assessment
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.models.health_risk_classifier import HealthRiskClassifier
from ml_engine.speech_engine import MultilingualSpeechEngine

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Initialize ML models
classifier = HealthRiskClassifier()
classifier.train()
speech_engine = MultilingualSpeechEngine(offline_mode=True)

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AI-Sanjivani',
        'version': '1.0.0'
    })

@app.route('/api/assess', methods=['POST'])
def assess_health():
    """
    Health risk assessment endpoint
    Accepts symptoms in any supported language
    """
    try:
        data = request.get_json()
        
        symptoms = data.get('symptoms', [])
        age = data.get('age', 30)
        gender = data.get('gender', 'M')
        language = data.get('language', 'english')
        
        if not symptoms:
            return jsonify({
                'error': 'No symptoms provided',
                'message': 'Please provide at least one symptom'
            }), 400
        
        # Perform assessment
        result = classifier.predict_risk(
            symptoms=symptoms,
            age=int(age),
            gender=gender
        )
        
        return jsonify({
            'success': True,
            'risk_level': result['risk_level'],
            'confidence': result['confidence'],
            'explanation': result['explanation'].get(language, result['explanation']['english']),
            'recommendations': result['recommendations'].get(language, result['recommendations']['english']),
            'symptoms_detected': result['symptoms_detected']
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Assessment failed'
        }), 500

@app.route('/api/extract-symptoms', methods=['POST'])
def extract_symptoms():
    """
    Extract symptoms from text input
    Supports Hindi, Marathi, Tamil, English
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'error': 'No text provided',
                'message': 'Please provide symptom description'
            }), 400
        
        symptoms = speech_engine.extract_symptoms_from_text(text)
        
        return jsonify({
            'success': True,
            'symptoms': symptoms,
            'original_text': text
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Symptom extraction failed'
        }), 500

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    """Get list of supported symptoms with translations"""
    return jsonify({
        'symptoms': classifier.symptom_mapping,
        'languages': ['english', 'hindi', 'marathi', 'tamil']
    })

if __name__ == '__main__':
    print("🏥 Starting AI-Sanjivani API Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)