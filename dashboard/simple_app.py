"""
Simple Flask Dashboard for AI-Sanjivani
Basic web interface for health risk assessment
"""

from flask import Flask, request, jsonify
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.models.health_risk_classifier import HealthRiskClassifier

app = Flask(__name__)

# Initialize classifier
classifier = HealthRiskClassifier()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI-Sanjivani Health Assessment</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .header { background: #2E8B57; color: white; padding: 20px; text-align: center; border-radius: 10px; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input[type="text"], input[type="number"], select { 
                width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; 
            }
            button { 
                background: #2E8B57; color: white; padding: 15px 30px; 
                border: none; border-radius: 5px; cursor: pointer; font-size: 16px; 
            }
            button:hover { background: #1e6f42; }
            .result { 
                margin-top: 20px; padding: 20px; border-radius: 5px; 
                display: none; 
            }
            .green { background: #d4edda; color: #155724; }
            .yellow { background: #fff3cd; color: #856404; }
            .red { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏥 AI-Sanjivani</h1>
            <p>Rural Healthcare Assistant</p>
        </div>
        
        <form id="assessmentForm">
            <div class="form-group">
                <label for="symptoms">Symptoms (comma separated):</label>
                <input type="text" id="symptoms" name="symptoms" 
                       placeholder="e.g., fever, cough, headache" required>
                <small>You can use English, Hindi, or Marathi</small>
            </div>
            
            <div class="form-group">
                <label for="age">Age:</label>
                <input type="number" id="age" name="age" min="1" max="100" value="30" required>
            </div>
            
            <div class="form-group">
                <label for="gender">Gender:</label>
                <select id="gender" name="gender" required>
                    <option value="M">Male / पुरुष / पुरुष</option>
                    <option value="F">Female / महिला / महिला</option>
                </select>
            </div>
            
            <button type="submit">Assess Health Risk</button>
        </form>
        
        <div id="result" class="result">
            <h3>Assessment Result</h3>
            <div id="resultContent"></div>
        </div>

        <script>
            document.getElementById('assessmentForm').onsubmit = function(e) {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const symptoms = formData.get('symptoms').split(',').map(s => s.trim());
                const age = parseInt(formData.get('age'));
                const gender = formData.get('gender');
                
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        symptoms: symptoms,
                        age: age,
                        gender: gender
                    })
                })
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('result');
                    const contentDiv = document.getElementById('resultContent');
                    
                    const riskClass = data.risk_level.toLowerCase();
                    resultDiv.className = `result ${riskClass}`;
                    
                    const riskEmoji = {
                        'green': '✅',
                        'yellow': '⚠️',
                        'red': '🚨'
                    };
                    
                    contentDiv.innerHTML = `
                        <h4>${riskEmoji[riskClass]} Risk Level: ${data.risk_level}</h4>
                        <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
                        <p><strong>Explanation:</strong> ${data.explanation.english}</p>
                        <p><strong>Symptoms Detected:</strong> ${data.symptoms_detected.join(', ')}</p>
                        <div>
                            <strong>Recommendations:</strong>
                            <ul>
                                ${data.recommendations.english.map(rec => `<li>${rec}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                    
                    resultDiv.style.display = 'block';
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error assessing health risk. Please try again.');
                });
            };
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symptoms = data.get('symptoms', [])
        age = data.get('age', 30)
        gender = data.get('gender', 'M')
        
        # Use the health risk classifier
        result = classifier.predict_risk(symptoms, age, gender)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🏥 Starting AI-Sanjivani Health Assessment Dashboard...")
    print("📱 Access the dashboard at: http://localhost:5000")
    print("💡 You can enter symptoms in English, Hindi, or Marathi")
    app.run(debug=True, host='0.0.0.0', port=5000)