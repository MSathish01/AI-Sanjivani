"""
AI-Sanjivani Health Risk Classification Model
Predicts health risk levels (Green/Yellow/Red) based on symptoms
Optimized for offline inference on low-end devices
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from typing import Dict, List, Tuple, Any

class HealthRiskClassifier:
    """
    Health risk classification model for rural healthcare
    Designed for explainability and offline inference
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=50,  # Reduced for mobile performance
            max_depth=10,
            random_state=42,
            n_jobs=1  # Single thread for mobile
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.symptom_mapping = self._load_symptom_mapping()
        
    def _load_symptom_mapping(self) -> Dict[str, List[str]]:
        """Load symptom mappings for Hindi/Marathi to English"""
        return {
            'fever': ['बुखार', 'ताप', 'fever', 'तापमान', 'காய்ச்சல்', 'உடல் சூடு', 'ஜுரம்'],
            'cough': ['खांसी', 'खोकला', 'cough', 'कफ', 'இருமல்', 'வறட்டு இருமல்', 'கபம் இருமல்'],
            'headache': ['सिरदर्द', 'डोकेदुखी', 'headache', 'माथा दुखना', 'தலைவலி', 'தலை வேदனை', 'மைக்ரேன்'],
            'body_ache': ['शरीर दर्द', 'अंग दुखी', 'body ache', 'बदन दर्द', 'உடல் வலி', 'மூட்டு வலி', 'தசை வலி'],
            'nausea': ['मतली', 'जी मिचलाना', 'nausea', 'उलटी', 'குமட்டல்', 'வாந்தி உணர்வு', 'மயக்கம்'],
            'diarrhea': ['दस्त', 'अतिसार', 'diarrhea', 'पेट खराब', 'வயிற்றுப்போக்கு', 'லூஸ் மோஷன்', 'வயிறு கெட்டது'],
            'vomiting': ['उल्टी', 'वमन', 'vomiting', 'कै', 'வாந்தி', 'ஓக்காளிப்பு', 'வமனம்'],
            'weakness': ['कमजोरी', 'अशक्तता', 'weakness', 'थकान', 'பலவீனம்', 'சோர்வு', 'களைப்பு'],
            'breathing_difficulty': ['सांस लेने में तकलीफ', 'श्वास कष्ट', 'breathing problem', 'दम फूलना', 'மூச்சு விடுவதில் சிரமம்', 'மூச்சு திணறல்', 'ஆஸ்துமா'],
            'chest_pain': ['छाती में दर्द', 'छातीत दुखी', 'chest pain', 'सीने में दर्द', 'மார்பு வலி', 'இதய வலி', 'நெஞ்சு வலி']
        }
    
    def preprocess_symptoms(self, symptoms: List[str]) -> np.ndarray:
        """
        Convert symptom list to feature vector
        Handles multilingual input (Hindi/Marathi/English)
        """
        feature_vector = np.zeros(len(self.symptom_mapping))
        
        for symptom in symptoms:
            symptom_lower = symptom.lower().strip()
            
            for idx, (eng_symptom, translations) in enumerate(self.symptom_mapping.items()):
                if symptom_lower in [t.lower() for t in translations]:
                    feature_vector[idx] = 1
                    break
        
        return feature_vector.reshape(1, -1)
    
    def train(self, training_data_path: str = None) -> Dict[str, Any]:
        """
        Train the health risk classification model
        Returns training metrics
        """
        if training_data_path:
            data = pd.read_csv(training_data_path)
        else:
            # Generate synthetic training data for demo
            data = self._generate_synthetic_data()
        
        # Prepare features and labels
        X = data.drop(['risk_level', 'patient_id'], axis=1, errors='ignore')
        y = data['risk_level']
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': self.model.score(X_test_scaled, y_test_encoded),
            'classification_report': classification_report(y_test_encoded, y_pred),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
        
        return metrics
    
    def predict_risk(self, symptoms: List[str], age: int = 30, gender: str = 'M') -> Dict[str, Any]:
        """
        Predict health risk level with explanation
        Returns risk level and reasoning in simple language
        """
        # Preprocess symptoms
        symptom_features = self.preprocess_symptoms(symptoms)
        
        # Add demographic features
        features = np.append(symptom_features[0], [age, 1 if gender.upper() == 'M' else 0])
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        risk_encoded = self.model.predict(features_scaled)[0]
        risk_proba = self.model.predict_proba(features_scaled)[0]
        
        risk_level = self.label_encoder.inverse_transform([risk_encoded])[0]
        confidence = max(risk_proba)
        
        # Generate explanation
        explanation = self._generate_explanation(symptoms, risk_level, confidence)
        
        return {
            'risk_level': risk_level,
            'confidence': float(confidence),
            'explanation': explanation,
            'recommendations': self._get_recommendations(risk_level),
            'symptoms_detected': symptoms
        }
    
    def _generate_explanation(self, symptoms: List[str], risk_level: str, confidence: float) -> Dict[str, str]:
        """Generate simple explanations in multiple languages"""
        
        explanations = {
            'Green': {
                'english': f"Your symptoms suggest low health risk. Confidence: {confidence:.1%}",
                'hindi': f"आपके लक्षण कम स्वास्थ्य जोखिम दर्शाते हैं। विश्वसनीयता: {confidence:.1%}",
                'marathi': f"तुमची लक्षणे कमी आरोग्य धोका दर्शवतात। विश्वसनीयता: {confidence:.1%}",
                'tamil': f"உங்கள் அறிகுறிகள் குறைந்த உடல்நல அபாயத்தைக் குறிக்கின்றன। நம்பகத்தன்மை: {confidence:.1%}"
            },
            'Yellow': {
                'english': f"Your symptoms suggest moderate health risk. Please consult a doctor. Confidence: {confidence:.1%}",
                'hindi': f"आपके लक्षण मध्यम स्वास्थ्य जोखिम दर्शाते हैं। कृपया डॉक्टर से सलाह लें। विश्वसनीयता: {confidence:.1%}",
                'marathi': f"तुमची लक्षणे मध्यम आरोग्य धोका दर्शवतात। कृपया डॉक्टरांचा सल्ला घ्या। विश्वसनीयता: {confidence:.1%}",
                'tamil': f"உங்கள் அறிகுறிகள் மிதமான உடல்நல அபாயத்தைக் குறிக்கின்றன. தயவுசெய்து மருத்துவரை அணுகவும். நம்பகத்தன்மை: {confidence:.1%}"
            },
            'Red': {
                'english': f"Your symptoms suggest high health risk. Seek immediate medical attention! Confidence: {confidence:.1%}",
                'hindi': f"आपके लक्षण उच्च स्वास्थ्य जोखिम दर्शाते हैं। तुरंत चिकित्सा सहायता लें! विश्वसनीयता: {confidence:.1%}",
                'marathi': f"तुमची लक्षणे उच्च आरोग्य धोका दर्शवतात। ताबडतोब वैद्यकीय मदत घ्या! विश्वसनीयता: {confidence:.1%}",
                'tamil': f"உங்கள் அறிகுறிகள் அதிக உடல்நல அபாயத்தைக் குறிக்கின்றன. உடனடியாக மருத்துவ உதவி பெறுங்கள்! நம்பகத்தன்மை: {confidence:.1%}"
            }
        }
        
        return explanations.get(risk_level, explanations['Yellow'])
    
    def _get_recommendations(self, risk_level: str) -> Dict[str, List[str]]:
        """Get recommendations based on risk level"""
        
        recommendations = {
            'Green': {
                'english': ["Rest well", "Drink plenty of water", "Monitor symptoms"],
                'hindi': ["अच्छी तरह आराम करें", "भरपूर पानी पिएं", "लक्षणों पर नजर रखें"],
                'marathi': ["चांगली विश्रांती घ्या", "भरपूर पाणी प्या", "लक्षणांवर लक्ष ठेवा"],
                'tamil': ["நன்றாக ஓய்வு எடுங்கள்", "நிறைய தண்ணீர் குடியுங்கள்", "அறிகுறிகளைக் கண்காணியுங்கள்"]
            },
            'Yellow': {
                'english': ["Consult doctor within 24 hours", "Take prescribed medicines", "Avoid crowded places"],
                'hindi': ["24 घंटे में डॉक्टर से मिलें", "दवाइयां लें", "भीड़भाड़ से बचें"],
                'marathi': ["24 तासांत डॉक्टरांना भेटा", "औषधे घ्या", "गर्दीपासून दूर राहा"],
                'tamil': ["24 மணி நேரத்திற்குள் மருத்துவரை அணுகவும்", "பரிந்துரைக்கப்பட்ட மருந்துகளை எடுத்துக்கொள்ளுங்கள்", "கூட்டமான இடங்களைத் தவிர்க்கவும்"]
            },
            'Red': {
                'english': ["Go to hospital immediately", "Call emergency services", "Don't delay treatment"],
                'hindi': ["तुरंत अस्पताल जाएं", "आपातकालीन सेवा बुलाएं", "इलाज में देरी न करें"],
                'marathi': ["ताबडतोब रुग्णालयात जा", "आपत्कालीन सेवा बोलवा", "उपचारात विलंब करू नका"],
                'tamil': ["உடனடியாக மருத்துவமனைக்குச் செல்லுங்கள்", "அவசர சேவைகளை அழையுங்கள்", "சிகிச்சையை தாமதப்படுத்த வேண்டாம்"]
            }
        }
        
        return recommendations.get(risk_level, recommendations['Yellow'])
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic training data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        data = []
        symptoms = list(self.symptom_mapping.keys())
        
        for i in range(n_samples):
            # Random symptom combination
            n_symptoms = np.random.randint(1, 6)
            patient_symptoms = np.random.choice(symptoms, n_symptoms, replace=False)
            
            # Create feature vector
            feature_vector = np.zeros(len(symptoms))
            for symptom in patient_symptoms:
                idx = symptoms.index(symptom)
                feature_vector[idx] = 1
            
            # Add demographic features
            age = np.random.randint(18, 80)
            gender = np.random.choice([0, 1])  # 0: Female, 1: Male
            
            # Determine risk level based on symptoms
            symptom_count = len(patient_symptoms)
            severity_symptoms = ['breathing_difficulty', 'chest_pain', 'vomiting']
            has_severe = any(s in patient_symptoms for s in severity_symptoms)
            
            if has_severe or symptom_count >= 4:
                risk = 'Red'
            elif symptom_count >= 2:
                risk = 'Yellow'
            else:
                risk = 'Green'
            
            # Create row
            row = list(feature_vector) + [age, gender, risk, f'patient_{i}']
            data.append(row)
        
        columns = symptoms + ['age', 'gender', 'risk_level', 'patient_id']
        return pd.DataFrame(data, columns=columns)
    
    def save_model(self, model_path: str = 'health_risk_model.joblib'):
        """Save trained model for offline use"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'symptom_mapping': self.symptom_mapping
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str = 'health_risk_model.joblib'):
        """Load pre-trained model for inference"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.symptom_mapping = model_data['symptom_mapping']
        print(f"Model loaded from {model_path}")

if __name__ == "__main__":
    # Demo usage
    classifier = HealthRiskClassifier()
    
    # Train model
    print("Training health risk classification model...")
    metrics = classifier.train()
    print(f"Model accuracy: {metrics['accuracy']:.3f}")
    
    # Save model
    classifier.save_model()
    
    # Test prediction with multiple languages
    test_cases = [
        (['बुखार', 'खांसी', 'सिरदर्द'], 'Hindi'),  # Hindi symptoms
        (['காய்ச்சல்', 'இருமல்', 'தலைவலி'], 'Tamil'),  # Tamil symptoms
        (['ताप', 'खोकला', 'डोकेदुखी'], 'Marathi')  # Marathi symptoms
    ]
    
    for symptoms, lang in test_cases:
        result = classifier.predict_risk(symptoms, age=35, gender='M')
        print(f"\nTest Prediction ({lang}):")
        print(f"Symptoms: {symptoms}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Confidence: {result['confidence']:.1%}")
        
        # Show explanation in the respective language
        lang_key = lang.lower()
        if lang_key in result['explanation']:
            print(f"Explanation ({lang}): {result['explanation'][lang_key]}")
        else:
            print(f"Explanation (English): {result['explanation']['english']}")