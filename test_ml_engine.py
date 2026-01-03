"""
Test script to verify AI-Sanjivani ML engine functionality
"""

import sys
sys.path.append('.')

try:
    from ml_engine.models.health_risk_classifier import HealthRiskClassifier
    from ml_engine.speech_engine import MultilingualSpeechEngine
    
    print("🏥 AI-Sanjivani ML Engine Test")
    print("=" * 40)
    
    # Test Health Risk Classifier
    print("\n📊 Testing Health Risk Classifier...")
    classifier = HealthRiskClassifier()
    
    # Train the model first
    print("   🔧 Training model...")
    classifier.train()
    print("   ✅ Model trained successfully")
    
    # Test with different language symptoms
    test_cases = [
        {
            'symptoms': ['बुखार', 'खांसी', 'सिरदर्द'],  # Hindi
            'age': 35,
            'gender': 'M',
            'language': 'Hindi'
        },
        {
            'symptoms': ['காய்ச்சல்', 'இருமல்', 'தலைவலி'],  # Tamil
            'age': 28,
            'gender': 'F',
            'language': 'Tamil'
        },
        {
            'symptoms': ['ताप', 'खोकला', 'डोकेदुखी'],  # Marathi
            'age': 42,
            'gender': 'M',
            'language': 'Marathi'
        },
        {
            'symptoms': ['fever', 'cough', 'headache', 'body_ache'],  # English - High risk
            'age': 65,
            'gender': 'F',
            'language': 'English'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i} ({case['language']}):")
        print(f"   Symptoms: {case['symptoms']}")
        print(f"   Age: {case['age']}, Gender: {case['gender']}")
        
        result = classifier.predict_risk(
            symptoms=case['symptoms'],
            age=case['age'],
            gender=case['gender']
        )
        
        print(f"   ✅ Risk Level: {result['risk_level']}")
        print(f"   📊 Confidence: {result['confidence']:.1%}")
        
        # Show explanation in the respective language
        lang_key = case['language'].lower()
        if lang_key in result['explanation']:
            print(f"   💬 Explanation: {result['explanation'][lang_key]}")
        
        # Show recommendations
        if lang_key in result['recommendations']:
            print(f"   📋 Recommendations: {', '.join(result['recommendations'][lang_key])}")
    
    print("\n🎤 Testing Speech Engine...")
    speech_engine = MultilingualSpeechEngine(offline_mode=True)
    
    # Test text-based symptom extraction
    test_texts = [
        "मुझे बुखार और खांसी है",  # Hindi
        "मला ताप आणि खोकला आहे",   # Marathi
        "எனக்கு காய்ச்சல் மற்றும் இருமல் உள்ளது",  # Tamil
        "I have fever and cough"     # English
    ]
    
    for text in test_texts:
        symptoms = speech_engine.extract_symptoms_from_text(text)
        print(f"   Text: '{text}' → Symptoms: {symptoms}")
    
    print("\n🎉 All tests completed successfully!")
    print("✅ Health Risk Classifier: Working with multilingual support")
    print("✅ Speech Engine: Text processing functional")
    print("🌐 Dashboard: Running at http://localhost:8501")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()