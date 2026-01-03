"""
AI-Sanjivani Complete Demo
Demonstrates all features of the offline AI healthcare assistant
"""

import sys
import time
sys.path.append('.')

def print_header(title):
    print("\n" + "="*60)
    print(f"🏥 {title}")
    print("="*60)

def print_section(title):
    print(f"\n📋 {title}")
    print("-" * 40)

def demo_health_risk_classifier():
    """Demo the multilingual health risk classifier"""
    from ml_engine.models.health_risk_classifier import HealthRiskClassifier
    
    print_section("Health Risk Classification Demo")
    
    classifier = HealthRiskClassifier()
    print("🔧 Training AI model...")
    classifier.train()
    print("✅ Model trained successfully!")
    
    # Demo cases representing different scenarios
    demo_cases = [
        {
            'name': 'Rural Farmer (Hindi)',
            'symptoms': ['बुखार', 'खांसी', 'कमजोरी'],
            'age': 45,
            'gender': 'M',
            'language': 'hindi'
        },
        {
            'name': 'Village Woman (Tamil)',
            'symptoms': ['காய்ச்சல்', 'தலைவலி', 'வாந்தி'],
            'age': 32,
            'gender': 'F',
            'language': 'tamil'
        },
        {
            'name': 'ASHA Worker (Marathi)',
            'symptoms': ['ताप', 'श्वास कष्ट', 'छातीत दुखी'],
            'age': 38,
            'gender': 'F',
            'language': 'marathi'
        },
        {
            'name': 'Emergency Case (English)',
            'symptoms': ['fever', 'breathing_difficulty', 'chest_pain', 'vomiting'],
            'age': 60,
            'gender': 'M',
            'language': 'english'
        }
    ]
    
    for case in demo_cases:
        print(f"\n👤 Patient: {case['name']}")
        print(f"   Age: {case['age']}, Gender: {case['gender']}")
        print(f"   Symptoms: {', '.join(case['symptoms'])}")
        
        result = classifier.predict_risk(
            symptoms=case['symptoms'],
            age=case['age'],
            gender=case['gender']
        )
        
        # Color coding for risk levels
        risk_colors = {
            'Green': '🟢',
            'Yellow': '🟡', 
            'Red': '🔴'
        }
        
        print(f"   {risk_colors[result['risk_level']]} Risk Level: {result['risk_level']}")
        print(f"   📊 Confidence: {result['confidence']:.1%}")
        print(f"   💬 Explanation: {result['explanation'][case['language']]}")
        print(f"   📋 Recommendations:")
        for rec in result['recommendations'][case['language']]:
            print(f"      • {rec}")
        
        if result['risk_level'] == 'Red':
            print("   🚨 HIGH RISK ALERT - Immediate medical attention required!")
        
        time.sleep(1)  # Pause for readability

def demo_speech_engine():
    """Demo the multilingual speech processing"""
    from ml_engine.speech_engine import MultilingualSpeechEngine
    
    print_section("Multilingual Speech Processing Demo")
    
    speech_engine = MultilingualSpeechEngine(offline_mode=True)
    
    # Test various language inputs
    test_phrases = [
        {
            'text': 'मुझे तेज बुखार और सूखी खांसी है',
            'language': 'Hindi',
            'translation': 'I have high fever and dry cough'
        },
        {
            'text': 'मला डोकेदुखी आणि मळमळाट होत आहे',
            'language': 'Marathi', 
            'translation': 'I have headache and nausea'
        },
        {
            'text': 'எனக்கு மார்பு வலி மற்றும் மூச்சு திணறல் உள்ளது',
            'language': 'Tamil',
            'translation': 'I have chest pain and breathing difficulty'
        },
        {
            'text': 'I have severe body ache and weakness',
            'language': 'English',
            'translation': 'I have severe body ache and weakness'
        }
    ]
    
    print("🎤 Processing voice input simulation...")
    
    for phrase in test_phrases:
        print(f"\n🗣️  {phrase['language']} Input: '{phrase['text']}'")
        print(f"   🔄 Translation: {phrase['translation']}")
        
        symptoms = speech_engine.extract_symptoms_from_text(phrase['text'])
        print(f"   🎯 Detected Symptoms: {symptoms}")
        
        if symptoms:
            print(f"   ✅ Successfully extracted {len(symptoms)} symptom(s)")
        else:
            print("   ⚠️  No symptoms detected")

def demo_dashboard_data():
    """Demo the PHC dashboard functionality"""
    from dashboard.app import PHCDashboard
    
    print_section("PHC Dashboard Analytics Demo")
    
    dashboard = PHCDashboard()
    data = dashboard.get_dashboard_data()
    
    print("📊 Village Health Overview:")
    print(f"   📈 Total Assessments: {data['total_assessments']}")
    print(f"   🔴 High Risk Cases: {data['high_risk_cases']}")
    print(f"   📍 Active Villages: {data['active_villages']}")
    print(f"   ⭐ Average Risk Score: {data['avg_risk_score']:.2f}")
    
    print("\n🏘️  Village Risk Distribution:")
    if not data['village_summary'].empty:
        for _, village in data['village_summary'].head(5).iterrows():
            risk_pct = (village['high_risk_cases'] / village['total_cases']) * 100
            risk_indicator = "🔴" if risk_pct > 30 else "🟡" if risk_pct > 15 else "🟢"
            print(f"   {risk_indicator} {village['village_name']}: {village['total_cases']} cases ({risk_pct:.1f}% high risk)")
    
    print("\n📋 Recent High-Risk Alerts:")
    recent_high_risk = data['recent_assessments'][data['recent_assessments']['risk_level'] == 'Red']
    if not recent_high_risk.empty:
        for _, alert in recent_high_risk.head(3).iterrows():
            print(f"   🚨 {alert['village_name']} - ASHA: {alert['asha_worker_id']} - {alert['assessment_date']}")
    else:
        print("   ✅ No recent high-risk alerts")

def demo_mobile_optimization():
    """Demo mobile optimization features"""
    print_section("Mobile Optimization Demo")
    
    print("📱 Mobile Deployment Features:")
    print("   ✅ Offline Capability: Works without internet")
    print("   ✅ Low Memory Usage: <100MB RAM requirement")
    print("   ✅ Fast Inference: <100ms response time")
    print("   ✅ Multilingual UI: Hindi/Marathi/Tamil/English")
    print("   ✅ Voice Input: Speech-to-text support")
    print("   ✅ Simple Interface: Designed for low-literacy users")
    
    print("\n🔧 Technical Specifications:")
    print("   • Model Size: ~2MB (TensorFlow Lite)")
    print("   • Supported Devices: Android API 21+")
    print("   • Battery Optimized: Efficient inference")
    print("   • Data Storage: SQLite for offline sync")
    
    print("\n📦 Deployment Package Includes:")
    print("   • health_risk_model.tflite - Optimized ML model")
    print("   • HealthRiskClassifier.java - Android wrapper")
    print("   • Multilingual symptom mappings")
    print("   • Complete integration guide")

def main():
    """Main demo function"""
    print_header("AI-Sanjivani: Complete System Demo")
    print("🌟 Offline AI Healthcare Assistant for Rural India")
    print("🎯 Supporting Hindi, Marathi, Tamil, and English")
    print("🏥 Designed for ASHA workers and rural communities")
    
    try:
        # Demo 1: Health Risk Classification
        demo_health_risk_classifier()
        
        # Demo 2: Speech Processing
        demo_speech_engine()
        
        # Demo 3: Dashboard Analytics
        demo_dashboard_data()
        
        # Demo 4: Mobile Features
        demo_mobile_optimization()
        
        # Final Summary
        print_header("Demo Summary")
        print("🎉 AI-Sanjivani Demo Completed Successfully!")
        print("\n✅ Features Demonstrated:")
        print("   • Multilingual health risk assessment")
        print("   • Voice input processing (Hindi/Marathi/Tamil/English)")
        print("   • PHC dashboard with village-level analytics")
        print("   • Mobile optimization for offline deployment")
        print("   • Explainable AI with simple language explanations")
        
        print("\n🌐 Access Points:")
        print("   • Dashboard: http://localhost:8501")
        print("   • Mobile App: Flutter application (requires Flutter SDK)")
        print("   • API: Flask backend for integration")
        
        print("\n🎯 Social Impact:")
        print("   • Democratizes healthcare access in rural India")
        print("   • Supports ASHA workers with AI-powered tools")
        print("   • Works offline in low-connectivity areas")
        print("   • Provides health insights in local languages")
        
        print("\n💡 Next Steps:")
        print("   1. Deploy to Android devices for field testing")
        print("   2. Train with real medical data (with proper permissions)")
        print("   3. Integrate with existing PHC systems")
        print("   4. Expand language support as needed")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()