"""
AI-Sanjivani Clinical Explanation Engine
Provides simple, culturally appropriate health explanations
Designed for low-literacy rural populations
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class ClinicalExplainer:
    """
    Generates simple, culturally appropriate health explanations
    Translates medical concepts into everyday language
    """
    
    def __init__(self):
        # Load explanation templates
        self.explanation_templates = self._load_explanation_templates()
        self.symptom_explanations = self._load_symptom_explanations()
        self.risk_level_explanations = self._load_risk_level_explanations()
        self.cultural_context = self._load_cultural_context()
    
    def _load_explanation_templates(self) -> Dict[str, Dict[str, str]]:
        """Load explanation templates for different languages"""
        return {
            'english': {
                'greeting': "Based on your symptoms, here's what we found:",
                'risk_intro': "Your health risk level is:",
                'symptoms_detected': "We noticed these symptoms:",
                'recommendations': "Here's what you should do:",
                'when_to_seek_help': "Seek immediate help if:",
                'general_advice': "General health advice:",
                'disclaimer': "This is not a replacement for professional medical advice."
            },
            'hindi': {
                'greeting': "आपके लक्षणों के आधार पर, यहाँ हमारी जांच है:",
                'risk_intro': "आपका स्वास्थ्य जोखिम स्तर है:",
                'symptoms_detected': "हमने ये लक्षण देखे हैं:",
                'recommendations': "आपको यह करना चाहिए:",
                'when_to_seek_help': "तुरंत मदद लें अगर:",
                'general_advice': "सामान्य स्वास्थ्य सलाह:",
                'disclaimer': "यह पेशेवर चिकित्सा सलाह का विकल्प नहीं है।"
            },
            'marathi': {
                'greeting': "तुमच्या लक्षणांच्या आधारे, आमची तपासणी:",
                'risk_intro': "तुमचा आरोग्य धोका पातळी आहे:",
                'symptoms_detected': "आम्ही ही लक्षणे पाहिली:",
                'recommendations': "तुम्ही हे करावे:",
                'when_to_seek_help': "ताबडतोब मदत घ्या जर:",
                'general_advice': "सामान्य आरोग्य सल्ला:",
                'disclaimer': "हे व्यावसायिक वैद्यकीय सल्ल्याचा पर्याय नाही।"
            },
            'tamil': {
                'greeting': "உங்கள் அறிகுறிகளின் அடிப்படையில், எங்கள் பரிசோதனை:",
                'risk_intro': "உங்கள் உடல்நல அபாய நிலை:",
                'symptoms_detected': "நாங்கள் இந்த அறிகுறிகளைக் கண்டோம்:",
                'recommendations': "நீங்கள் இதைச் செய்ய வேண்டும்:",
                'when_to_seek_help': "உடனடியாக உதவி பெறுங்கள்:",
                'general_advice': "பொதுவான உடல்நல ஆலோசனை:",
                'disclaimer': "இது தொழில்முறை மருத்துவ ஆலோசனைக்கு மாற்றாக இல்லை."
            }
        }
    
    def _load_symptom_explanations(self) -> Dict[str, Dict[str, str]]:
        """Load simple explanations for symptoms"""
        return {
            'fever': {
                'english': "Your body temperature is higher than normal. This usually means your body is fighting an infection.",
                'hindi': "आपके शरीर का तापमान सामान्य से अधिक है। यह आमतौर पर संक्रमण से लड़ने का संकेत है।",
                'marathi': "तुमच्या शरीराचे तापमान सामान्यापेक्षा जास्त आहे. हे सहसा संसर्गाशी लढण्याचे लक्षण आहे.",
                'tamil': "உங்கள் உடல் வெப்பநிலை சாதாரணத்தை விட அதிகமாக உள்ளது. இது பொதுவாக உங்கள் உடல் தொற்றுநோயுடன் போராடுவதைக் குறிக்கிறது."
            },
            'cough': {
                'english': "Coughing helps clear your throat and lungs. It can be due to cold, dust, or infection.",
                'hindi': "खांसी आपके गले और फेफड़ों को साफ करने में मदद करती है। यह सर्दी, धूल या संक्रमण के कारण हो सकती है।",
                'marathi': "खोकला तुमचा गळा आणि फुफ्फुसे स्वच्छ करण्यास मदत करतो. हे सर्दी, धूळ किंवा संसर्गामुळे होऊ शकते.",
                'tamil': "இருமல் உங்கள் தொண்டை மற்றும் நுரையீரலை சுத்தம் செய்ய உதவுகிறது. இது சளி, தூசி அல்லது தொற்றுநோய் காரணமாக இருக்கலாம்."
            },
            'headache': {
                'english': "Head pain can be caused by stress, lack of sleep, dehydration, or illness.",
                'hindi': "सिर दर्द तनाव, नींद की कमी, पानी की कमी या बीमारी के कारण हो सकता है।",
                'marathi': "डोकेदुखी तणाव, झोपेची कमतरता, पाण्याची कमतरता किंवा आजारामुळे होऊ शकते.",
                'tamil': "தலைவலி மன அழுத்தம், தூக்கமின்மை, நீரிழப்பு அல்லது நோய் காரணமாக ஏற்படலாம்."
            },
            'breathing_difficulty': {
                'english': "Trouble breathing is serious. It means your lungs or heart may need immediate attention.",
                'hindi': "सांस लेने में तकलीफ गंभीर है। इसका मतलब है कि आपके फेफड़े या दिल को तुरंत ध्यान की जरूरत हो सकती है।",
                'marathi': "श्वास घेण्यात अडचण गंभीर आहे. याचा अर्थ तुमच्या फुफ्फुसांना किंवा हृदयाला तातडीने लक्ष देण्याची गरज असू शकते.",
                'tamil': "மூச்சு விடுவதில் சிரமம் தீவிரமானது. உங்கள் நுரையீரல் அல்லது இதயத்திற்கு உடனடி கவனம் தேவைப்படலாம்."
            },
            'chest_pain': {
                'english': "Chest pain can be serious, especially if it affects your heart. Don't ignore it.",
                'hindi': "छाती में दर्द गंभीर हो सकता है, खासकर अगर यह आपके दिल को प्रभावित करता है। इसे नजरअंदाज न करें।",
                'marathi': "छातीत दुखी गंभीर असू शकते, विशेषतः जर ते तुमच्या हृदयावर परिणाम करत असेल. याकडे दुर्लक्ष करू नका.",
                'tamil': "மார்பு வலி தீவிரமானதாக இருக்கலாம், குறிப்பாக அது உங்கள் இதயத்தை பாதிக்கும் போது. அதை புறக்கணிக்காதீர்கள்."
            },
            'vomiting': {
                'english': "Vomiting can lead to dehydration. It may be due to food poisoning, infection, or other illness.",
                'hindi': "उल्टी से पानी की कमी हो सकती है। यह खाना खराब होने, संक्रमण या अन्य बीमारी के कारण हो सकती है।",
                'marathi': "उलटीमुळे पाण्याची कमतरता होऊ शकते. हे अन्न विषबाधा, संसर्ग किंवा इतर आजारामुळे होऊ शकते.",
                'tamil': "வாந்தி நீரிழப்புக்கு வழிவகுக்கும். இது உணவு விஷம், தொற்றுநோய் அல்லது பிற நோய் காரணமாக இருக்கலாம்."
            },
            'diarrhea': {
                'english': "Loose motions can cause dehydration quickly. Drink plenty of clean water and ORS.",
                'hindi': "दस्त से जल्दी पानी की कमी हो सकती है। भरपूर साफ पानी और ORS पिएं।",
                'marathi': "जुलाबामुळे लवकर पाण्याची कमतरता होऊ शकते. भरपूर स्वच्छ पाणी आणि ORS प्या.",
                'tamil': "வயிற்றுப்போக்கு விரைவில் நீரிழப்பை ஏற்படுத்தும். நிறைய சுத்தமான தண்ணீர் மற்றும் ORS குடியுங்கள்."
            }
        }
    
    def _load_risk_level_explanations(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Load detailed risk level explanations"""
        return {
            'Green': {
                'meaning': {
                    'english': "Low risk - Your symptoms are mild and not immediately concerning.",
                    'hindi': "कम जोखिम - आपके लक्षण हल्के हैं और तुरंत चिंता की बात नहीं है।",
                    'marathi': "कमी धोका - तुमची लक्षणे सौम्य आहेत आणि तातडीने चिंताजनक नाहीत.",
                    'tamil': "குறைந்த அபாயம் - உங்கள் அறிகுறிகள் லேசானவை மற்றும் உடனடியாக கவலைக்குரியவை அல்ல."
                },
                'action': {
                    'english': "Rest, drink water, and monitor your symptoms. See a doctor if they worsen.",
                    'hindi': "आराम करें, पानी पिएं, और अपने लक्षणों पर नजर रखें। अगर बिगड़ें तो डॉक्टर से मिलें।",
                    'marathi': "विश्रांती घ्या, पाणी प्या आणि तुमच्या लक्षणांवर लक्ष ठेवा. जर ती वाढली तर डॉक्टरांना भेटा.",
                    'tamil': "ஓய்வு எடுங்கள், தண்ணீர் குடியுங்கள், உங்கள் அறிகுறிகளைக் கண்காணியுங்கள். அவை மோசமடைந்தால் மருத்துவரைப் பாருங்கள்."
                }
            },
            'Yellow': {
                'meaning': {
                    'english': "Moderate risk - Your symptoms need medical attention within 24 hours.",
                    'hindi': "मध्यम जोखिम - आपके लक्षणों को 24 घंटे में चिकित्सा ध्यान की जरूरत है।",
                    'marathi': "मध्यम धोका - तुमच्या लक्षणांना 24 तासांत वैद्यकीय लक्ष देण्याची गरज आहे.",
                    'tamil': "மிதமான அபாயம் - உங்கள் அறிகுறிகளுக்கு 24 மணி நேரத்திற்குள் மருத்துவ கவனம் தேவை."
                },
                'action': {
                    'english': "Visit a doctor or health center today. Don't wait for symptoms to get worse.",
                    'hindi': "आज ही डॉक्टर या स्वास्थ्य केंद्र जाएं। लक्षणों के बिगड़ने का इंतजार न करें।",
                    'marathi': "आजच डॉक्टर किंवा आरोग्य केंद्राला भेट द्या. लक्षणे वाढण्याची वाट पाहू नका.",
                    'tamil': "இன்றே மருத்துவர் அல்லது சுகாதார மையத்தைப் பார்வையிடுங்கள். அறிகுறிகள் மோசமாகும் வரை காத்திருக்காதீர்கள்."
                }
            },
            'Red': {
                'meaning': {
                    'english': "High risk - Your symptoms are serious and need immediate medical attention.",
                    'hindi': "उच्च जोखिम - आपके लक्षण गंभीर हैं और तुरंत चिकित्सा ध्यान की जरूरत है।",
                    'marathi': "उच्च धोका - तुमची लक्षणे गंभीर आहेत आणि तातडीने वैद्यकीय लक्ष देण्याची गरज आहे.",
                    'tamil': "அதிக அபாயம் - உங்கள் அறிகுறிகள் தீவிரமானவை மற்றும் உடனடி மருத்துவ கவனம் தேவை."
                },
                'action': {
                    'english': "Go to the nearest hospital immediately. Call 108 for ambulance if needed.",
                    'hindi': "तुरंत नजदीकी अस्पताल जाएं। जरूरत हो तो एम्बुलेंस के लिए 108 पर कॉल करें।",
                    'marathi': "ताबडतोब जवळच्या रुग्णालयात जा. गरज असल्यास रुग्णवाहिकेसाठी 108 वर कॉल करा.",
                    'tamil': "உடனடியாக அருகிலுள்ள மருத்துவமனைக்குச் செல்லுங்கள். தேவைப்பட்டால் ஆம்புலன்ஸுக்கு 108 ஐ அழையுங்கள்."
                }
            }
        }
    
    def _load_cultural_context(self) -> Dict[str, Any]:
        """Load culturally appropriate context and advice"""
        return {
            'home_remedies': {
                'fever': {
                    'english': "Apply cool cloth to forehead, drink plenty of fluids",
                    'hindi': "माथे पर ठंडा कपड़ा रखें, भरपूर तरल पदार्थ पिएं",
                    'marathi': "कपाळावर थंड कापड ठेवा, भरपूर द्रव प्या",
                    'tamil': "நெற்றியில் குளிர்ந்த துணியை வைக்கவும், நிறைய திரவங்களை குடிக்கவும்"
                },
                'cough': {
                    'english': "Warm water with honey and ginger can help soothe throat",
                    'hindi': "शहद और अदरक के साथ गर्म पानी गले को आराम दे सकता है",
                    'marathi': "मध आणि आले सह कोमट पाणी गळ्याला आराम देऊ शकते",
                    'tamil': "தேன் மற்றும் இஞ்சியுடன் வெதுவெதுப்பான நீர் தொண்டையை ஆற்ற உதவும்"
                },
                'headache': {
                    'english': "Rest in a quiet, dark room. Gentle head massage may help",
                    'hindi': "शांत, अंधेरे कमरे में आराम करें। हल्की सिर की मालिश मदद कर सकती है",
                    'marathi': "शांत, अंधारकोठडीत विश्रांती घ्या. हलकी डोक्याची मालिश मदत करू शकते",
                    'tamil': "அமைதியான, இருண்ட அறையில் ஓய்வு எடுங்கள். மென்மையான தலை மசாஜ் உதவக்கூடும்"
                }
            },
            'when_to_worry': {
                'english': [
                    "Difficulty breathing or chest pain",
                    "High fever that won't come down",
                    "Severe vomiting or diarrhea",
                    "Signs of dehydration (dry mouth, no urination)",
                    "Confusion or unusual behavior"
                ],
                'hindi': [
                    "सांस लेने में तकलीफ या छाती में दर्द",
                    "तेज बुखार जो कम नहीं हो रहा",
                    "गंभीर उल्टी या दस्त",
                    "पानी की कमी के संकेत (सूखा मुंह, पेशाब न आना)",
                    "भ्रम या असामान्य व्यवहार"
                ],
                'marathi': [
                    "श्वास घेण्यात अडचण किंवा छातीत दुखी",
                    "उच्च ताप जो कमी होत नाही",
                    "गंभीर उलटी किंवा जुलाब",
                    "पाण्याची कमतरतेची चिन्हे (कोरडे तोंड, लघवी न होणे)",
                    "गोंधळ किंवा असामान्य वर्तन"
                ],
                'tamil': [
                    "மூச்சு விடுவதில் சிரமம் அல்லது மார்பு வலி",
                    "குறையாத அதிக காய்ச்சல்",
                    "கடுமையான வாந்தி அல்லது வயிற்றுப்போக்கு",
                    "நீரிழப்பின் அறிகுறிகள் (வறண்ட வாய், சிறுநீர் இல்லாமை)",
                    "குழப்பம் அல்லது அசாதாரண நடத்தை"
                ]
            }
        }
    
    def generate_comprehensive_explanation(self, 
                                         symptoms: List[str], 
                                         risk_level: str, 
                                         confidence: float,
                                         age: int = 30,
                                         gender: str = 'M',
                                         language: str = 'english') -> Dict[str, Any]:
        """Generate comprehensive, culturally appropriate explanation"""
        
        templates = self.explanation_templates[language]
        risk_info = self.risk_level_explanations[risk_level]
        
        explanation = {
            'greeting': templates['greeting'],
            'risk_assessment': {
                'level': risk_level,
                'confidence': f"{confidence:.1%}",
                'meaning': risk_info['meaning'][language],
                'recommended_action': risk_info['action'][language]
            },
            'symptoms_analysis': self._explain_symptoms(symptoms, language),
            'recommendations': self._generate_recommendations(symptoms, risk_level, age, language),
            'warning_signs': self._get_warning_signs(language),
            'home_care': self._get_home_care_advice(symptoms, language),
            'disclaimer': templates['disclaimer']
        }
        
        return explanation
    
    def _explain_symptoms(self, symptoms: List[str], language: str) -> List[Dict[str, str]]:
        """Explain each symptom in simple terms"""
        explanations = []
        
        for symptom in symptoms:
            if symptom in self.symptom_explanations:
                explanations.append({
                    'symptom': symptom,
                    'explanation': self.symptom_explanations[symptom][language]
                })
        
        return explanations
    
    def _generate_recommendations(self, symptoms: List[str], risk_level: str, 
                                age: int, language: str) -> List[str]:
        """Generate specific recommendations based on symptoms and risk"""
        
        recommendations = []
        
        # Risk-based recommendations
        if risk_level == 'Green':
            recommendations.extend([
                self._get_text('rest_and_monitor', language),
                self._get_text('stay_hydrated', language),
                self._get_text('avoid_stress', language)
            ])
        elif risk_level == 'Yellow':
            recommendations.extend([
                self._get_text('see_doctor_24h', language),
                self._get_text('monitor_closely', language),
                self._get_text('avoid_crowds', language)
            ])
        else:  # Red
            recommendations.extend([
                self._get_text('seek_immediate_help', language),
                self._get_text('call_emergency', language),
                self._get_text('dont_delay', language)
            ])
        
        # Age-specific recommendations
        if age >= 60:
            recommendations.append(self._get_text('elderly_care', language))
        
        # Symptom-specific recommendations
        if 'fever' in symptoms:
            recommendations.append(self._get_text('fever_care', language))
        
        if 'vomiting' in symptoms or 'diarrhea' in symptoms:
            recommendations.append(self._get_text('dehydration_prevention', language))
        
        return recommendations
    
    def _get_warning_signs(self, language: str) -> List[str]:
        """Get warning signs that require immediate attention"""
        return self.cultural_context['when_to_worry'][language]
    
    def _get_home_care_advice(self, symptoms: List[str], language: str) -> List[str]:
        """Get appropriate home care advice"""
        advice = []
        
        for symptom in symptoms:
            if symptom in self.cultural_context['home_remedies']:
                advice.append(self.cultural_context['home_remedies'][symptom][language])
        
        return advice
    
    def _get_text(self, key: str, language: str) -> str:
        """Get localized text for common phrases"""
        
        common_phrases = {
            'rest_and_monitor': {
                'english': "Rest well and monitor your symptoms",
                'hindi': "अच्छी तरह आराम करें और लक्षणों पर नजर रखें",
                'marathi': "चांगली विश्रांती घ्या आणि लक्षणांवर लक्ष ठेवा",
                'tamil': "நன்றாக ஓய்வு எடுத்து உங்கள் அறிகுறிகளைக் கண்காணியுங்கள்"
            },
            'stay_hydrated': {
                'english': "Drink plenty of clean water",
                'hindi': "भरपूर साफ पानी पिएं",
                'marathi': "भरपूर स्वच्छ पाणी प्या",
                'tamil': "நிறைய சுத்தமான தண்ணீர் குடியுங்கள்"
            },
            'see_doctor_24h': {
                'english': "See a doctor within 24 hours",
                'hindi': "24 घंटे में डॉक्टर से मिलें",
                'marathi': "24 तासांत डॉक्टरांना भेटा",
                'tamil': "24 மணி நேரத்திற்குள் மருத்துவரைப் பாருங்கள்"
            },
            'seek_immediate_help': {
                'english': "Seek immediate medical help",
                'hindi': "तुरंत चिकित्सा सहायता लें",
                'marathi': "ताबडतोब वैद्यकीय मदत घ्या",
                'tamil': "உடனடியாக மருத்துவ உதவி பெறுங்கள்"
            },
            'call_emergency': {
                'english': "Call 108 for emergency services",
                'hindi': "आपातकालीन सेवाओं के लिए 108 पर कॉल करें",
                'marathi': "आपत्कालीन सेवांसाठी 108 वर कॉल करा",
                'tamil': "அவசர சேவைகளுக்கு 108 ஐ அழையுங்கள்"
            },
            'elderly_care': {
                'english': "Extra care needed due to age - monitor closely",
                'hindi': "उम्र के कारण अतिरिक्त देखभाल की जरूरत - बारीकी से निगरानी करें",
                'marathi': "वयामुळे अतिरिक्त काळजी आवश्यक - बारकाईने निरीक्षण करा",
                'tamil': "வயது காரணமாக கூடுதல் கவனம் தேவை - நெருக்கமாக கண்காணிக்கவும்"
            },
            'fever_care': {
                'english': "Keep body cool, use wet cloth on forehead",
                'hindi': "शरीर को ठंडा रखें, माथे पर गीला कपड़ा रखें",
                'marathi': "शरीर थंड ठेवा, कपाळावर ओले कापड ठेवा",
                'tamil': "உடலை குளிர்ச்சியாக வைத்திருங்கள், நெற்றியில் ஈரமான துணியைப் பயன்படுத்துங்கள்"
            },
            'dehydration_prevention': {
                'english': "Drink ORS solution to prevent dehydration",
                'hindi': "निर्जलीकरण को रोकने के लिए ORS घोल पिएं",
                'marathi': "निर्जलीकरण टाळण्यासाठी ORS द्रावण प्या",
                'tamil': "நீரிழப்பைத் தடுக்க ORS கரைசலைக் குடியுங்கள்"
            }
        }
        
        return common_phrases.get(key, {}).get(language, key)
    
    def format_for_voice_output(self, explanation: Dict[str, Any], language: str = 'english') -> str:
        """Format explanation for text-to-speech output"""
        
        voice_text = []
        
        # Greeting and risk level
        voice_text.append(explanation['greeting'])
        voice_text.append(f"{explanation['risk_assessment']['meaning']}")
        
        # Key recommendations (max 3 for voice)
        voice_text.append("यहाँ मुख्य सुझाव हैं:" if language == 'hindi' else "Here are the key recommendations:")
        for rec in explanation['recommendations'][:3]:
            voice_text.append(rec)
        
        # Warning if high risk
        if explanation['risk_assessment']['level'] == 'Red':
            voice_text.append(explanation['risk_assessment']['recommended_action'])
        
        return " ".join(voice_text)

def main():
    """Demo clinical explainer"""
    explainer = ClinicalExplainer()
    
    # Test explanation generation
    test_cases = [
        {
            'symptoms': ['fever', 'cough'],
            'risk_level': 'Yellow',
            'confidence': 0.75,
            'age': 35,
            'gender': 'M'
        },
        {
            'symptoms': ['chest_pain', 'breathing_difficulty'],
            'risk_level': 'Red',
            'confidence': 0.90,
            'age': 60,
            'gender': 'M'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print("=" * 30)
        
        for language in ['english', 'hindi', 'tamil']:
            explanation = explainer.generate_comprehensive_explanation(
                case['symptoms'], case['risk_level'], case['confidence'],
                case['age'], case['gender'], language
            )
            
            print(f"\n{language.upper()} Explanation:")
            print(f"Risk: {explanation['risk_assessment']['meaning']}")
            print(f"Action: {explanation['risk_assessment']['recommended_action']}")
            
            # Voice format
            voice_output = explainer.format_for_voice_output(explanation, language)
            print(f"Voice Output: {voice_output[:100]}...")

if __name__ == "__main__":
    main()