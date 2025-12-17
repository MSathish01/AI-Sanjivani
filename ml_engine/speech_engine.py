"""
AI-Sanjivani Speech Recognition Engine
Handles Hindi/Marathi/Tamil voice input for symptom collection
Optimized for offline operation and low-end devices
"""

import speech_recognition as sr
import numpy as np
from typing import List, Dict, Optional
import json
import re
from googletrans import Translator
import logging

class MultilingualSpeechEngine:
    """
    Speech recognition engine supporting Hindi/Marathi/Tamil/English
    Designed for rural healthcare workers with limited tech literacy
    """
    
    def __init__(self, offline_mode: bool = True):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.offline_mode = offline_mode
        self.translator = None if offline_mode else Translator()
        
        # Load offline language mappings
        self.language_patterns = self._load_language_patterns()
        
        # Configure recognizer for noisy environments
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_language_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load offline language pattern mappings"""
        return {
            'symptoms': {
                'fever': {
                    'hindi': ['बुखार', 'तेज बुखार', 'हल्का बुखार', 'ताप'],
                    'marathi': ['ताप', 'तापमान', 'ज्वर'],
                    'tamil': ['காய்ச்சல்', 'உடல் சூடு', 'ஜுரம்', 'தாபம்'],
                    'english': ['fever', 'temperature', 'hot', 'burning']
                },
                'cough': {
                    'hindi': ['खांसी', 'सूखी खांसी', 'कफ वाली खांसी'],
                    'marathi': ['खोकला', 'कोरडा खोकला', 'कफाचा खोकला'],
                    'tamil': ['இருமல்', 'வறட்டு இருமல்', 'கபம் இருமல்'],
                    'english': ['cough', 'dry cough', 'wet cough']
                },
                'headache': {
                    'hindi': ['सिरदर्द', 'सिर में दर्द', 'माथा दुखना'],
                    'marathi': ['डोकेदुखी', 'डोक्यात दुखी', 'माथा दुखणे'],
                    'tamil': ['தலைவலி', 'தலை வேदனை', 'மைக்ரேன்'],
                    'english': ['headache', 'head pain', 'migraine']
                },
                'body_ache': {
                    'hindi': ['शरीर दर्द', 'बदन दर्द', 'अंग दुखना'],
                    'marathi': ['अंग दुखी', 'शरीर दुखी', 'हाडे दुखणे'],
                    'tamil': ['உடல் வலி', 'மூட்டு வலி', 'தசை வலி'],
                    'english': ['body ache', 'muscle pain', 'joint pain']
                },
                'nausea': {
                    'hindi': ['मतली', 'जी मिचलाना', 'उबकाई'],
                    'marathi': ['मळमळाट', 'जी मळमळणे'],
                    'tamil': ['குமட்டல்', 'வாந்தி உணர்வு', 'மயக்கம்'],
                    'english': ['nausea', 'feeling sick', 'queasy']
                },
                'diarrhea': {
                    'hindi': ['दस्त', 'पेट खराब', 'लूज मोशन'],
                    'marathi': ['जुलाब', 'पोट खराब', 'अतिसार'],
                    'tamil': ['வயிற்றுப்போக்கு', 'லூஸ் மோஷன்', 'வயிறு கெட்டது'],
                    'english': ['diarrhea', 'loose motion', 'stomach upset']
                },
                'vomiting': {
                    'hindi': ['उल्टी', 'कै', 'वमन'],
                    'marathi': ['उलटी', 'वमन', 'ओकारणे'],
                    'tamil': ['வாந்தி', 'ஓக்காளிப்பு', 'வமனம்'],
                    'english': ['vomiting', 'throwing up', 'puking']
                },
                'weakness': {
                    'hindi': ['कमजोरी', 'थकान', 'अशक्तता'],
                    'marathi': ['अशक्तपणा', 'दुर्बलता', 'थकवा'],
                    'tamil': ['பலவீனம்', 'சோர்வு', 'களைப்பு'],
                    'english': ['weakness', 'fatigue', 'tiredness']
                },
                'breathing_difficulty': {
                    'hindi': ['सांस लेने में तकलीफ', 'दम फूलना', 'श्वास कष्ट'],
                    'marathi': ['श्वास घेण्यात त्रास', 'दम लागणे', 'श्वासोच्छवास'],
                    'tamil': ['மூச்சு விடுவதில் சிரமம்', 'மூச்சு திணறல்', 'ஆஸ்துமா'],
                    'english': ['breathing difficulty', 'shortness of breath', 'breathless']
                },
                'chest_pain': {
                    'hindi': ['छाती में दर्द', 'सीने में दर्द', 'हृदय में दर्द'],
                    'marathi': ['छातीत दुखी', 'हृदयात दुखी'],
                    'tamil': ['மார்பு வலி', 'இதய வலி', 'நெஞ்சு வலி'],
                    'english': ['chest pain', 'heart pain', 'chest discomfort']
                }
            },
            'common_phrases': {
                'yes': {
                    'hindi': ['हां', 'जी हां', 'हाँ', 'सही'],
                    'marathi': ['होय', 'हो', 'बरोबर'],
                    'tamil': ['ஆம்', 'ஓம்', 'சரி', 'ஆமாம்'],
                    'english': ['yes', 'yeah', 'correct', 'right']
                },
                'no': {
                    'hindi': ['नहीं', 'ना', 'गलत'],
                    'marathi': ['नाही', 'ना', 'चुकीचे'],
                    'tamil': ['இல்லை', 'வேண்டாம்', 'தவறு'],
                    'english': ['no', 'nope', 'wrong', 'incorrect']
                }
            }
        }
    
    def calibrate_microphone(self) -> bool:
        """
        Calibrate microphone for ambient noise
        Important for rural environments with background noise
        """
        try:
            with self.microphone as source:
                self.logger.info("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                self.logger.info(f"Microphone calibrated. Energy threshold: {self.recognizer.energy_threshold}")
                return True
        except Exception as e:
            self.logger.error(f"Microphone calibration failed: {e}")
            return False
    
    def listen_for_symptoms(self, timeout: int = 10, phrase_time_limit: int = 5) -> Optional[str]:
        """
        Listen for symptom input with timeout
        Returns raw audio text or None if failed
        """
        try:
            with self.microphone as source:
                self.logger.info("Listening for symptoms... Speak now!")
                
                # Listen with timeout
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
                
                # Recognize speech
                if self.offline_mode:
                    # Use offline recognition (limited but works without internet)
                    text = self.recognizer.recognize_sphinx(audio)
                else:
                    # Use Google Speech Recognition (better accuracy, needs internet)
                    text = self.recognizer.recognize_google(
                        audio, 
                        language='hi-IN'  # Hindi India
                    )
                
                self.logger.info(f"Recognized text: {text}")
                return text.lower().strip()
                
        except sr.WaitTimeoutError:
            self.logger.warning("Listening timeout - no speech detected")
            return None
        except sr.UnknownValueError:
            self.logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in speech recognition: {e}")
            return None
    
    def extract_symptoms_from_text(self, text: str) -> List[str]:
        """
        Extract symptoms from recognized text
        Handles multilingual input and fuzzy matching
        """
        if not text:
            return []
        
        detected_symptoms = []
        text_lower = text.lower()
        
        # Check each symptom pattern
        for symptom, languages in self.language_patterns['symptoms'].items():
            for lang, patterns in languages.items():
                for pattern in patterns:
                    if pattern.lower() in text_lower:
                        if symptom not in detected_symptoms:
                            detected_symptoms.append(symptom)
                        break
        
        return detected_symptoms
    
    def interactive_symptom_collection(self) -> Dict[str, any]:
        """
        Interactive symptom collection with voice guidance
        Returns collected symptoms and metadata
        """
        collected_symptoms = []
        session_data = {
            'symptoms': [],
            'confidence_scores': [],
            'language_detected': 'mixed',
            'session_duration': 0
        }
        
        # Calibrate microphone
        if not self.calibrate_microphone():
            return session_data
        
        # Welcome message (would be played as audio in full implementation)
        welcome_messages = {
            'hindi': "नमस्ते! कृपया अपने लक्षण बताएं। मैं सुन रहा हूं।",
            'marathi': "नमस्कार! कृपया तुमची लक्षणे सांगा. मी ऐकत आहे.",
            'english': "Hello! Please tell me your symptoms. I'm listening."
        }
        
        print("AI-Sanjivani: " + welcome_messages['hindi'])
        print("AI-Sanjivani: " + welcome_messages['english'])
        
        # Collect symptoms in multiple rounds
        max_rounds = 3
        for round_num in range(max_rounds):
            print(f"\nRound {round_num + 1}: Please describe your symptoms...")
            
            # Listen for symptoms
            recognized_text = self.listen_for_symptoms()
            
            if recognized_text:
                # Extract symptoms
                round_symptoms = self.extract_symptoms_from_text(recognized_text)
                
                if round_symptoms:
                    # Add new symptoms
                    for symptom in round_symptoms:
                        if symptom not in collected_symptoms:
                            collected_symptoms.append(symptom)
                    
                    print(f"Detected symptoms: {round_symptoms}")
                    
                    # Ask for confirmation
                    print("Did I understand correctly? Say 'yes' or 'no'")
                    confirmation = self.listen_for_symptoms(timeout=5)
                    
                    if confirmation and self._is_positive_response(confirmation):
                        session_data['symptoms'].extend(round_symptoms)
                    
                else:
                    print("No symptoms detected. Please try again.")
            
            # Ask if there are more symptoms
            if round_num < max_rounds - 1:
                print("Any other symptoms? Say 'no' if done.")
                more_symptoms = self.listen_for_symptoms(timeout=5)
                
                if more_symptoms and self._is_negative_response(more_symptoms):
                    break
        
        session_data['symptoms'] = list(set(collected_symptoms))  # Remove duplicates
        return session_data
    
    def _is_positive_response(self, text: str) -> bool:
        """Check if response is positive (yes)"""
        if not text:
            return False
        
        text_lower = text.lower()
        for lang_patterns in self.language_patterns['common_phrases']['yes'].values():
            for pattern in lang_patterns:
                if pattern.lower() in text_lower:
                    return True
        return False
    
    def _is_negative_response(self, text: str) -> bool:
        """Check if response is negative (no)"""
        if not text:
            return False
        
        text_lower = text.lower()
        for lang_patterns in self.language_patterns['common_phrases']['no'].values():
            for pattern in lang_patterns:
                if pattern.lower() in text_lower:
                    return True
        return False
    
    def text_to_speech_guidance(self, message: str, language: str = 'hindi') -> str:
        """
        Generate voice guidance messages
        In full implementation, this would use TTS
        """
        guidance_templates = {
            'hindi': {
                'welcome': "नमस्ते! AI-संजीवनी में आपका स्वागत है। कृपया अपने लक्षण बताएं।",
                'listening': "मैं सुन रहा हूं... कृपया बोलें।",
                'not_understood': "मैं समझ नहीं पाया। कृपया दोबारा बोलें।",
                'confirm': "क्या यह सही है?",
                'more_symptoms': "क्या कोई और लक्षण हैं?",
                'thank_you': "धन्यवाद! आपकी जांच हो रही है।"
            },
            'marathi': {
                'welcome': "नमस्कार! AI-संजीवनी मध्ये तुमचे स्वागत आहे। कृपया तुमची लक्षणे सांगा.",
                'listening': "मी ऐकत आहे... कृपया बोला.",
                'not_understood': "मला समजले नाही. कृपया पुन्हा बोला.",
                'confirm': "हे बरोबर आहे का?",
                'more_symptoms': "आणखी काही लक्षणे आहेत का?",
                'thank_you': "धन्यवाद! तुमची तपासणी होत आहे."
            },
            'tamil': {
                'welcome': "வணக்கம்! AI-சஞ்சீவனியில் உங்களை வரவேற்கிறோம். உங்கள் அறிகுறிகளைச் சொல்லுங்கள்.",
                'listening': "நான் கேட்டுக்கொண்டிருக்கிறேன்... தயவுசெய்து பேசுங்கள்.",
                'not_understood': "எனக்குப் புரியவில்லை. தயவுசெய்து மீண்டும் சொல்லுங்கள்.",
                'confirm': "இது சரியா?",
                'more_symptoms': "வேறு ஏதேனும் அறிகுறிகள் உள்ளதா?",
                'thank_you': "நன்றி! உங்கள் பரிசோதனை நடைபெறுகிறது."
            },
            'english': {
                'welcome': "Welcome to AI-Sanjivani! Please tell me your symptoms.",
                'listening': "I'm listening... Please speak.",
                'not_understood': "I didn't understand. Please speak again.",
                'confirm': "Is this correct?",
                'more_symptoms': "Are there any other symptoms?",
                'thank_you': "Thank you! Analyzing your symptoms."
            }
        }
        
        return guidance_templates.get(language, guidance_templates['english']).get(message, message)

# Demo usage
if __name__ == "__main__":
    # Initialize speech engine
    speech_engine = MultilingualSpeechEngine(offline_mode=True)
    
    print("AI-Sanjivani Speech Engine Demo")
    print("=" * 40)
    
    # Test text-based symptom extraction
    test_texts = [
        "मुझे बुखार और खांसी है",  # Hindi
        "मला ताप आणि खोकला आहे",   # Marathi
        "எனக்கு காய்ச்சல் மற்றும் இருமல் உள்ளது",  # Tamil
        "I have fever and cough"     # English
    ]
    
    for text in test_texts:
        symptoms = speech_engine.extract_symptoms_from_text(text)
        print(f"Text: {text}")
        print(f"Extracted symptoms: {symptoms}")
        print()
    
    # Interactive demo (uncomment to test with microphone)
    # print("Starting interactive symptom collection...")
    # result = speech_engine.interactive_symptom_collection()
    # print(f"Session result: {result}")