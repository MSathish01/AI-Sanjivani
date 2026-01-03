// AI-Sanjivani Client-Side Health Assessment
// Works offline with on-device inference

let selectedSymptoms = [];
let currentLang = 'english';

const symptoms = ['fever','cough','headache','body_ache','nausea','diarrhea','vomiting','weakness','breathing_difficulty','chest_pain'];

const symptomLabels = {
    english: {fever:'Fever',cough:'Cough',headache:'Headache',body_ache:'Body Ache',nausea:'Nausea',diarrhea:'Diarrhea',vomiting:'Vomiting',weakness:'Weakness',breathing_difficulty:'Breathing Difficulty',chest_pain:'Chest Pain'},
    hindi: {fever:'बुखार',cough:'खांसी',headache:'सिरदर्द',body_ache:'शरीर दर्द',nausea:'मतली',diarrhea:'दस्त',vomiting:'उल्टी',weakness:'कमजोरी',breathing_difficulty:'सांस की तकलीफ',chest_pain:'छाती में दर्द'},
    marathi: {fever:'ताप',cough:'खोकला',headache:'डोकेदुखी',body_ache:'अंग दुखी',nausea:'मळमळाट',diarrhea:'जुलाब',vomiting:'उलटी',weakness:'अशक्तपणा',breathing_difficulty:'श्वास त्रास',chest_pain:'छातीत दुखी'},
    tamil: {fever:'காய்ச்சல்',cough:'இருமல்',headache:'தலைவலி',body_ache:'உடல் வலி',nausea:'குமட்டல்',diarrhea:'வயிற்றுப்போக்கு',vomiting:'வாந்தி',weakness:'பலவீனம்',breathing_difficulty:'மூச்சு திணறல்',chest_pain:'மார்பு வலி'}
};

const explanations = {
    Green: {
        english: "Your symptoms suggest low health risk. Rest well and stay hydrated.",
        hindi: "आपके लक्षण कम स्वास्थ्य जोखिम दर्शाते हैं। आराम करें और पानी पिएं।",
        marathi: "तुमची लक्षणे कमी आरोग्य धोका दर्शवतात। विश्रांती घ्या आणि पाणी प्या.",
        tamil: "உங்கள் அறிகுறிகள் குறைந்த உடல்நல அபாயத்தைக் குறிக்கின்றன. ஓய்வு எடுங்கள்."
    },
    Yellow: {
        english: "Your symptoms suggest moderate health risk. Please consult a doctor within 24 hours.",
        hindi: "आपके लक्षण मध्यम स्वास्थ्य जोखिम दर्शाते हैं। 24 घंटे में डॉक्टर से मिलें।",
        marathi: "तुमची लक्षणे मध्यम आरोग्य धोका दर्शवतात। 24 तासांत डॉक्टरांना भेटा.",
        tamil: "உங்கள் அறிகுறிகள் மிதமான அபாயத்தைக் குறிக்கின்றன. 24 மணி நேரத்திற்குள் மருத்துவரை அணுகவும்."
    },
    Red: {
        english: "Your symptoms suggest HIGH health risk! Seek immediate medical attention!",
        hindi: "आपके लक्षण उच्च स्वास्थ्य जोखिम दर्शाते हैं! तुरंत चिकित्सा सहायता लें!",
        marathi: "तुमची लक्षणे उच्च आरोग्य धोका दर्शवतात! ताबडतोब वैद्यकीय मदत घ्या!",
        tamil: "உங்கள் அறிகுறிகள் அதிக அபாயத்தைக் குறிக்கின்றன! உடனடியாக மருத்துவ உதவி பெறுங்கள்!"
    }
};

const recommendations = {
    Green: {
        english: ["Rest well", "Drink plenty of water", "Monitor symptoms for 2-3 days"],
        hindi: ["अच्छी तरह आराम करें", "भरपूर पानी पिएं", "2-3 दिन लक्षणों पर नजर रखें"],
        marathi: ["चांगली विश्रांती घ्या", "भरपूर पाणी प्या", "2-3 दिवस लक्षणांवर लक्ष ठेवा"],
        tamil: ["நன்றாக ஓய்வு எடுங்கள்", "நிறைய தண்ணீர் குடியுங்கள்", "2-3 நாட்கள் கண்காணியுங்கள்"]
    },
    Yellow: {
        english: ["Consult doctor within 24 hours", "Take prescribed medicines", "Avoid crowded places"],
        hindi: ["24 घंटे में डॉक्टर से मिलें", "दवाइयां लें", "भीड़भाड़ से बचें"],
        marathi: ["24 तासांत डॉक्टरांना भेटा", "औषधे घ्या", "गर्दीपासून दूर राहा"],
        tamil: ["24 மணி நேரத்திற்குள் மருத்துவரை அணுகவும்", "மருந்துகளை எடுங்கள்", "கூட்டத்தைத் தவிர்க்கவும்"]
    },
    Red: {
        english: ["Go to hospital IMMEDIATELY", "Call emergency: 108", "Don't delay treatment"],
        hindi: ["तुरंत अस्पताल जाएं", "आपातकालीन: 108 पर कॉल करें", "इलाज में देरी न करें"],
        marathi: ["ताबडतोब रुग्णालयात जा", "आपत्कालीन: 108 वर कॉल करा", "उपचारात विलंब करू नका"],
        tamil: ["உடனடியாக மருத்துவமனைக்குச் செல்லுங்கள்", "அவசர: 108 அழைக்கவும்", "சிகிச்சையை தாமதிக்காதீர்கள்"]
    }
};

// Severity weights for symptoms
const severityWeights = {
    fever: 1, cough: 1, headache: 1, body_ache: 1, nausea: 1.5,
    diarrhea: 1.5, vomiting: 2, weakness: 1, breathing_difficulty: 3, chest_pain: 3
};

function renderSymptoms() {
    const container = document.getElementById('symptomButtons');
    container.innerHTML = symptoms.map(s => 
        `<button class="btn btn-outline-secondary symptom-btn ${selectedSymptoms.includes(s)?'active':''}" onclick="toggleSymptom('${s}')">
            ${symptomLabels[currentLang][s]}
        </button>`
    ).join('');
}

function toggleSymptom(s) {
    if(selectedSymptoms.includes(s)) {
        selectedSymptoms = selectedSymptoms.filter(x => x !== s);
    } else {
        selectedSymptoms.push(s);
    }
    renderSymptoms();
}

function setLanguage(lang) {
    currentLang = lang;
    document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    renderSymptoms();
}

function assessHealth() {
    if(selectedSymptoms.length === 0) {
        alert(currentLang === 'hindi' ? 'कृपया कम से कम एक लक्षण चुनें' : 
              currentLang === 'tamil' ? 'குறைந்தது ஒரு அறிகுறியைத் தேர்ந்தெடுக்கவும்' :
              'Please select at least one symptom');
        return;
    }
    
    document.getElementById('loading').style.display = 'block';
    document.getElementById('resultCard').style.display = 'none';
    
    // Simulate processing delay for UX
    setTimeout(() => {
        const result = performLocalAssessment();
        document.getElementById('loading').style.display = 'none';
        showResult(result);
    }, 1500);
}

function performLocalAssessment() {
    // Calculate risk score based on symptoms
    let riskScore = 0;
    let maxScore = 0;
    
    selectedSymptoms.forEach(s => {
        riskScore += severityWeights[s] || 1;
    });
    
    // Age factor
    const age = parseInt(document.getElementById('age').value) || 30;
    if(age > 60) riskScore *= 1.3;
    else if(age < 5) riskScore *= 1.2;
    
    // Determine risk level
    let riskLevel, confidence;
    
    const hasSevere = selectedSymptoms.some(s => ['breathing_difficulty', 'chest_pain'].includes(s));
    
    if(hasSevere || riskScore >= 6) {
        riskLevel = 'Red';
        confidence = Math.min(0.95, 0.7 + (riskScore / 20));
    } else if(riskScore >= 3 || selectedSymptoms.length >= 3) {
        riskLevel = 'Yellow';
        confidence = Math.min(0.85, 0.5 + (riskScore / 15));
    } else {
        riskLevel = 'Green';
        confidence = Math.min(0.8, 0.6 + (selectedSymptoms.length / 10));
    }
    
    return { riskLevel, confidence, symptoms: selectedSymptoms };
}

function showResult(data) {
    const badge = document.getElementById('riskBadge');
    badge.textContent = data.riskLevel;
    badge.className = 'badge fs-2 p-3 mb-3 risk-' + data.riskLevel.toLowerCase();
    
    document.getElementById('confidence').textContent = 
        `${currentLang === 'hindi' ? 'विश्वसनीयता' : 'Confidence'}: ${(data.confidence * 100).toFixed(0)}%`;
    
    document.getElementById('explanation').textContent = explanations[data.riskLevel][currentLang];
    
    const recList = document.getElementById('recommendations');
    recList.innerHTML = recommendations[data.riskLevel][currentLang]
        .map(r => `<li class="list-group-item"><i class="fas fa-check-circle text-success me-2"></i>${r}</li>`)
        .join('');
    
    document.getElementById('emergencyAlert').style.display = data.riskLevel === 'Red' ? 'block' : 'none';
    document.getElementById('resultCard').style.display = 'block';
    
    // Scroll to result
    document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth' });
}

// Initialize
renderSymptoms();