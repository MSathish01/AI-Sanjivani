# 🏥 AI-Sanjivani: Offline AI Healthcare Assistant for Rural India

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Firebase-orange)](https://sample-firebase-ai-app-cca91.web.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/MSathish01/AI-Sanjivani)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **AI-powered offline healthcare assistant designed for rural India, supporting ASHA workers and villagers with multilingual health risk assessment.**

## 🌐 Live Demo

**🔗 [https://sample-firebase-ai-app-cca91.web.app](https://sample-firebase-ai-app-cca91.web.app)**

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌍 **Multilingual Support** | Hindi, Marathi, Tamil, English |
| 📴 **Offline Capable** | Works without internet connectivity |
| 🎯 **Risk Classification** | Green/Yellow/Red health risk levels |
| 🎤 **Voice Input** | Speech-to-text for symptom collection |
| 📊 **PHC Dashboard** | Village-level disease heatmap |
| 💬 **Explainable AI** | Simple, non-medical language explanations |
| 📱 **Mobile Optimized** | Designed for low-end Android devices |
| 🚨 **Emergency Contacts** | Quick dial 108, 104, 102 |

## 🏗️ Architecture

```
ai-sanjivani/
├── ml_engine/           # Core ML models and inference
│   ├── models/          # Health risk classifier
│   ├── speech_engine.py # Multilingual voice processing
│   └── train_model.py   # Model training pipeline
├── api/                 # Flask REST API
│   ├── app.py          # API endpoints
│   └── templates/      # Web interface
├── public/             # Firebase hosted static files
│   ├── index.html      # Main web app
│   └── app.js          # Client-side inference
├── dashboard/          # Streamlit PHC dashboard
│   └── app.py          # Analytics dashboard
├── mobile_app/         # Flutter mobile application
│   └── lib/            # Dart source files
└── deployment/         # Docker and deployment configs
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/MSathish01/AI-Sanjivani.git
cd AI-Sanjivani
```

### 2. Install Dependencies
```bash
pip install -r ml_engine/requirements.txt
```

### 3. Run ML Engine Demo
```bash
python demo_ai_sanjivani.py
```

### 4. Start Flask API
```bash
python api/app.py
# Access at http://localhost:5000
```

### 5. Start Streamlit Dashboard
```bash
streamlit run dashboard/app.py
# Access at http://localhost:8501
```

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| ML Engine | Python, scikit-learn, TensorFlow Lite |
| Web API | Flask, Flask-CORS |
| Frontend | HTML5, Bootstrap 5, JavaScript |
| Dashboard | Streamlit, Plotly |
| Mobile App | Flutter, Dart |
| Database | SQLite (offline storage) |
| Hosting | Firebase Hosting |
| Voice | SpeechRecognition, Web Speech API |

## 🌐 Supported Languages

| Language | Code | Sample Symptoms |
|----------|------|-----------------|
| English | `en` | fever, cough, headache |
| Hindi | `hi` | बुखार, खांसी, सिरदर्द |
| Marathi | `mr` | ताप, खोकला, डोकेदुखी |
| Tamil | `ta` | காய்ச்சல், இருமல், தலைவலி |

## 📊 Risk Levels

| Level | Color | Action Required |
|-------|-------|-----------------|
| 🟢 Green | Low Risk | Rest, hydrate, monitor |
| 🟡 Yellow | Moderate Risk | Consult doctor within 24 hours |
| 🔴 Red | High Risk | Seek immediate medical attention |

## 📱 API Endpoints

```
GET  /api/health          # Health check
POST /api/assess          # Health risk assessment
POST /api/extract-symptoms # Extract symptoms from text
GET  /api/symptoms        # Get supported symptoms list
```

### Example API Request
```bash
curl -X POST https://your-api.com/api/assess \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["fever", "cough", "headache"],
    "age": 35,
    "gender": "M",
    "language": "hindi"
  }'
```

## 🎯 Social Impact

- **Democratizes Healthcare**: Brings AI-powered health assessment to rural areas
- **Supports ASHA Workers**: Empowers frontline health workers with technology
- **Works Offline**: Functions in areas with poor internet connectivity
- **Multilingual**: Communicates in local languages for better understanding
- **Low-End Device Support**: Optimized for basic smartphones

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | >85% |
| Inference Time | <100ms |
| Model Size | ~2MB (TFLite) |
| Memory Usage | <100MB |

## 🚨 Emergency Contacts (India)

| Service | Number |
|---------|--------|
| Emergency | 108 |
| Health Helpline | 104 |
| Ambulance | 102 |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**M. Sathish**
- GitHub: [@MSathish01](https://github.com/MSathish01)

## 🙏 Acknowledgments

- Designed for rural healthcare workers in India
- Inspired by the need for accessible healthcare technology
- Built with ❤️ for social impact

---

<p align="center">
  <b>🏥 AI-Sanjivani - Healthcare for Every Village</b><br>
  <i>ग्रामीण भारत के लिए AI स्वास्थ्य सहायक</i>
</p>