# AI-Sanjivani: Offline AI Healthcare Assistant for Rural India

## Overview
AI-Sanjivani is an offline-capable AI healthcare assistant designed for rural India, enabling ASHA workers and villagers to assess health risks using simple symptom input via text and voice in Hindi/Marathi.

## Features
- **Risk Prediction**: Green/Yellow/Red health risk classification
- **Multilingual Support**: Hindi/Marathi voice and text input
- **Offline Capability**: On-device ML inference
- **Simple UI**: Designed for low-literacy users
- **PHC Dashboard**: Village-level disease heatmap
- **Explainable Results**: Simple, non-medical language explanations

## Architecture
```
ai-sanjivani/
├── ml_engine/          # Core ML models and inference
├── mobile_app/         # Flutter mobile application
├── dashboard/          # Streamlit PHC dashboard
├── data/              # Training data and models
├── api/               # Backend API services
└── deployment/        # Docker and deployment configs
```

## Tech Stack
- **ML Engine**: Python, scikit-learn, TensorFlow Lite
- **Mobile App**: Flutter with offline capabilities
- **Dashboard**: Streamlit for PHC visualization
- **Database**: SQLite for offline storage
- **Voice**: Speech-to-text with Hindi/Marathi support

## Quick Start
1. Set up ML engine: `cd ml_engine && pip install -r requirements.txt`
2. Train models: `python train_models.py`
3. Run dashboard: `cd dashboard && streamlit run app.py`
4. Build mobile app: `cd mobile_app && flutter build apk`

## Social Impact
Designed for low-connectivity, low-end devices to democratize healthcare access in rural India.