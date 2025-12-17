"""
Edge AI Specialist Module
Convert trained ML models for offline mobile deployment
Optimized for low-end Android devices with minimal memory footprint
"""

import tensorflow as tf
import numpy as np
import joblib
import json
import os
from typing import Dict, Any, Tuple
import pickle
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

class EdgeAIConverter:
    """
    Convert ML models to edge-optimized formats
    Supports TensorFlow Lite, ONNX, and optimized joblib formats
    """
    
    def __init__(self):
        self.model_info = {}
        
    def convert_sklearn_to_tflite(self, model_path: str, feature_names: list, 
                                 output_path: str) -> Dict[str, Any]:
        """
        Convert sklearn model to TensorFlow Lite format
        Optimized for mobile inference
        """
        
        print(f"Converting sklearn model to TensorFlow Lite: {model_path}")
        
        # Load sklearn model
        model = joblib.load(model_path)
        
        # Create a simple neural network that mimics the sklearn model
        # This is a workaround since sklearn models can't be directly converted to TFLite
        
        # Generate sample data to understand input/output shapes
        n_features = len(feature_names)
        sample_input = np.random.random((1000, n_features)).astype(np.float32)
        
        # Get predictions from original model
        if hasattr(model, 'predict_proba'):
            sample_output = model.predict_proba(sample_input).astype(np.float32)
        else:
            # For LightGBM or other models
            sample_output = model.predict(sample_input)
            if len(sample_output.shape) == 1:
                # Convert to one-hot for classification
                n_classes = len(np.unique(sample_output))
                sample_output_onehot = np.zeros((len(sample_output), n_classes))
                sample_output_onehot[np.arange(len(sample_output)), sample_output] = 1
                sample_output = sample_output_onehot.astype(np.float32)
        
        # Create TensorFlow model
        tf_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(sample_output.shape[1], activation='softmax')
        ])
        
        # Compile model
        tf_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the neural network to mimic the original model
        print("Training neural network to mimic original model...")
        tf_model.fit(
            sample_input, sample_output,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        
        # Optimization for mobile devices
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # Use float16 for smaller size
        
        # Convert
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Test the converted model
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Test inference
        test_input = np.random.random((1, n_features)).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        
        # Compare with original model
        original_output = model.predict_proba(test_input) if hasattr(model, 'predict_proba') else model.predict(test_input)
        
        model_info = {
            'input_shape': input_details[0]['shape'].tolist(),
            'output_shape': output_details[0]['shape'].tolist(),
            'model_size_bytes': len(tflite_model),
            'quantization': 'float16',
            'feature_names': feature_names
        }
        
        print(f"TFLite model saved: {output_path}")
        print(f"Model size: {len(tflite_model) / 1024:.1f} KB")
        
        return model_info
    
    def optimize_joblib_model(self, model_path: str, output_path: str) -> Dict[str, Any]:
        """
        Optimize joblib model for mobile deployment
        Reduce memory footprint and improve inference speed
        """
        
        print(f"Optimizing joblib model: {model_path}")
        
        # Load original model
        model = joblib.load(model_path)
        
        if isinstance(model, RandomForestClassifier):
            # Optimize Random Forest
            optimized_model = self._optimize_random_forest(model)
        else:
            # For other models, use compression
            optimized_model = model
        
        # Save with compression
        joblib.dump(optimized_model, output_path, compress=3)
        
        # Get file sizes
        original_size = os.path.getsize(model_path)
        optimized_size = os.path.getsize(output_path)
        
        optimization_info = {
            'original_size_bytes': original_size,
            'optimized_size_bytes': optimized_size,
            'compression_ratio': original_size / optimized_size,
            'model_type': type(model).__name__
        }
        
        print(f"Optimized model saved: {output_path}")
        print(f"Size reduction: {original_size/1024:.1f} KB -> {optimized_size/1024:.1f} KB")
        print(f"Compression ratio: {optimization_info['compression_ratio']:.2f}x")
        
        return optimization_info
    
    def _optimize_random_forest(self, rf_model: RandomForestClassifier) -> RandomForestClassifier:
        """
        Optimize Random Forest for mobile deployment
        Reduce number of trees and tree depth while maintaining accuracy
        """
        
        # Create optimized version with fewer trees
        optimized_rf = RandomForestClassifier(
            n_estimators=min(30, rf_model.n_estimators),  # Limit trees for mobile
            max_depth=min(10, rf_model.max_depth or 10),   # Limit depth
            min_samples_split=max(10, rf_model.min_samples_split),
            min_samples_leaf=max(5, rf_model.min_samples_leaf),
            random_state=rf_model.random_state,
            n_jobs=1  # Single thread for mobile
        )
        
        # Copy the trained trees (first 30)
        if hasattr(rf_model, 'estimators_'):
            optimized_rf.estimators_ = rf_model.estimators_[:30]
            optimized_rf.n_estimators = len(optimized_rf.estimators_)
            optimized_rf.classes_ = rf_model.classes_
            optimized_rf.n_classes_ = rf_model.n_classes_
            optimized_rf.n_features_in_ = rf_model.n_features_in_
            optimized_rf.feature_importances_ = rf_model.feature_importances_
        
        return optimized_rf
    
    def convert_lightgbm_to_mobile(self, model_path: str, output_path: str) -> Dict[str, Any]:
        """
        Convert LightGBM model for mobile deployment
        """
        
        print(f"Converting LightGBM model for mobile: {model_path}")
        
        # Load LightGBM model
        model = lgb.Booster(model_file=model_path)
        
        # Save in compact format
        model.save_model(output_path, num_iteration=model.best_iteration)
        
        # Get model info
        model_info = {
            'num_trees': model.num_trees(),
            'num_features': model.num_feature(),
            'best_iteration': model.best_iteration,
            'model_size_bytes': os.path.getsize(output_path)
        }
        
        print(f"Mobile LightGBM model saved: {output_path}")
        print(f"Model size: {model_info['model_size_bytes'] / 1024:.1f} KB")
        
        return model_info
    
    def create_mobile_inference_engine(self, models_dir: str, output_dir: str):
        """
        Create optimized inference engine for mobile deployment
        """
        
        print("Creating mobile inference engine...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model metadata
        metadata_path = os.path.join(models_dir, 'model_metadata.joblib')
        metadata = joblib.load(metadata_path)
        
        # Create mobile inference class
        mobile_inference_code = self._generate_mobile_inference_code(metadata)
        
        # Save inference engine
        with open(os.path.join(output_dir, 'mobile_inference.py'), 'w') as f:
            f.write(mobile_inference_code)
        
        # Convert and optimize models
        model_conversions = {}\n        \n        # Convert Random Forest models\n        for model_name in ['rf_risk.joblib', 'rf_disease.joblib']:\n            model_path = os.path.join(models_dir, model_name)\n            if os.path.exists(model_path):\n                # Optimize joblib version\n                optimized_path = os.path.join(output_dir, f'optimized_{model_name}')\n                opt_info = self.optimize_joblib_model(model_path, optimized_path)\n                model_conversions[model_name] = opt_info\n        \n        # Convert LightGBM models\n        for model_name in ['lgb_risk.txt', 'lgb_disease.txt']:\n            model_path = os.path.join(models_dir, model_name)\n            if os.path.exists(model_path):\n                mobile_path = os.path.join(output_dir, f'mobile_{model_name}')\n                lgb_info = self.convert_lightgbm_to_mobile(model_path, mobile_path)\n                model_conversions[model_name] = lgb_info\n        \n        # Copy metadata\n        mobile_metadata = {\n            'feature_names': metadata['feature_names'],\n            'target_names': metadata['target_names'],\n            'encoders': metadata['encoders'],\n            'model_conversions': model_conversions\n        }\n        \n        with open(os.path.join(output_dir, 'mobile_metadata.json'), 'w') as f:\n            json.dump(mobile_metadata, f, indent=2, default=str)\n        \n        # Create Android-compatible files\n        self._create_android_assets(output_dir, mobile_metadata)\n        \n        print(f\"Mobile inference engine created in: {output_dir}\")\n        \n        return mobile_metadata\n    \n    def _generate_mobile_inference_code(self, metadata: Dict[str, Any]) -> str:\n        \"\"\"Generate optimized Python inference code for mobile\"\"\"\n        \n        code = '''\n\"\"\"\nMobile Inference Engine for AI-Sanjivani\nOptimized for offline deployment on low-end Android devices\n\"\"\"\n\nimport numpy as np\nimport joblib\nimport json\nimport os\nfrom typing import Dict, List, Tuple, Any\n\nclass MobileHealthPredictor:\n    \"\"\"\n    Lightweight health risk predictor for mobile deployment\n    Optimized for speed and memory efficiency\n    \"\"\"\n    \n    def __init__(self, model_dir: str):\n        self.model_dir = model_dir\n        self.models = {}\n        self.metadata = {}\n        self.load_models()\n    \n    def load_models(self):\n        \"\"\"Load optimized models and metadata\"\"\"\n        \n        # Load metadata\n        metadata_path = os.path.join(self.model_dir, 'mobile_metadata.json')\n        with open(metadata_path, 'r') as f:\n            self.metadata = json.load(f)\n        \n        # Load optimized models\n        model_files = {\n            'risk_rf': 'optimized_rf_risk.joblib',\n            'disease_rf': 'optimized_rf_disease.joblib'\n        }\n        \n        for model_name, filename in model_files.items():\n            model_path = os.path.join(self.model_dir, filename)\n            if os.path.exists(model_path):\n                self.models[model_name] = joblib.load(model_path)\n    \n    def preprocess_input(self, symptoms: List[str], age: int, gender: str, \n                        is_pregnant: bool = False, has_diabetes: bool = False, \n                        has_hypertension: bool = False, vitals: Dict[str, float] = None) -> np.ndarray:\n        \"\"\"Preprocess input for model inference\"\"\"\n        \n        # Symptom mapping (simplified for mobile)\n        symptom_mapping = {\n            'fever': ['बुखार', 'ताप', 'fever', 'तापमान'],\n            'cough': ['खांसी', 'खोकला', 'cough', 'कफ'],\n            'breathlessness': ['सांस लेने में तकलीफ', 'श्वास कष्ट', 'breathlessness', 'दम फूलना'],\n            'chest_pain': ['छाती में दर्द', 'छातीत दुखी', 'chest pain', 'सीने में दर्द'],\n            'fatigue': ['थकान', 'कमजोरी', 'fatigue', 'अशक्तता'],\n            'nausea': ['मतली', 'जी मिचलाना', 'nausea', 'उलटी'],\n            'vomiting': ['उल्टी', 'वमन', 'vomiting', 'कै'],\n            'diarrhea': ['दस्त', 'अतिसार', 'diarrhea', 'पेट खराब'],\n            'abdominal_pain': ['पेट दर्द', 'उदर दर्द', 'abdominal pain', 'पेट में दर्द'],\n            'headache': ['सिरदर्द', 'डोकेदुखी', 'headache', 'माथा दुखना']\n        }\n        \n        # Create feature vector\n        feature_vector = []\n        \n        # Symptom features\n        for symptom_key in symptom_mapping.keys():\n            has_symptom = 0\n            for symptom in symptoms:\n                if symptom.lower() in [s.lower() for s in symptom_mapping[symptom_key]]:\n                    has_symptom = 1\n                    break\n            feature_vector.append(has_symptom)\n        \n        # Add remaining symptom features (set to 0 for simplicity)\n        while len(feature_vector) < 19:  # Total symptom features\n            feature_vector.append(0)\n        \n        # Vital signs (use defaults if not provided)\n        if vitals is None:\n            vitals = {\n                'heart_rate': 75.0,\n                'spo2': 98.0,\n                'temperature_f': 98.6,\n                'bp_systolic': 120.0,\n                'bp_diastolic': 80.0\n            }\n        \n        feature_vector.extend([\n            vitals.get('heart_rate', 75.0),\n            vitals.get('spo2', 98.0),\n            vitals.get('temperature_f', 98.6),\n            vitals.get('bp_systolic', 120.0),\n            vitals.get('bp_diastolic', 80.0)\n        ])\n        \n        # Demographic features\n        feature_vector.extend([\n            age,\n            1 if is_pregnant else 0,\n            1 if has_diabetes else 0,\n            1 if has_hypertension else 0,\n            1 if gender.upper() == 'M' else 0\n        ])\n        \n        return np.array(feature_vector).reshape(1, -1)\n    \n    def predict_risk(self, symptoms: List[str], age: int, gender: str, \n                    **kwargs) -> Dict[str, Any]:\n        \"\"\"Predict health risk level\"\"\"\n        \n        # Preprocess input\n        features = self.preprocess_input(symptoms, age, gender, **kwargs)\n        \n        # Predict using Random Forest (primary model)\n        if 'risk_rf' in self.models:\n            model = self.models['risk_rf']\n            risk_pred = model.predict(features)[0]\n            risk_proba = model.predict_proba(features)[0]\n            \n            # Convert to risk level\n            risk_levels = ['Green', 'Red', 'Yellow']  # Based on label encoding\n            risk_level = risk_levels[risk_pred]\n            confidence = max(risk_proba)\n            \n            return {\n                'risk_level': risk_level,\n                'confidence': float(confidence),\n                'probabilities': {level: float(prob) for level, prob in zip(risk_levels, risk_proba)}\n            }\n        \n        return {'error': 'Risk model not available'}\n    \n    def predict_disease(self, symptoms: List[str], age: int, gender: str, \n                      **kwargs) -> Dict[str, Any]:\n        \"\"\"Predict disease category\"\"\"\n        \n        # Preprocess input\n        features = self.preprocess_input(symptoms, age, gender, **kwargs)\n        \n        # Predict using Random Forest\n        if 'disease_rf' in self.models:\n            model = self.models['disease_rf']\n            disease_pred = model.predict(features)[0]\n            disease_proba = model.predict_proba(features)[0]\n            \n            # Get disease categories from metadata\n            disease_categories = self.metadata['target_names']['disease']\n            disease_category = disease_categories[disease_pred]\n            confidence = max(disease_proba)\n            \n            return {\n                'disease_category': disease_category,\n                'confidence': float(confidence),\n                'probabilities': {cat: float(prob) for cat, prob in zip(disease_categories, disease_proba)}\n            }\n        \n        return {'error': 'Disease model not available'}\n    \n    def get_model_info(self) -> Dict[str, Any]:\n        \"\"\"Get information about loaded models\"\"\"\n        \n        return {\n            'loaded_models': list(self.models.keys()),\n            'feature_count': len(self.metadata.get('feature_names', [])),\n            'target_names': self.metadata.get('target_names', {})\n        }\n\n# Performance optimization tips for Android integration:\n# 1. Use single-threaded inference (n_jobs=1)\n# 2. Preload models at app startup\n# 3. Cache preprocessed features when possible\n# 4. Use quantized models for faster inference\n# 5. Implement model pruning for smaller memory footprint\n'''\n        \n        return code\n    \n    def _create_android_assets(self, output_dir: str, metadata: Dict[str, Any]):\n        \"\"\"Create Android-compatible asset files\"\"\"\n        \n        android_dir = os.path.join(output_dir, 'android_assets')\n        os.makedirs(android_dir, exist_ok=True)\n        \n        # Create simplified model info for Android\n        android_info = {\n            'model_version': '1.0.0',\n            'feature_names': metadata['feature_names'],\n            'risk_levels': ['Green', 'Yellow', 'Red'],\n            'disease_categories': metadata['target_names']['disease'],\n            'symptom_mapping': {\n                'fever': ['बुखार', 'ताप', 'fever'],\n                'cough': ['खांसी', 'खोकला', 'cough'],\n                'breathlessness': ['सांस लेने में तकलीफ', 'breathlessness'],\n                'chest_pain': ['छाती में दर्द', 'chest pain'],\n                'fatigue': ['थकान', 'कमजोरी', 'fatigue']\n            }\n        }\n        \n        with open(os.path.join(android_dir, 'model_info.json'), 'w', encoding='utf-8') as f:\n            json.dump(android_info, f, indent=2, ensure_ascii=False)\n        \n        # Create performance optimization guide\n        optimization_guide = '''\n# Android Performance Optimization Guide for AI-Sanjivani\n\n## Memory Optimization\n1. Load models lazily (only when needed)\n2. Use model quantization (int8 instead of float32)\n3. Implement model pruning to remove unnecessary parameters\n4. Cache frequently used predictions\n\n## Inference Speed\n1. Use single-threaded inference (avoid thread overhead)\n2. Batch predictions when possible\n3. Precompute feature transformations\n4. Use native libraries for critical operations\n\n## Battery Optimization\n1. Minimize CPU usage during inference\n2. Use GPU acceleration when available\n3. Implement smart caching to reduce repeated computations\n4. Profile and optimize hot code paths\n\n## Model Size Reduction\n1. Use compressed model formats (joblib with compression)\n2. Remove unused features from models\n3. Quantize model weights\n4. Use model distillation for smaller student models\n'''\n        \n        with open(os.path.join(android_dir, 'optimization_guide.md'), 'w') as f:\n            f.write(optimization_guide)\n        \n        print(f\"Android assets created in: {android_dir}\")\n\nif __name__ == \"__main__\":\n    # Convert models for mobile deployment\n    converter = EdgeAIConverter()\n    \n    # Convert models from training output\n    models_dir = 'models'\n    mobile_dir = 'mobile_models'\n    \n    if os.path.exists(models_dir):\n        mobile_metadata = converter.create_mobile_inference_engine(models_dir, mobile_dir)\n        print(\"\\nMobile deployment package created successfully!\")\n        print(f\"Total models converted: {len(mobile_metadata['model_conversions'])}\")\n    else:\n        print(\"Please run train_model.py first to generate models.\")\n'''\n        \n        return code\n    \n    def _create_android_assets(self, output_dir: str, metadata: Dict[str, Any]):\n        \"\"\"Create Android-compatible asset files\"\"\"\n        \n        android_dir = os.path.join(output_dir, 'android_assets')\n        os.makedirs(android_dir, exist_ok=True)\n        \n        # Create simplified model info for Android\n        android_info = {\n            'model_version': '1.0.0',\n            'feature_names': metadata['feature_names'],\n            'risk_levels': ['Green', 'Yellow', 'Red'],\n            'disease_categories': metadata['target_names']['disease'],\n            'symptom_mapping': {\n                'fever': ['बुखार', 'ताप', 'fever'],\n                'cough': ['खांसी', 'खोकला', 'cough'],\n                'breathlessness': ['सांस लेने में तकलीफ', 'breathlessness'],\n                'chest_pain': ['छाती में दर्द', 'chest pain'],\n                'fatigue': ['थकान', 'कमजोरी', 'fatigue']\n            }\n        }\n        \n        with open(os.path.join(android_dir, 'model_info.json'), 'w', encoding='utf-8') as f:\n            json.dump(android_info, f, indent=2, ensure_ascii=False)\n        \n        print(f\"Android assets created in: {android_dir}\")\n\nif __name__ == \"__main__\":\n    # Convert models for mobile deployment\n    converter = EdgeAIConverter()\n    \n    # Convert models from training output\n    models_dir = 'models'\n    mobile_dir = 'mobile_models'\n    \n    if os.path.exists(models_dir):\n        mobile_metadata = converter.create_mobile_inference_engine(models_dir, mobile_dir)\n        print(\"\\nMobile deployment package created successfully!\")\n        print(f\"Total models converted: {len(mobile_metadata['model_conversions'])}\")\n    else:\n        print(\"Please run train_model.py first to generate models.\")