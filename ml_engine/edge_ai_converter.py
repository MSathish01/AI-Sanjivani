"""
Edge AI Specialist Module
Convert trained ML models for offline mobile deployment
Optimized for low-end Android devices with minimal memory footprint
"""

try:
    import tensorflow as tf  # type: ignore[import-not-found]
    HAS_TENSORFLOW = True
except ImportError:
    tf = None  # type: ignore
    HAS_TENSORFLOW = False

import numpy as np
import joblib
import json
import os
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class EdgeAIConverter:
    """
    Convert ML models to edge-optimized formats
    Supports TensorFlow Lite, ONNX, and optimized joblib formats
    """

    def __init__(self):
        self.model_info = {}

    def convert_sklearn_to_tflite(
        self, model_path: str, feature_names: list, output_path: str
    ) -> Dict[str, Any]:
        """
        Convert sklearn model to TensorFlow Lite format
        Optimized for mobile inference
        """

        if not HAS_TENSORFLOW or tf is None:
            return {"error": "TensorFlow not installed"}

        print(f"Converting sklearn model to TensorFlow Lite: {model_path}")

        # Load sklearn model
        model = joblib.load(model_path)

        # Generate sample data to understand input/output shapes
        n_features = len(feature_names)
        sample_input = np.random.random((1000, n_features)).astype(np.float32)

        # Get predictions from original model
        if hasattr(model, "predict_proba"):
            sample_output = model.predict_proba(sample_input).astype(np.float32)
        else:
            sample_output = model.predict(sample_input)
            if len(sample_output.shape) == 1:
                n_classes = len(np.unique(sample_output))
                sample_output_onehot = np.zeros((len(sample_output), n_classes))
                sample_output_onehot[np.arange(len(sample_output)), sample_output] = 1
                sample_output = sample_output_onehot.astype(np.float32)

        # Create TensorFlow model
        tf_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(n_features,)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(sample_output.shape[1], activation="softmax"),
            ]
        )

        tf_model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        print("Training neural network to mimic original model...")
        tf_model.fit(
            sample_input,
            sample_output,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
        )

        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)

        # Test the converted model
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        model_info = {
            "input_shape": input_details[0]["shape"].tolist(),
            "output_shape": output_details[0]["shape"].tolist(),
            "model_size_bytes": len(tflite_model),
            "quantization": "float16",
            "feature_names": feature_names,
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

        model = joblib.load(model_path)

        if isinstance(model, RandomForestClassifier):
            optimized_model = self._optimize_random_forest(model)
        else:
            optimized_model = model

        joblib.dump(optimized_model, output_path, compress=3)

        original_size = os.path.getsize(model_path)
        optimized_size = os.path.getsize(output_path)

        optimization_info = {
            "original_size_bytes": original_size,
            "optimized_size_bytes": optimized_size,
            "compression_ratio": original_size / optimized_size,
            "model_type": type(model).__name__,
        }

        print(f"Optimized model saved: {output_path}")
        print(f"Size reduction: {original_size/1024:.1f} KB -> {optimized_size/1024:.1f} KB")
        print(f"Compression ratio: {optimization_info['compression_ratio']:.2f}x")

        return optimization_info

    def _optimize_random_forest(self, rf_model: RandomForestClassifier) -> RandomForestClassifier:
        """
        Optimize Random Forest for mobile deployment
        """

        # Use getattr for safe attribute access
        n_estimators = getattr(rf_model, "n_estimators", 100)
        max_depth = getattr(rf_model, "max_depth", None) or 10
        min_samples_split = getattr(rf_model, "min_samples_split", 2)
        min_samples_leaf = getattr(rf_model, "min_samples_leaf", 1)
        random_state = getattr(rf_model, "random_state", None)

        optimized_rf = RandomForestClassifier(
            n_estimators=min(30, n_estimators),
            max_depth=min(10, max_depth),
            min_samples_split=max(10, min_samples_split),
            min_samples_leaf=max(5, min_samples_leaf),
            random_state=random_state,
            n_jobs=1,
        )

        if hasattr(rf_model, "estimators_"):
            optimized_rf.estimators_ = rf_model.estimators_[:30]  # type: ignore
            optimized_rf.n_estimators = len(optimized_rf.estimators_)  # type: ignore
            optimized_rf.classes_ = rf_model.classes_  # type: ignore
            optimized_rf.n_classes_ = rf_model.n_classes_  # type: ignore
            optimized_rf.n_features_in_ = rf_model.n_features_in_  # type: ignore
            optimized_rf.feature_importances_ = rf_model.feature_importances_  # type: ignore

        return optimized_rf

    def convert_lightgbm_to_mobile(self, model_path: str, output_path: str) -> Dict[str, Any]:
        """Convert LightGBM model for mobile deployment"""

        if not HAS_LIGHTGBM:
            return {"error": "LightGBM not installed"}

        print(f"Converting LightGBM model for mobile: {model_path}")

        model = lgb.Booster(model_file=model_path)
        model.save_model(output_path, num_iteration=model.best_iteration)

        model_info = {
            "num_trees": model.num_trees(),
            "num_features": model.num_feature(),
            "best_iteration": model.best_iteration,
            "model_size_bytes": os.path.getsize(output_path),
        }

        print(f"Mobile LightGBM model saved: {output_path}")
        print(f"Model size: {model_info['model_size_bytes'] / 1024:.1f} KB")

        return model_info

    def create_mobile_inference_engine(self, models_dir: str, output_dir: str) -> Dict[str, Any]:
        """Create optimized inference engine for mobile deployment"""

        print("Creating mobile inference engine...")

        os.makedirs(output_dir, exist_ok=True)

        metadata_path = os.path.join(models_dir, "model_metadata.joblib")
        metadata = joblib.load(metadata_path)

        mobile_inference_code = self._generate_mobile_inference_code(metadata)

        with open(os.path.join(output_dir, "mobile_inference.py"), "w") as f:
            f.write(mobile_inference_code)

        model_conversions = {}

        for model_name in ["rf_risk.joblib", "rf_disease.joblib"]:
            model_path = os.path.join(models_dir, model_name)
            if os.path.exists(model_path):
                optimized_path = os.path.join(output_dir, f"optimized_{model_name}")
                opt_info = self.optimize_joblib_model(model_path, optimized_path)
                model_conversions[model_name] = opt_info

        for model_name in ["lgb_risk.txt", "lgb_disease.txt"]:
            model_path = os.path.join(models_dir, model_name)
            if os.path.exists(model_path):
                mobile_path = os.path.join(output_dir, f"mobile_{model_name}")
                lgb_info = self.convert_lightgbm_to_mobile(model_path, mobile_path)
                model_conversions[model_name] = lgb_info

        mobile_metadata = {
            "feature_names": metadata["feature_names"],
            "target_names": metadata["target_names"],
            "encoders": metadata["encoders"],
            "model_conversions": model_conversions,
        }

        with open(os.path.join(output_dir, "mobile_metadata.json"), "w") as f:
            json.dump(mobile_metadata, f, indent=2, default=str)

        self._create_android_assets(output_dir, mobile_metadata)

        print(f"Mobile inference engine created in: {output_dir}")

        return mobile_metadata

    def _generate_mobile_inference_code(self, metadata: Dict[str, Any]) -> str:
        """Generate optimized Python inference code for mobile"""

        code = '''"""
Mobile Inference Engine for AI-Sanjivani
Optimized for offline deployment on low-end Android devices
"""

import numpy as np
import joblib
import json
import os
from typing import Dict, List, Any


class MobileHealthPredictor:
    """
    Lightweight health risk predictor for mobile deployment
    Optimized for speed and memory efficiency
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models = {}
        self.metadata = {}
        self.load_models()

    def load_models(self):
        """Load optimized models and metadata"""

        metadata_path = os.path.join(self.model_dir, "mobile_metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        model_files = {
            "risk_rf": "optimized_rf_risk.joblib",
            "disease_rf": "optimized_rf_disease.joblib",
        }

        for model_name, filename in model_files.items():
            model_path = os.path.join(self.model_dir, filename)
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)

    def preprocess_input(
        self,
        symptoms: List[str],
        age: int,
        gender: str,
        is_pregnant: bool = False,
        has_diabetes: bool = False,
        has_hypertension: bool = False,
        vitals: Dict[str, float] = None,
    ) -> np.ndarray:
        """Preprocess input for model inference"""

        symptom_mapping = {
            "fever": ["बुखार", "ताप", "fever", "तापमान"],
            "cough": ["खांसी", "खोकला", "cough", "कफ"],
            "breathlessness": ["सांस लेने में तकलीफ", "श्वास कष्ट", "breathlessness", "दम फूलना"],
            "chest_pain": ["छाती में दर्द", "छातीत दुखी", "chest pain", "सीने में दर्द"],
            "fatigue": ["थकान", "कमजोरी", "fatigue", "अशक्तता"],
            "nausea": ["मतली", "जी मिचलाना", "nausea", "उलटी"],
            "vomiting": ["उल्टी", "वमन", "vomiting", "कै"],
            "diarrhea": ["दस्त", "अतिसार", "diarrhea", "पेट खराब"],
            "abdominal_pain": ["पेट दर्द", "उदर दर्द", "abdominal pain", "पेट में दर्द"],
            "headache": ["सिरदर्द", "डोकेदुखी", "headache", "माथा दुखना"],
        }

        feature_vector = []

        for symptom_key in symptom_mapping.keys():
            has_symptom = 0
            for symptom in symptoms:
                if symptom.lower() in [s.lower() for s in symptom_mapping[symptom_key]]:
                    has_symptom = 1
                    break
            feature_vector.append(has_symptom)

        while len(feature_vector) < 19:
            feature_vector.append(0)

        if vitals is None:
            vitals = {
                "heart_rate": 75.0,
                "spo2": 98.0,
                "temperature_f": 98.6,
                "bp_systolic": 120.0,
                "bp_diastolic": 80.0,
            }

        feature_vector.extend(
            [
                vitals.get("heart_rate", 75.0),
                vitals.get("spo2", 98.0),
                vitals.get("temperature_f", 98.6),
                vitals.get("bp_systolic", 120.0),
                vitals.get("bp_diastolic", 80.0),
            ]
        )

        feature_vector.extend(
            [
                age,
                1 if is_pregnant else 0,
                1 if has_diabetes else 0,
                1 if has_hypertension else 0,
                1 if gender.upper() == "M" else 0,
            ]
        )

        return np.array(feature_vector).reshape(1, -1)

    def predict_risk(self, symptoms: List[str], age: int, gender: str, **kwargs) -> Dict[str, Any]:
        """Predict health risk level"""

        features = self.preprocess_input(symptoms, age, gender, **kwargs)

        if "risk_rf" in self.models:
            model = self.models["risk_rf"]
            risk_pred = model.predict(features)[0]
            risk_proba = model.predict_proba(features)[0]

            risk_levels = ["Green", "Red", "Yellow"]
            risk_level = risk_levels[risk_pred]
            confidence = max(risk_proba)

            return {
                "risk_level": risk_level,
                "confidence": float(confidence),
                "probabilities": {level: float(prob) for level, prob in zip(risk_levels, risk_proba)},
            }

        return {"error": "Risk model not available"}

    def predict_disease(self, symptoms: List[str], age: int, gender: str, **kwargs) -> Dict[str, Any]:
        """Predict disease category"""

        features = self.preprocess_input(symptoms, age, gender, **kwargs)

        if "disease_rf" in self.models:
            model = self.models["disease_rf"]
            disease_pred = model.predict(features)[0]
            disease_proba = model.predict_proba(features)[0]

            disease_categories = self.metadata["target_names"]["disease"]
            disease_category = disease_categories[disease_pred]
            confidence = max(disease_proba)

            return {
                "disease_category": disease_category,
                "confidence": float(confidence),
                "probabilities": {cat: float(prob) for cat, prob in zip(disease_categories, disease_proba)},
            }

        return {"error": "Disease model not available"}

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""

        return {
            "loaded_models": list(self.models.keys()),
            "feature_count": len(self.metadata.get("feature_names", [])),
            "target_names": self.metadata.get("target_names", {}),
        }
'''
        return code

    def _create_android_assets(self, output_dir: str, metadata: Dict[str, Any]):
        """Create Android-compatible asset files"""

        android_dir = os.path.join(output_dir, "android_assets")
        os.makedirs(android_dir, exist_ok=True)

        android_info = {
            "model_version": "1.0.0",
            "feature_names": metadata["feature_names"],
            "risk_levels": ["Green", "Yellow", "Red"],
            "disease_categories": metadata["target_names"]["disease"],
            "symptom_mapping": {
                "fever": ["बुखार", "ताप", "fever"],
                "cough": ["खांसी", "खोकला", "cough"],
                "breathlessness": ["सांस लेने में तकलीफ", "breathlessness"],
                "chest_pain": ["छाती में दर्द", "chest pain"],
                "fatigue": ["थकान", "कमजोरी", "fatigue"],
            },
        }

        with open(os.path.join(android_dir, "model_info.json"), "w", encoding="utf-8") as f:
            json.dump(android_info, f, indent=2, ensure_ascii=False)

        optimization_guide = """# Android Performance Optimization Guide for AI-Sanjivani

## Memory Optimization
1. Load models lazily (only when needed)
2. Use model quantization (int8 instead of float32)
3. Implement model pruning to remove unnecessary parameters
4. Cache frequently used predictions

## Inference Speed
1. Use single-threaded inference (avoid thread overhead)
2. Batch predictions when possible
3. Precompute feature transformations
4. Use native libraries for critical operations

## Battery Optimization
1. Minimize CPU usage during inference
2. Use GPU acceleration when available
3. Implement smart caching to reduce repeated computations
4. Profile and optimize hot code paths

## Model Size Reduction
1. Use compressed model formats (joblib with compression)
2. Remove unused features from models
3. Quantize model weights
4. Use model distillation for smaller student models
"""

        with open(os.path.join(android_dir, "optimization_guide.md"), "w") as f:
            f.write(optimization_guide)

        print(f"Android assets created in: {android_dir}")


if __name__ == "__main__":
    converter = EdgeAIConverter()

    models_dir = "models"
    mobile_dir = "mobile_models"

    if os.path.exists(models_dir):
        mobile_metadata = converter.create_mobile_inference_engine(models_dir, mobile_dir)
        print("\nMobile deployment package created successfully!")
        print(f"Total models converted: {len(mobile_metadata['model_conversions'])}")
    else:
        print("Please run train_model.py first to generate models.")
