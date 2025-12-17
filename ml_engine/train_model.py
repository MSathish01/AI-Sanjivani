"""
Healthcare Data Scientist - Model Training Module
Train Random Forest and LightGBM models for health risk prediction
Optimized for rural healthcare scenarios with feature importance analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import json
import os
from data_generator import RuralHealthDataGenerator

class HealthRiskModelTrainer:
    """
    Train and evaluate health risk prediction models
    Supports both Random Forest and LightGBM with hyperparameter tuning
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.target_names = []
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Load training data from CSV or generate synthetic data"""
        
        if data_path and os.path.exists(data_path):
            print(f"Loading data from {data_path}")
            data = pd.read_csv(data_path)
        else:
            print("Generating synthetic training data...")
            generator = RuralHealthDataGenerator()
            data = generator.generate_dataset(n_patients=5000)
            
            # Save generated data
            os.makedirs('data', exist_ok=True)
            data.to_csv('data/rural_health_dataset.csv', index=False)
        
        print(f"Data loaded: {data.shape}")
        return data
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data for model training
        Handle categorical encoding and feature scaling
        """
        
        # Define feature columns
        symptom_features = [
            'fever', 'cough', 'breathlessness', 'chest_pain', 'fatigue',
            'nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'headache',
            'body_ache', 'dizziness', 'chills', 'rash', 'weight_loss',
            'night_sweats', 'frequent_urination', 'excessive_thirst', 'blurred_vision'
        ]
        
        vital_features = ['heart_rate', 'spo2', 'temperature_f', 'bp_systolic', 'bp_diastolic']
        demographic_features = ['age', 'is_pregnant', 'has_diabetes', 'has_hypertension']
        
        # Encode gender
        gender_encoder = LabelEncoder()
        data['gender_encoded'] = gender_encoder.fit_transform(data['gender'])
        
        # Select features
        feature_columns = symptom_features + vital_features + demographic_features + ['gender_encoded']
        X = data[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Target variables
        y_risk = data['risk_level']
        y_disease = data['disease_category']
        
        # Encode targets
        risk_encoder = LabelEncoder()
        disease_encoder = LabelEncoder()
        
        y_risk_encoded = risk_encoder.fit_transform(y_risk)
        y_disease_encoded = disease_encoder.fit_transform(y_disease)
        
        # Store encoders and feature names
        self.encoders['gender'] = gender_encoder
        self.encoders['risk'] = risk_encoder
        self.encoders['disease'] = disease_encoder
        self.feature_names = feature_columns
        self.target_names = {
            'risk': risk_encoder.classes_.tolist(),
            'disease': disease_encoder.classes_.tolist()
        }
        
        # Split data
        X_train, X_test, y_risk_train, y_risk_test, y_disease_train, y_disease_test = train_test_split(
            X, y_risk_encoded, y_disease_encoded, 
            test_size=0.2, random_state=42, stratify=y_risk_encoded
        )
        
        # Scale features for some models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Risk level distribution: {np.bincount(y_risk_train)}")
        
        return (X_train, X_test, X_train_scaled, X_test_scaled, 
                y_risk_train, y_risk_test, y_disease_train, y_disease_test)
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                           task: str = 'risk') -> Dict[str, Any]:
        """
        Train Random Forest model with hyperparameter tuning
        Optimized for mobile deployment
        """
        
        print(f"Training Random Forest for {task} prediction...")
        
        # Hyperparameter grid for mobile optimization
        param_grid = {
            'n_estimators': [30, 50, 100],  # Reduced for mobile performance
            'max_depth': [8, 12, 16],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
        
        # Base model
        rf = RandomForestClassifier(
            random_state=42,
            n_jobs=1,  # Single thread for mobile compatibility
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_rf = grid_search.best_estimator_
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='f1_weighted')
        
        # Store model
        self.models[f'rf_{task}'] = best_rf
        
        results = {
            'model': best_rf,
            'best_params': grid_search.best_params_,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(self.feature_names, best_rf.feature_importances_))
        }
        
        print(f"Random Forest {task} - CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        return results
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, 
                       task: str = 'risk') -> Dict[str, Any]:
        """
        Train LightGBM model optimized for mobile deployment
        """
        
        print(f"Training LightGBM for {task} prediction...")
        
        # LightGBM parameters optimized for mobile
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,  # Reduced for mobile
            'max_depth': 8,    # Reduced for mobile
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': 1  # Single thread for mobile
        }
        
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Train with early stopping
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,  # Reduced for mobile
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Cross-validation
        cv_results = lgb.cv(
            params,
            train_data,
            num_boost_round=100,
            nfold=5,
            stratified=True,
            shuffle=True,
            seed=42,
            return_cvbooster=True,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # Store model
        self.models[f'lgb_{task}'] = model
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, model.feature_importance()))
        
        results = {
            'model': model,
            'cv_scores': cv_results['valid multi_logloss-mean'],
            'best_iteration': model.best_iteration,
            'feature_importance': feature_importance
        }
        
        print(f"LightGBM {task} - Best CV Score: {min(cv_results['valid multi_logloss-mean']):.3f}")
        
        return results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, 
                       task: str = 'risk') -> Dict[str, Any]:
        """Evaluate trained models on test set"""
        
        results = {}
        
        for model_name in [f'rf_{task}', f'lgb_{task}']:
            if model_name in self.models:
                model = self.models[model_name]
                
                # Predictions
                if 'lgb' in model_name:
                    y_pred_proba = model.predict(X_test)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Classification report
                target_names = self.target_names[task]
                class_report = classification_report(
                    y_test, y_pred, target_names=target_names, output_dict=True
                )
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'classification_report': class_report,
                    'confusion_matrix': cm,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"{model_name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
        
        return results
    
    def plot_feature_importance(self, task: str = 'risk', top_n: int = 15):
        """Plot feature importance for trained models"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, model_name in enumerate([f'rf_{task}', f'lgb_{task}']):
            if model_name in self.models:
                model = self.models[model_name]
                
                if 'lgb' in model_name:
                    importance = model.feature_importance()
                else:
                    importance = model.feature_importances_
                
                # Create DataFrame for plotting
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False).head(top_n)
                
                # Plot
                sns.barplot(data=importance_df, y='feature', x='importance', ax=axes[i])
                axes[i].set_title(f'{model_name.upper()} Feature Importance')
                axes[i].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(f'feature_importance_{task}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self, model_dir: str = 'models'):
        """Save trained models for deployment"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f'{model_name}.joblib')
            
            if 'lgb' in model_name:
                # Save LightGBM model
                model.save_model(model_path.replace('.joblib', '.txt'))
            else:
                # Save sklearn model
                joblib.dump(model, model_path)
            
            print(f"Saved {model_name} to {model_path}")
        
        # Save encoders and scalers
        metadata = {
            'encoders': self.encoders,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
        
        metadata_path = os.path.join(model_dir, 'model_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        print(f"Saved metadata to {metadata_path}")
        
        # Save model info for mobile deployment
        mobile_info = {
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'model_files': {
                'risk_rf': f'rf_risk.joblib',
                'risk_lgb': f'lgb_risk.txt',
                'disease_rf': f'rf_disease.joblib',
                'disease_lgb': f'lgb_disease.txt'
            },
            'preprocessing': {
                'scaling_required': True,
                'encoding_required': True
            }
        }
        
        with open(os.path.join(model_dir, 'mobile_deployment_info.json'), 'w') as f:
            json.dump(mobile_info, f, indent=2)
    
    def train_complete_pipeline(self, data_path: str = None) -> Dict[str, Any]:
        """Complete training pipeline for both risk and disease prediction"""
        
        # Load and preprocess data
        data = self.load_data(data_path)
        (X_train, X_test, X_train_scaled, X_test_scaled, 
         y_risk_train, y_risk_test, y_disease_train, y_disease_test) = self.preprocess_data(data)
        
        results = {}
        
        # Train risk prediction models
        print("\n" + "="*50)
        print("TRAINING RISK PREDICTION MODELS")
        print("="*50)
        
        rf_risk_results = self.train_random_forest(X_train, y_risk_train, 'risk')
        lgb_risk_results = self.train_lightgbm(X_train, y_risk_train, 'risk')
        
        # Evaluate risk models
        risk_evaluation = self.evaluate_models(X_test, y_risk_test, 'risk')
        
        # Train disease prediction models
        print("\n" + "="*50)
        print("TRAINING DISEASE PREDICTION MODELS")
        print("="*50)
        
        rf_disease_results = self.train_random_forest(X_train, y_disease_train, 'disease')
        lgb_disease_results = self.train_lightgbm(X_train, y_disease_train, 'disease')
        
        # Evaluate disease models
        disease_evaluation = self.evaluate_models(X_test, y_disease_test, 'disease')
        
        # Compile results
        results = {
            'risk_models': {
                'random_forest': rf_risk_results,
                'lightgbm': lgb_risk_results,
                'evaluation': risk_evaluation
            },
            'disease_models': {
                'random_forest': rf_disease_results,
                'lightgbm': lgb_disease_results,
                'evaluation': disease_evaluation
            }
        }
        
        # Plot feature importance
        self.plot_feature_importance('risk')
        self.plot_feature_importance('disease')
        
        # Save models
        self.save_models()
        
        return results

if __name__ == "__main__":
    # Initialize trainer
    trainer = HealthRiskModelTrainer()
    
    # Train complete pipeline
    results = trainer.train_complete_pipeline()
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    
    for task in ['risk', 'disease']:
        print(f"\n{task.upper()} PREDICTION:")
        for model_type in ['random_forest', 'lightgbm']:
            model_key = f"{model_type[:2]}_{task}"
            if model_key in results[f'{task}_models']['evaluation']:
                eval_results = results[f'{task}_models']['evaluation'][model_key]
                print(f"  {model_type}: Accuracy={eval_results['accuracy']:.3f}, F1={eval_results['f1_score']:.3f}")
    
    print("\nModels saved successfully for mobile deployment!")