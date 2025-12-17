"""
Healthcare Data Scientist Module
Synthetic Dataset Generation for Rural Health Risk Prediction
Designed for Indian rural healthcare scenarios
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import random

class RuralHealthDataGenerator:
    """
    Generate synthetic healthcare data for rural India
    Includes realistic disease patterns and demographic distributions
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Disease patterns common in rural India
        self.disease_categories = {
            'respiratory': ['fever', 'cough', 'breathlessness', 'chest_pain'],
            'gastrointestinal': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain'],
            'cardiovascular': ['chest_pain', 'breathlessness', 'fatigue', 'dizziness'],
            'infectious': ['fever', 'fatigue', 'body_ache', 'headache'],
            'maternal': ['nausea', 'fatigue', 'dizziness', 'abdominal_pain'],
            'diabetes_related': ['fatigue', 'frequent_urination', 'excessive_thirst', 'blurred_vision'],
            'hypertension': ['headache', 'dizziness', 'chest_pain', 'fatigue'],
            'malaria': ['fever', 'chills', 'headache', 'nausea'],
            'dengue': ['fever', 'body_ache', 'headache', 'rash'],
            'tuberculosis': ['cough', 'fever', 'weight_loss', 'night_sweats']
        }
        
        # Risk level mapping based on disease severity
        self.risk_mapping = {
            'respiratory': {'mild': 'Green', 'moderate': 'Yellow', 'severe': 'Red'},
            'gastrointestinal': {'mild': 'Green', 'moderate': 'Yellow', 'severe': 'Yellow'},
            'cardiovascular': {'mild': 'Yellow', 'moderate': 'Red', 'severe': 'Red'},
            'infectious': {'mild': 'Green', 'moderate': 'Yellow', 'severe': 'Red'},
            'maternal': {'mild': 'Yellow', 'moderate': 'Yellow', 'severe': 'Red'},
            'diabetes_related': {'mild': 'Yellow', 'moderate': 'Yellow', 'severe': 'Red'},
            'hypertension': {'mild': 'Yellow', 'moderate': 'Red', 'severe': 'Red'},
            'malaria': {'mild': 'Yellow', 'moderate': 'Red', 'severe': 'Red'},
            'dengue': {'mild': 'Yellow', 'moderate': 'Red', 'severe': 'Red'},
            'tuberculosis': {'mild': 'Yellow', 'moderate': 'Red', 'severe': 'Red'}
        }
    
    def generate_patient_demographics(self, n_patients: int) -> pd.DataFrame:
        """Generate realistic demographic data for rural Indian population"""
        
        demographics = []
        
        for i in range(n_patients):
            # Age distribution reflecting rural India demographics
            age_group = np.random.choice(['child', 'adult', 'elderly'], p=[0.3, 0.6, 0.1])
            
            if age_group == 'child':
                age = np.random.randint(1, 18)
            elif age_group == 'adult':
                age = np.random.randint(18, 60)
            else:
                age = np.random.randint(60, 85)
            
            # Gender distribution
            gender = np.random.choice(['M', 'F'], p=[0.52, 0.48])
            
            # Pregnancy flag (only for women of reproductive age)
            is_pregnant = False
            if gender == 'F' and 15 <= age <= 45:
                is_pregnant = np.random.choice([True, False], p=[0.05, 0.95])
            
            # Diabetes history (higher in adults)
            has_diabetes = False
            if age >= 30:
                diabetes_prob = min(0.15 + (age - 30) * 0.01, 0.25)
                has_diabetes = np.random.choice([True, False], p=[diabetes_prob, 1-diabetes_prob])
            
            # Hypertension history
            has_hypertension = False
            if age >= 25:
                hyp_prob = min(0.10 + (age - 25) * 0.008, 0.20)
                has_hypertension = np.random.choice([True, False], p=[hyp_prob, 1-hyp_prob])
            
            demographics.append({
                'patient_id': f'P_{i:06d}',
                'age': age,
                'gender': gender,
                'is_pregnant': is_pregnant,
                'has_diabetes': has_diabetes,
                'has_hypertension': has_hypertension,
                'age_group': age_group
            })
        
        return pd.DataFrame(demographics)
    
    def generate_symptoms_and_vitals(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Generate symptoms and vital signs based on demographics and disease patterns"""
        
        data = []
        
        for _, patient in demographics.iterrows():
            # Select disease category based on demographics
            disease_category = self._select_disease_category(patient)
            
            # Generate symptoms based on disease
            symptoms = self._generate_symptoms(disease_category, patient)
            
            # Generate vital signs
            vitals = self._generate_vitals(disease_category, symptoms, patient)
            
            # Determine severity and risk level
            severity = self._determine_severity(symptoms, vitals, patient)
            risk_level = self.risk_mapping[disease_category][severity]
            
            # Combine all data
            patient_data = {
                **patient.to_dict(),
                **symptoms,
                **vitals,
                'disease_category': disease_category,
                'severity': severity,
                'risk_level': risk_level
            }
            
            data.append(patient_data)
        
        return pd.DataFrame(data)
    
    def _select_disease_category(self, patient: pd.Series) -> str:
        """Select disease category based on patient demographics"""
        
        age = patient['age']
        gender = patient['gender']
        is_pregnant = patient['is_pregnant']
        has_diabetes = patient['has_diabetes']
        
        # Probability weights based on demographics
        if is_pregnant:
            weights = [0.1, 0.2, 0.1, 0.2, 0.3, 0.05, 0.05, 0.0, 0.0, 0.0]
        elif age < 5:
            weights = [0.3, 0.2, 0.05, 0.25, 0.0, 0.0, 0.0, 0.1, 0.05, 0.05]
        elif age >= 60:
            weights = [0.2, 0.15, 0.25, 0.15, 0.0, 0.1, 0.15, 0.0, 0.0, 0.0]
        elif has_diabetes:
            weights = [0.15, 0.15, 0.2, 0.15, 0.0, 0.2, 0.1, 0.025, 0.025, 0.0]
        else:
            weights = [0.2, 0.15, 0.1, 0.2, 0.0, 0.05, 0.05, 0.1, 0.1, 0.05]
        
        categories = list(self.disease_categories.keys())
        return np.random.choice(categories, p=weights)
    
    def _generate_symptoms(self, disease_category: str, patient: pd.Series) -> Dict[str, int]:
        """Generate symptoms based on disease category"""
        
        # All possible symptoms
        all_symptoms = [
            'fever', 'cough', 'breathlessness', 'chest_pain', 'fatigue',
            'nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'headache',
            'body_ache', 'dizziness', 'chills', 'rash', 'weight_loss',
            'night_sweats', 'frequent_urination', 'excessive_thirst', 'blurred_vision'
        ]
        
        # Initialize all symptoms as 0
        symptoms = {symptom: 0 for symptom in all_symptoms}
        
        # Get primary symptoms for this disease
        primary_symptoms = self.disease_categories[disease_category]
        
        # Add primary symptoms with high probability
        for symptom in primary_symptoms:
            if symptom in symptoms:
                symptoms[symptom] = np.random.choice([0, 1], p=[0.2, 0.8])
        
        # Add some random secondary symptoms
        secondary_symptoms = [s for s in all_symptoms if s not in primary_symptoms]
        n_secondary = np.random.randint(0, 3)
        
        for symptom in np.random.choice(secondary_symptoms, min(n_secondary, len(secondary_symptoms)), replace=False):
            symptoms[symptom] = np.random.choice([0, 1], p=[0.8, 0.2])
        
        return symptoms
    
    def _generate_vitals(self, disease_category: str, symptoms: Dict[str, int], patient: pd.Series) -> Dict[str, float]:
        """Generate vital signs based on symptoms and patient condition"""
        
        age = patient['age']
        
        # Normal ranges
        normal_hr = 70 + np.random.normal(0, 10)  # Heart rate
        normal_spo2 = 98 + np.random.normal(0, 1.5)  # SpO2
        normal_temp = 98.6 + np.random.normal(0, 0.5)  # Temperature (F)
        normal_bp_sys = 120 + np.random.normal(0, 10)  # Systolic BP
        normal_bp_dia = 80 + np.random.normal(0, 8)  # Diastolic BP
        
        # Adjust based on age
        if age >= 60:
            normal_hr += 5
            normal_bp_sys += 20
            normal_bp_dia += 10
        elif age < 18:
            normal_hr += 20
            normal_bp_sys -= 20
            normal_bp_dia -= 15
        
        # Adjust based on symptoms
        if symptoms['fever']:
            normal_temp += np.random.uniform(2, 5)  # Fever
            normal_hr += np.random.uniform(10, 25)  # Tachycardia with fever
        
        if symptoms['breathlessness'] or symptoms['chest_pain']:
            normal_spo2 -= np.random.uniform(2, 8)  # Reduced oxygen saturation
            normal_hr += np.random.uniform(15, 30)  # Tachycardia
        
        if disease_category == 'cardiovascular':
            normal_bp_sys += np.random.uniform(20, 40)
            normal_bp_dia += np.random.uniform(10, 20)
        
        # Ensure realistic ranges
        heart_rate = max(50, min(180, normal_hr))
        spo2 = max(70, min(100, normal_spo2))
        temperature = max(95, min(108, normal_temp))
        bp_systolic = max(80, min(200, normal_bp_sys))
        bp_diastolic = max(50, min(120, normal_bp_dia))
        
        return {
            'heart_rate': round(heart_rate, 1),
            'spo2': round(spo2, 1),
            'temperature_f': round(temperature, 1),
            'bp_systolic': round(bp_systolic, 1),
            'bp_diastolic': round(bp_diastolic, 1)
        }
    
    def _determine_severity(self, symptoms: Dict[str, int], vitals: Dict[str, float], patient: pd.Series) -> str:
        """Determine severity based on symptoms and vitals"""
        
        # Count active symptoms
        symptom_count = sum(symptoms.values())
        
        # Check for severe vital signs
        severe_vitals = (
            vitals['spo2'] < 90 or
            vitals['temperature_f'] > 103 or
            vitals['heart_rate'] > 120 or
            vitals['bp_systolic'] > 180
        )
        
        # Check for high-risk symptoms
        high_risk_symptoms = (
            symptoms.get('breathlessness', 0) or
            symptoms.get('chest_pain', 0) or
            symptoms.get('severe_abdominal_pain', 0)
        )
        
        # Age-based risk
        age_risk = patient['age'] >= 65 or patient['age'] < 2
        
        # Comorbidity risk
        comorbidity_risk = patient['has_diabetes'] or patient['has_hypertension']
        
        # Determine severity
        if severe_vitals or (high_risk_symptoms and (age_risk or comorbidity_risk)):
            return 'severe'
        elif symptom_count >= 3 or high_risk_symptoms or (symptom_count >= 2 and comorbidity_risk):
            return 'moderate'
        else:
            return 'mild'
    
    def generate_dataset(self, n_patients: int = 5000) -> pd.DataFrame:
        """Generate complete synthetic dataset"""
        
        print(f"Generating synthetic healthcare dataset for {n_patients} patients...")
        
        # Generate demographics
        demographics = self.generate_patient_demographics(n_patients)
        
        # Generate symptoms and vitals
        complete_data = self.generate_symptoms_and_vitals(demographics)
        
        print(f"Dataset generated successfully!")
        print(f"Shape: {complete_data.shape}")
        print(f"Risk level distribution:")
        print(complete_data['risk_level'].value_counts())
        
        return complete_data
    
    def save_dataset(self, data: pd.DataFrame, filepath: str = 'rural_health_dataset.csv'):
        """Save dataset to CSV"""
        data.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        
        # Save schema information
        schema = {
            'features': {
                'demographics': ['age', 'gender', 'is_pregnant', 'has_diabetes', 'has_hypertension'],
                'symptoms': ['fever', 'cough', 'breathlessness', 'chest_pain', 'fatigue', 'nausea', 
                           'vomiting', 'diarrhea', 'abdominal_pain', 'headache', 'body_ache', 
                           'dizziness', 'chills', 'rash', 'weight_loss', 'night_sweats', 
                           'frequent_urination', 'excessive_thirst', 'blurred_vision'],
                'vitals': ['heart_rate', 'spo2', 'temperature_f', 'bp_systolic', 'bp_diastolic']
            },
            'targets': {
                'disease_category': list(self.disease_categories.keys()),
                'risk_level': ['Green', 'Yellow', 'Red']
            },
            'data_types': {
                'categorical': ['gender', 'disease_category', 'risk_level', 'severity'],
                'binary': ['is_pregnant', 'has_diabetes', 'has_hypertension'] + 
                         ['fever', 'cough', 'breathlessness', 'chest_pain', 'fatigue', 'nausea', 
                          'vomiting', 'diarrhea', 'abdominal_pain', 'headache', 'body_ache', 
                          'dizziness', 'chills', 'rash', 'weight_loss', 'night_sweats', 
                          'frequent_urination', 'excessive_thirst', 'blurred_vision'],
                'numerical': ['age', 'heart_rate', 'spo2', 'temperature_f', 'bp_systolic', 'bp_diastolic']
            }
        }
        
        schema_file = filepath.replace('.csv', '_schema.json')
        with open(schema_file, 'w') as f:
            json.dump(schema, f, indent=2)
        print(f"Schema saved to {schema_file}")

if __name__ == "__main__":
    # Generate synthetic dataset
    generator = RuralHealthDataGenerator()
    dataset = generator.generate_dataset(n_patients=5000)
    
    # Save dataset
    generator.save_dataset(dataset, 'data/rural_health_dataset.csv')
    
    # Display sample data
    print("\nSample data:")
    print(dataset.head())
    
    print("\nDataset statistics:")
    print(dataset.describe())