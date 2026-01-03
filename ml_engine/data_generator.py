"""
AI-Sanjivani Training Data Generator
Generates realistic health assessment data for model training
Includes demographic and geographic diversity for rural India
"""

import numpy as np
import pandas as pd
import json
from typing import List, Dict, Tuple, Any
from datetime import datetime, timedelta
import random

class HealthDataGenerator:
    """
    Generates comprehensive health assessment training data
    Simulates real-world symptom patterns and risk distributions
    """
    
    def __init__(self):
        # Symptom categories and their relationships
        self.symptom_categories = {
            'respiratory': ['fever', 'cough', 'breathing_difficulty', 'chest_pain'],
            'gastrointestinal': ['nausea', 'vomiting', 'diarrhea'],
            'neurological': ['headache', 'weakness'],
            'musculoskeletal': ['body_ache'],
            'general': ['fever', 'weakness']
        }
        
        # Disease patterns (symptom combinations with typical risk levels)
        self.disease_patterns = {
            'common_cold': {
                'symptoms': ['fever', 'cough', 'headache'],
                'risk_level': 'Yellow',
                'probability': 0.25
            },
            'flu': {
                'symptoms': ['fever', 'cough', 'headache', 'body_ache', 'weakness'],
                'risk_level': 'Yellow',
                'probability': 0.15
            },
            'gastroenteritis': {
                'symptoms': ['nausea', 'vomiting', 'diarrhea', 'weakness'],
                'risk_level': 'Yellow',
                'probability': 0.12
            },
            'pneumonia': {
                'symptoms': ['fever', 'cough', 'breathing_difficulty', 'chest_pain', 'weakness'],
                'risk_level': 'Red',
                'probability': 0.08
            },
            'severe_dehydration': {
                'symptoms': ['vomiting', 'diarrhea', 'weakness', 'headache'],
                'risk_level': 'Red',
                'probability': 0.06
            },
            'cardiac_event': {
                'symptoms': ['chest_pain', 'breathing_difficulty', 'weakness'],
                'risk_level': 'Red',
                'probability': 0.04
            },
            'mild_headache': {
                'symptoms': ['headache'],
                'risk_level': 'Green',
                'probability': 0.15
            },
            'minor_ache': {
                'symptoms': ['body_ache'],
                'risk_level': 'Green',
                'probability': 0.10
            },
            'mild_fever': {
                'symptoms': ['fever'],
                'risk_level': 'Green',
                'probability': 0.05
            }
        }
        
        # Demographic distributions for rural India
        self.demographics = {
            'age_distribution': {
                'young_adult': (18, 35, 0.35),
                'middle_aged': (36, 55, 0.40),
                'elderly': (56, 80, 0.25)
            },
            'gender_distribution': {
                'male': 0.52,
                'female': 0.48
            }
        }
        
        # Regional and seasonal factors
        self.regional_factors = {
            'monsoon_diseases': ['diarrhea', 'fever', 'vomiting'],
            'summer_diseases': ['fever', 'weakness', 'headache'],
            'winter_diseases': ['cough', 'fever', 'body_ache']
        }
    
    def generate_patient_demographics(self) -> Dict[str, Any]:
        """Generate realistic patient demographics"""
        
        # Age based on distribution
        age_group = np.random.choice(
            list(self.demographics['age_distribution'].keys()),
            p=[dist[2] for dist in self.demographics['age_distribution'].values()]
        )
        
        age_range = self.demographics['age_distribution'][age_group]
        age = np.random.randint(age_range[0], age_range[1] + 1)
        
        # Gender
        gender = np.random.choice(
            ['M', 'F'],
            p=[self.demographics['gender_distribution']['male'],
               self.demographics['gender_distribution']['female']]
        )
        
        return {
            'age': age,
            'gender': gender,
            'age_group': age_group
        }
    
    def generate_symptom_combination(self) -> Tuple[List[str], str]:
        """Generate realistic symptom combination with risk level"""
        
        # Choose disease pattern or random combination
        if np.random.random() < 0.7:  # 70% follow known patterns
            pattern_name = np.random.choice(
                list(self.disease_patterns.keys()),
                p=[pattern['probability'] for pattern in self.disease_patterns.values()]
            )
            
            pattern = self.disease_patterns[pattern_name]
            base_symptoms = pattern['symptoms'].copy()
            base_risk = pattern['risk_level']
            
            # Add some variation
            if np.random.random() < 0.3:  # 30% chance to modify
                # Remove a symptom
                if len(base_symptoms) > 1 and np.random.random() < 0.5:
                    base_symptoms.pop(np.random.randint(len(base_symptoms)))
                
                # Add a related symptom
                if np.random.random() < 0.4:
                    all_symptoms = [s for symptoms in self.symptom_categories.values() 
                                  for s in symptoms]
                    additional = np.random.choice(all_symptoms)
                    if additional not in base_symptoms:
                        base_symptoms.append(additional)
            
            return base_symptoms, base_risk
        
        else:  # 30% random combinations
            num_symptoms = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.15, 0.05])
            
            all_symptoms = list(set([s for symptoms in self.symptom_categories.values() 
                                   for s in symptoms]))
            symptoms = np.random.choice(all_symptoms, num_symptoms, replace=False).tolist()
            
            # Determine risk based on symptom count and severity
            severe_symptoms = ['chest_pain', 'breathing_difficulty', 'vomiting']
            has_severe = any(s in symptoms for s in severe_symptoms)
            
            if has_severe or num_symptoms >= 4:
                risk = 'Red'
            elif num_symptoms >= 2:
                risk = 'Yellow'
            else:
                risk = 'Green'
            
            return symptoms, risk
    
    def add_demographic_risk_factors(self, symptoms: List[str], risk_level: str, 
                                   demographics: Dict[str, Any]) -> str:
        """Adjust risk level based on demographic factors"""
        
        age = demographics['age']
        
        # Elderly patients have higher risk
        if age >= 60:
            if risk_level == 'Green' and len(symptoms) >= 2:
                risk_level = 'Yellow'
            elif risk_level == 'Yellow' and any(s in symptoms for s in 
                                              ['fever', 'breathing_difficulty', 'chest_pain']):
                risk_level = 'Red'
        
        # Very young adults with severe symptoms
        elif age <= 25:
            if risk_level == 'Red' and len(symptoms) <= 2:
                risk_level = 'Yellow'  # Slightly lower risk for young adults
        
        return risk_level
    
    def create_feature_vector(self, symptoms: List[str], demographics: Dict[str, Any]) -> Dict[str, Any]:
        """Create feature vector for machine learning"""
        
        # All possible symptoms
        all_symptoms = list(set([s for symptoms in self.symptom_categories.values() 
                               for s in symptoms]))
        
        # Binary encoding for symptoms
        feature_vector = {}
        for symptom in all_symptoms:
            feature_vector[symptom] = 1 if symptom in symptoms else 0
        
        # Add demographic features
        feature_vector['age'] = demographics['age']
        feature_vector['gender'] = 1 if demographics['gender'] == 'M' else 0
        
        # Add derived features
        feature_vector['symptom_count'] = len(symptoms)
        feature_vector['has_fever'] = 1 if 'fever' in symptoms else 0
        feature_vector['has_respiratory'] = 1 if any(s in symptoms for s in 
                                                   self.symptom_categories['respiratory']) else 0
        feature_vector['has_gi'] = 1 if any(s in symptoms for s in 
                                          self.symptom_categories['gastrointestinal']) else 0
        
        return feature_vector
    
    def generate_single_record(self, patient_id: str = None) -> Dict[str, Any]:
        """Generate a single health assessment record"""
        
        if patient_id is None:
            patient_id = f"patient_{np.random.randint(10000, 99999)}"
        
        # Generate demographics
        demographics = self.generate_patient_demographics()
        
        # Generate symptoms and base risk
        symptoms, base_risk = self.generate_symptom_combination()
        
        # Adjust risk based on demographics
        final_risk = self.add_demographic_risk_factors(symptoms, base_risk, demographics)
        
        # Create feature vector
        features = self.create_feature_vector(symptoms, demographics)
        
        # Add metadata
        record = features.copy()
        record.update({
            'patient_id': patient_id,
            'risk_level': final_risk,
            'symptoms_list': symptoms,
            'age_group': demographics['age_group']
        })
        
        return record
    
    def generate_comprehensive_dataset(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate comprehensive training dataset"""
        
        print(f"Generating {n_samples} health assessment records...")
        
        records = []
        
        # Ensure balanced risk distribution
        target_distribution = {'Green': 0.4, 'Yellow': 0.45, 'Red': 0.15}
        risk_counts = {risk: 0 for risk in target_distribution.keys()}
        
        for i in range(n_samples):
            record = self.generate_single_record(f"patient_{i:06d}")
            
            # Check if we need to balance the dataset
            current_risk = record['risk_level']
            current_proportion = risk_counts[current_risk] / max(i, 1)
            target_proportion = target_distribution[current_risk]
            
            # If we're over-representing this risk level, try to generate a different one
            if current_proportion > target_proportion * 1.2 and i > 100:
                # Try up to 3 times to get a different risk level
                for _ in range(3):
                    new_record = self.generate_single_record(f"patient_{i:06d}")
                    if new_record['risk_level'] != current_risk:
                        record = new_record
                        break
            
            risk_counts[record['risk_level']] += 1
            records.append(record)
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1} records...")
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Total records: {len(df)}")
        print(f"Risk level distribution:")
        for risk, count in df['risk_level'].value_counts().items():
            print(f"  {risk}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"Age distribution:")
        for age_group, count in df['age_group'].value_counts().items():
            print(f"  {age_group}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"Gender distribution:")
        gender_counts = df['gender'].value_counts()
        print(f"  Male: {gender_counts.get(1, 0)} ({gender_counts.get(1, 0)/len(df)*100:.1f}%)")
        print(f"  Female: {gender_counts.get(0, 0)} ({gender_counts.get(0, 0)/len(df)*100:.1f}%)")
        
        # Remove helper columns
        df = df.drop(['symptoms_list', 'age_group'], axis=1)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "training_data.csv"):
        """Save dataset to CSV file"""
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
    
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate specific test cases for model validation"""
        
        test_cases = [
            # Green risk cases
            {
                'symptoms': ['headache'],
                'age': 25,
                'gender': 'F',
                'expected_risk': 'Green',
                'description': 'Young adult with mild headache'
            },
            {
                'symptoms': ['body_ache'],
                'age': 30,
                'gender': 'M',
                'expected_risk': 'Green',
                'description': 'Adult with minor body ache'
            },
            
            # Yellow risk cases
            {
                'symptoms': ['fever', 'cough'],
                'age': 35,
                'gender': 'F',
                'expected_risk': 'Yellow',
                'description': 'Adult with common cold symptoms'
            },
            {
                'symptoms': ['nausea', 'vomiting'],
                'age': 40,
                'gender': 'M',
                'expected_risk': 'Yellow',
                'description': 'Adult with gastric symptoms'
            },
            {
                'symptoms': ['fever', 'headache', 'body_ache'],
                'age': 65,
                'gender': 'F',
                'expected_risk': 'Yellow',
                'description': 'Elderly with flu-like symptoms'
            },
            
            # Red risk cases
            {
                'symptoms': ['chest_pain', 'breathing_difficulty'],
                'age': 55,
                'gender': 'M',
                'expected_risk': 'Red',
                'description': 'Middle-aged with cardiac symptoms'
            },
            {
                'symptoms': ['fever', 'vomiting', 'diarrhea', 'weakness'],
                'age': 70,
                'gender': 'F',
                'expected_risk': 'Red',
                'description': 'Elderly with severe dehydration symptoms'
            },
            {
                'symptoms': ['breathing_difficulty', 'fever', 'cough', 'chest_pain'],
                'age': 45,
                'gender': 'M',
                'expected_risk': 'Red',
                'description': 'Adult with pneumonia symptoms'
            }
        ]
        
        return test_cases

def main():
    """Demo data generation"""
    generator = HealthDataGenerator()
    
    # Generate sample dataset
    dataset = generator.generate_comprehensive_dataset(1000)
    
    # Save dataset
    generator.save_dataset(dataset, "sample_training_data.csv")
    
    # Generate test cases
    test_cases = generator.generate_test_cases()
    
    print(f"\nGenerated {len(test_cases)} test cases:")
    for i, case in enumerate(test_cases, 1):
        print(f"{i}. {case['description']}")
        print(f"   Symptoms: {case['symptoms']}")
        print(f"   Expected Risk: {case['expected_risk']}")

if __name__ == "__main__":
    main()