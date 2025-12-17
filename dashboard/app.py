"""
AI-Sanjivani PHC Dashboard
Streamlit dashboard for Primary Health Centers
Shows village-level disease heatmap and analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import json
from typing import Dict, List, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.models.health_risk_classifier import HealthRiskClassifier

class PHCDashboard:
    """
    Primary Health Center Dashboard for AI-Sanjivani
    Provides village-level health analytics and disease tracking
    """
    
    def __init__(self):
        self.classifier = HealthRiskClassifier()
        self.db_path = "data/health_records.db"
        self.init_database()
        
        # Language support
        self.languages = {
            'English': 'english',
            'हिंदी': 'hindi', 
            'मराठी': 'marathi',
            'தமிழ்': 'tamil'
        }
        
        # Dashboard text translations
        self.translations = {
            'english': {
                'title': 'AI-Sanjivani PHC Dashboard',
                'subtitle': 'Village-Level Health Risk Monitoring',
                'village_overview': 'Village Health Overview',
                'risk_distribution': 'Risk Level Distribution',
                'symptom_trends': 'Symptom Trends',
                'recent_assessments': 'Recent Health Assessments',
                'high_risk_alerts': 'High Risk Alerts',
                'village_heatmap': 'Village Risk Heatmap',
                'total_assessments': 'Total Assessments',
                'high_risk_cases': 'High Risk Cases',
                'avg_risk_score': 'Average Risk Score',
                'active_villages': 'Active Villages'
            },
            'hindi': {
                'title': 'AI-संजीवनी PHC डैशबोर्ड',
                'subtitle': 'गांव-स्तरीय स्वास्थ्य जोखिम निगरानी',
                'village_overview': 'गांव स्वास्थ्य अवलोकन',
                'risk_distribution': 'जोखिम स्तर वितरण',
                'symptom_trends': 'लक्षण रुझान',
                'recent_assessments': 'हाल की स्वास्थ्य जांच',
                'high_risk_alerts': 'उच्च जोखिम अलर्ट',
                'village_heatmap': 'गांव जोखिम हीटमैप',
                'total_assessments': 'कुल जांच',
                'high_risk_cases': 'उच्च जोखिम मामले',
                'avg_risk_score': 'औसत जोखिम स्कोर',
                'active_villages': 'सक्रिय गांव'
            },
            'marathi': {
                'title': 'AI-संजीवनी PHC डॅशबोर्ड',
                'subtitle': 'गाव-पातळीवरील आरोग्य धोका निरीक्षण',
                'village_overview': 'गाव आरोग्य विहंगावलोकन',
                'risk_distribution': 'धोका पातळी वितरण',
                'symptom_trends': 'लक्षण ट्रेंड',
                'recent_assessments': 'अलीकडील आरोग्य मूल्यांकन',
                'high_risk_alerts': 'उच्च धोका अलर्ट',
                'village_heatmap': 'गाव धोका हीटमॅप',
                'total_assessments': 'एकूण मूल्यांकन',
                'high_risk_cases': 'उच्च धोका प्रकरणे',
                'avg_risk_score': 'सरासरी धोका स्कोअर',
                'active_villages': 'सक्रिय गावे'
            },
            'tamil': {
                'title': 'AI-சஞ்சீவனி PHC டாஷ்போர்டு',
                'subtitle': 'கிராம-நிலை உடல்நல அபாய கண்காணிப்பு',
                'village_overview': 'கிராம உடல்நல மேலோட்டம்',
                'risk_distribution': 'அபாய நிலை விநியோகம்',
                'symptom_trends': 'அறிகுறி போக்குகள்',
                'recent_assessments': 'சமீபத்திய உடல்நல மதிப்பீடுகள்',
                'high_risk_alerts': 'அதிக அபாய எச்சரிக்கைகள்',
                'village_heatmap': 'கிராம அபாய வெப்ப வரைபடம்',
                'total_assessments': 'மொத்த மதிப்பீடுகள்',
                'high_risk_cases': 'அதிக அபாய வழக்குகள்',
                'avg_risk_score': 'சராசரி அபாய மதிப்பெண்',
                'active_villages': 'செயலில் உள்ள கிராமங்கள்'
            }
        }
    
    def init_database(self):
        """Initialize SQLite database for health records"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                village_name TEXT NOT NULL,
                patient_age INTEGER,
                patient_gender TEXT,
                symptoms TEXT,
                risk_level TEXT,
                confidence REAL,
                assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                asha_worker_id TEXT,
                latitude REAL,
                longitude REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS villages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                latitude REAL,
                longitude REAL,
                population INTEGER,
                phc_distance REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Generate sample data if database is empty
        self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate sample health assessment data for demonstration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM health_assessments")
        if cursor.fetchone()[0] > 0:
            conn.close()
            return
        
        # Sample villages in rural Maharashtra/Tamil Nadu
        villages = [
            ('Shirdi', 19.7645, 74.4777, 32000, 5.2),
            ('Ahmednagar', 19.0948, 74.7480, 45000, 0.0),
            ('Sangamner', 19.5717, 74.2098, 28000, 12.5),
            ('Kopargaon', 19.8833, 74.4833, 35000, 8.3),
            ('Rahuri', 19.3931, 74.6497, 22000, 15.7),
            ('Thanjavur', 10.7870, 79.1378, 52000, 0.0),
            ('Kumbakonam', 10.9601, 79.3788, 38000, 18.2),
            ('Mayiladuthurai', 11.1085, 79.6540, 31000, 25.4),
            ('Tiruvarur', 10.7733, 79.6345, 28000, 22.1),
            ('Nagapattinam', 10.7661, 79.8448, 42000, 35.6)
        ]
        
        # Insert villages
        cursor.executemany(
            "INSERT INTO villages (name, latitude, longitude, population, phc_distance) VALUES (?, ?, ?, ?, ?)",
            villages
        )
        
        # Generate sample health assessments
        symptoms_combinations = [
            ['fever', 'cough'],
            ['headache', 'body_ache'],
            ['fever', 'headache', 'weakness'],
            ['cough', 'breathing_difficulty'],
            ['nausea', 'vomiting'],
            ['diarrhea', 'weakness'],
            ['fever', 'cough', 'headache', 'body_ache'],
            ['chest_pain', 'breathing_difficulty'],
            ['fever'],
            ['headache']
        ]
        
        risk_levels = ['Green', 'Yellow', 'Red']
        genders = ['M', 'F']
        
        # Generate 500 sample assessments over last 30 days
        assessments = []
        for i in range(500):
            village = np.random.choice([v[0] for v in villages])
            symptoms = np.random.choice(symptoms_combinations)
            age = np.random.randint(18, 80)
            gender = np.random.choice(genders)
            
            # Simulate risk prediction
            if len(symptoms) >= 4 or 'chest_pain' in symptoms or 'breathing_difficulty' in symptoms:
                risk = 'Red'
                confidence = np.random.uniform(0.7, 0.95)
            elif len(symptoms) >= 2:
                risk = 'Yellow'
                confidence = np.random.uniform(0.6, 0.85)
            else:
                risk = 'Green'
                confidence = np.random.uniform(0.5, 0.8)
            
            # Random date in last 30 days
            days_ago = np.random.randint(0, 30)
            assessment_date = datetime.now() - timedelta(days=days_ago)
            
            # Get village coordinates
            village_data = next(v for v in villages if v[0] == village)
            lat = village_data[1] + np.random.uniform(-0.01, 0.01)
            lon = village_data[2] + np.random.uniform(-0.01, 0.01)
            
            assessments.append((
                village, age, gender, json.dumps(symptoms), risk, confidence,
                assessment_date, f"ASHA_{np.random.randint(1, 20):03d}", lat, lon
            ))
        
        cursor.executemany('''
            INSERT INTO health_assessments 
            (village_name, patient_age, patient_gender, symptoms, risk_level, confidence, 
             assessment_date, asha_worker_id, latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', assessments)
        
        conn.commit()
        conn.close()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Fetch dashboard analytics data"""
        conn = sqlite3.connect(self.db_path)
        
        # Overall statistics
        total_assessments = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM health_assessments", conn
        ).iloc[0]['count']
        
        high_risk_cases = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM health_assessments WHERE risk_level = 'Red'", conn
        ).iloc[0]['count']
        
        avg_risk_score = pd.read_sql_query(
            "SELECT AVG(confidence) as avg_score FROM health_assessments", conn
        ).iloc[0]['avg_score']
        
        active_villages = pd.read_sql_query(
            "SELECT COUNT(DISTINCT village_name) as count FROM health_assessments", conn
        ).iloc[0]['count']
        
        # Risk distribution
        risk_distribution = pd.read_sql_query('''
            SELECT risk_level, COUNT(*) as count 
            FROM health_assessments 
            GROUP BY risk_level
        ''', conn)
        
        # Village-wise risk summary
        village_summary = pd.read_sql_query('''
            SELECT 
                village_name,
                COUNT(*) as total_cases,
                SUM(CASE WHEN risk_level = 'Red' THEN 1 ELSE 0 END) as high_risk_cases,
                AVG(confidence) as avg_confidence,
                AVG(latitude) as latitude,
                AVG(longitude) as longitude
            FROM health_assessments 
            GROUP BY village_name
        ''', conn)
        
        # Recent assessments
        recent_assessments = pd.read_sql_query('''
            SELECT village_name, risk_level, symptoms, assessment_date, asha_worker_id
            FROM health_assessments 
            ORDER BY assessment_date DESC 
            LIMIT 10
        ''', conn)
        
        # Symptom trends
        symptom_trends = pd.read_sql_query('''
            SELECT 
                DATE(assessment_date) as date,
                symptoms,
                COUNT(*) as count
            FROM health_assessments 
            WHERE assessment_date >= date('now', '-7 days')
            GROUP BY DATE(assessment_date), symptoms
            ORDER BY date DESC
        ''', conn)
        
        conn.close()
        
        return {
            'total_assessments': total_assessments,
            'high_risk_cases': high_risk_cases,
            'avg_risk_score': avg_risk_score or 0,
            'active_villages': active_villages,
            'risk_distribution': risk_distribution,
            'village_summary': village_summary,
            'recent_assessments': recent_assessments,
            'symptom_trends': symptom_trends
        }
    
    def render_dashboard(self):
        """Render the main dashboard"""
        # Language selector
        col1, col2 = st.columns([3, 1])
        with col2:
            selected_lang = st.selectbox(
                "Language / भाषा / भाषा / மொழி",
                options=list(self.languages.keys()),
                index=0
            )
        
        lang_code = self.languages[selected_lang]
        t = self.translations[lang_code]
        
        # Dashboard header
        st.title(t['title'])
        st.markdown(f"### {t['subtitle']}")
        
        # Get data
        data = self.get_dashboard_data()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label=t['total_assessments'],
                value=data['total_assessments']
            )
        
        with col2:
            st.metric(
                label=t['high_risk_cases'],
                value=data['high_risk_cases'],
                delta=f"{(data['high_risk_cases']/max(data['total_assessments'], 1)*100):.1f}%"
            )
        
        with col3:
            st.metric(
                label=t['avg_risk_score'],
                value=f"{data['avg_risk_score']:.2f}"
            )
        
        with col4:
            st.metric(
                label=t['active_villages'],
                value=data['active_villages']
            )
        
        # Risk distribution chart
        st.subheader(t['risk_distribution'])
        if not data['risk_distribution'].empty:
            fig_risk = px.pie(
                data['risk_distribution'], 
                values='count', 
                names='risk_level',
                color='risk_level',
                color_discrete_map={'Green': '#28a745', 'Yellow': '#ffc107', 'Red': '#dc3545'}
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # Village heatmap
        st.subheader(t['village_heatmap'])
        if not data['village_summary'].empty:
            # Calculate risk percentage for color coding
            data['village_summary']['risk_percentage'] = (
                data['village_summary']['high_risk_cases'] / 
                data['village_summary']['total_cases'] * 100
            )
            
            fig_map = px.scatter_mapbox(
                data['village_summary'],
                lat='latitude',
                lon='longitude',
                size='total_cases',
                color='risk_percentage',
                hover_name='village_name',
                hover_data=['total_cases', 'high_risk_cases'],
                color_continuous_scale='Reds',
                size_max=20,
                zoom=7,
                mapbox_style='open-street-map'
            )
            
            fig_map.update_layout(height=500)
            st.plotly_chart(fig_map, use_container_width=True)
        
        # Recent assessments table
        st.subheader(t['recent_assessments'])
        if not data['recent_assessments'].empty:
            # Format the data for display
            display_data = data['recent_assessments'].copy()
            display_data['symptoms'] = display_data['symptoms'].apply(
                lambda x: ', '.join(json.loads(x)) if x else ''
            )
            display_data['assessment_date'] = pd.to_datetime(
                display_data['assessment_date']
            ).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(display_data, use_container_width=True)
        
        # High risk alerts
        st.subheader(t['high_risk_alerts'])
        high_risk_recent = data['recent_assessments'][
            data['recent_assessments']['risk_level'] == 'Red'
        ]
        
        if not high_risk_recent.empty:
            for _, alert in high_risk_recent.iterrows():
                st.error(
                    f"🚨 High Risk Alert: {alert['village_name']} - "
                    f"ASHA Worker: {alert['asha_worker_id']} - "
                    f"Date: {alert['assessment_date']}"
                )
        else:
            st.success("No high-risk alerts in recent assessments")

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="AI-Sanjivani PHC Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize and render dashboard
    dashboard = PHCDashboard()
    dashboard.render_dashboard()
    
    # Sidebar with additional info
    with st.sidebar:
        st.header("About AI-Sanjivani")
        st.write("""
        AI-Sanjivani is an offline-capable AI healthcare assistant designed for rural India.
        
        **Features:**
        - Multilingual support (Hindi/Marathi/Tamil/English)
        - Offline health risk assessment
        - Voice input for symptoms
        - Village-level health monitoring
        - ASHA worker support tools
        """)
        
        st.header("Emergency Contacts")
        st.write("""
        **Emergency:** 108
        **PHC Helpline:** 104
        **Ambulance:** 102
        """)

if __name__ == "__main__":
    main()