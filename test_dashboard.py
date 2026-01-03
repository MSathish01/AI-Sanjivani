"""
Test script to verify AI-Sanjivani dashboard functionality
"""

import sys
import os
sys.path.append('.')

try:
    from dashboard.app import PHCDashboard
    print("✅ Dashboard imports successful")
    
    # Test dashboard initialization
    dashboard = PHCDashboard()
    print("✅ Dashboard initialized successfully")
    
    # Test data generation
    data = dashboard.get_dashboard_data()
    print(f"✅ Dashboard data loaded: {data['total_assessments']} assessments")
    print(f"✅ High risk cases: {data['high_risk_cases']}")
    print(f"✅ Active villages: {data['active_villages']}")
    
    print("\n🎉 Dashboard is working correctly!")
    print("🌐 Run 'streamlit run dashboard/app.py' to start the web interface")
    
except Exception as e:
    print(f"❌ Dashboard test failed: {e}")
    import traceback
    traceback.print_exc()