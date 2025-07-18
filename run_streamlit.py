"""
Streamlit Application Runner
Simple script to run the health monitoring Streamlit app
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def run_streamlit():
    """Run the Streamlit application"""
    try:
        # Set environment variables
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    print("🏥 AI-Powered Health Monitoring System")
    print("=" * 50)
    
    # Install requirements
    print("📦 Installing requirements...")
    if install_requirements():
        print("\n🚀 Starting Streamlit application...")
        print("🌐 Open your browser to: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application")
        print("=" * 50)
        run_streamlit()
    else:
        print("❌ Failed to install requirements. Please check your Python environment.")