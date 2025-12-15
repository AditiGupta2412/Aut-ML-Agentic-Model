"""
AutoBench Setup Script
Downloads required models and configures the environment.
"""
import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a shell command with status display."""
    print(f"\n{'='*50}")
    print(f"📦 {description}")
    print(f"{'='*50}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} - Complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e}")
        return False


def main():
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║   AutoBench - Glass-Box Agentic AI System Setup       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Install requirements
    run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )
    
    # Download SpaCy model
    run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Downloading SpaCy English model"
    )
    
    # Download NLTK data
    print("\n📥 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded!")
    except Exception as e:
        print(f"⚠️ NLTK download skipped: {e}")
    
    # Pre-download transformers models (optional)
    print("\n📥 Pre-downloading transformer models...")
    print("   (This may take a few minutes on first run)")
    
    try:
        from transformers import pipeline
        
        print("   Loading sentiment model...")
        _ = pipeline("sentiment-analysis", 
                    model="distilbert-base-uncased-finetuned-sst-2-english")
        print("   ✅ Sentiment model ready")
        
    except Exception as e:
        print(f"   ⚠️ Transformer models will be downloaded on first use: {e}")
    
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║   ✅ Setup Complete!                                  ║
    ╠═══════════════════════════════════════════════════════╣
    ║                                                       ║
    ║   To start the server, run:                          ║
    ║                                                       ║
    ║   cd backend                                          ║
    ║   uvicorn main:app --reload --port 8000              ║
    ║                                                       ║
    ║   Then open: http://localhost:8000                   ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()

