# 🍎 FoodVisor - AI-Powered Personal Food Safety & Nutrition Scanner

> **Scan a barcode or type a food name → instantly know if it's safe for YOU (allergies + medical conditions) + get a health score.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Gemini AI](https://img.shields.io/badge/Gemini-AI-orange.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 What It Does

FoodVisor analyzes food products and ingredients to provide **personalized safety reports** based on your:
- 🥜 **Allergies** (from text input or uploaded medical reports)
- 🏥 **Medical conditions** (diabetes, hypertension, etc.)
- 📊 **Personal health profile** (age, weight, fitness goals)

### Key Features

✅ **Barcode Scanning** - Scan product barcodes via camera or upload  
✅ **Ingredient Analysis** - AI-powered analysis of every ingredient  
✅ **Food Code Detection** - Identifies E-codes (E100, E621, etc.) with RAG system  
✅ **Personalized Safety** - Custom recommendations based on YOUR health  
✅ **Health Score** - 0-100 rating with "Eat / Limit / Avoid" guidance  
✅ **Food Logs** - Track your consumption history  
✅ **Medical Report OCR** - Extract data from allergy/medical reports  

## 🏗️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python + Flask |
| **AI Models** | Google Gemini 1.5/2.5 Flash + Hugging Face |
| **Barcode** | pyzbar + OpenCV + OpenFoodFacts API |
| **OCR** | Tesseract + Gemini Vision |
| **RAG System** | LangChain + FAISS + Google Embeddings |
| **Database** | SQLite |
| **Frontend** | HTML + CSS + JavaScript |

## 📁 Project Structure

```
FoodVisor/
├── food.py                 # Main Flask application
├── cleanrag.pdf            # Food additive knowledge base for RAG
├── requirements.txt        # Python dependencies
├── .env                    # API keys (create this!)
├── templates/              # HTML templates
│   ├── index.html
│   ├── foodsafety.html
│   └── ...
├── static/                 # CSS, JS, images
│   ├── styles.css
│   ├── script.js
│   └── ...
├── images/                 # Uploaded images (auto-created)
├── logs/                   # Application logs (auto-created)
└── foodvisor.db           # SQLite database (auto-created)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Google Gemini API keys ([Get them here](https://aistudio.google.com/app/apikey))
- Hugging Face API key (optional, for pro model)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/amoggh03/foodvisor7.git
cd foodvisor7
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:

```bash
# Google Gemini API Keys (get 4-8 keys for better rate limits)
GOOGLE_API_KEY_1=AIza...
GOOGLE_API_KEY_2=AIza...
GOOGLE_API_KEY_3=AIza...
GOOGLE_API_KEY_4=AIza...

# Flask Secret Key
FLASK_SECRET_KEY=your-secret-key-here

# Hugging Face API (optional)
HF_API_TOKEN=hf_...
```

5. **Run the application**
```bash
python food.py
```

6. **Open in browser**
```
http://127.0.0.1:5000
```

## 📖 How to Use

### Step 1: Set Up Your Profile
1. Enter personal details (age, weight, height, gender, activity level)
2. Upload allergy reports or type allergies manually
3. Upload medical reports or enter conditions manually
4. Save your profile

### Step 2: Analyze Food
Choose one of two methods:

**Method A: Barcode Scan**
1. Click "Scan Barcode"
2. Upload barcode image or use camera
3. Wait 3-6 seconds for analysis

**Method B: Type Food Name**
1. Enter food name (e.g., "Lays Classic Chips")
2. Click "Analyze"
3. Get instant results

### Step 3: View Results
- ✅ **Safety Report** - Personalized for your health
- 📊 **Nutrition Facts** - Calories, sugar, fat, sodium, etc.
- ⚠️ **Concerns** - Ingredients to avoid
- 💚 **Likes** - Positive aspects
- 🎯 **Health Score** - 0-100 rating
- 📝 **Recommendation** - Eat / Limit / Avoid

### Step 4: Track History
- View all analyzed foods in "Food Logs"
- Track your consumption patterns
- Review past safety reports

## 🔧 Configuration

### API Key Management

The app uses **smart API key rotation** to handle rate limits:
- Automatically cycles through multiple keys
- Skips exhausted keys
- Handles 429 errors gracefully

**Free Tier Limits** (per key):
- 15 requests per minute
- 1,500 requests per day

**Recommendation**: Use 4-8 API keys for smooth operation

### Check API Key Status
```bash
python check_api_keys.py
```

## 🧪 Testing

Test the application:
```bash
# Check API keys
python check_api_keys.py

# Run the app
python food.py

# Test with sample foods:
# - "Coca Cola"
# - "Lays Classic"
# - "Nutella"
```

## 📊 API Usage & Costs

### Free Tier
- **Cost**: $0
- **Limits**: 15 RPM, 1,500 RPD per key
- **Best for**: Testing, small-scale use

### Paid Tier
- **Cost**: ~$0.0045 per food scan
- **Limits**: 1000+ RPM
- **Best for**: Production, high volume

**Example**: 1,000 scans = $4.50

[Upgrade to paid tier](https://ai.google.dev/pricing)

## 🐛 Troubleshooting

### Issue: 429 Quota Errors
**Solution**: 
- Run `python check_api_keys.py` to check key status
- Add more API keys to `.env`
- Wait 24 hours for quota reset
- Upgrade to paid tier

### Issue: Barcode Not Detected
**Solution**:
- Ensure barcode is clear and well-lit
- Try adjusting brightness/contrast
- Use the manual barcode number input

### Issue: OCR Extraction Errors
**Solution**:
- Upload higher quality images
- Ensure text is clearly visible
- Use the manual text input option

## 🚀 Production Deployment

### Before Going Live:

1. **Use Production WSGI Server**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 food:app
```

2. **Enable HTTPS**
- Use nginx as reverse proxy
- Get SSL certificate (Let's Encrypt)

3. **Upgrade API Tier**
- Switch to paid Gemini API
- Implement caching for common ingredients

4. **Add Rate Limiting**
```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=lambda: request.remote_addr)
```

5. **Use Production Database**
- Consider PostgreSQL instead of SQLite
- Implement proper backups

6. **Monitor & Log**
- Set up error tracking (Sentry)
- Monitor API usage
- Track user analytics

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Google Gemini AI](https://ai.google.dev/) - AI models
- [OpenFoodFacts](https://world.openfoodfacts.org/) - Product database
- [Hugging Face](https://huggingface.co/) - ML models
- [LangChain](https://www.langchain.com/) - RAG framework

## 📧 Contact

**Developer**: Amogh  
**GitHub**: [@amoggh03](https://github.com/amoggh03)  
**Project**: [FoodVisor](https://github.com/amoggh03/foodvisor7)

---

⭐ **Star this repo if you find it useful!**

Built with ❤️ for healthier food choices
