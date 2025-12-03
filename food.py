from flask import Flask, render_template, request, jsonify, redirect, url_for, session  # Add session to imports
import os
import re
import base64
from PIL import Image
import pytesseract
from langchain_google_genai import ChatGoogleGenerativeAI
import sqlite3
import json
from datetime import datetime
import requests
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time
from pyzbar import pyzbar
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/processBarcode": {"origins": "chrome-extension://*"}})

# Set a secret key for session management (required for Flask sessions)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-default-secret-key-here')  # Use an env variable or a default secure keyclearSession

# API Keys and Configurations
GOOGLE_API_KEYS = [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4")
]
for key in GOOGLE_API_KEYS:
    if not key:
        raise ValueError("One or more GOOGLE_API_KEYs not found in .env file")

# Global counter to alternate API keys
current_api_key_index = 0
exhausted_keys = set()  # Track exhausted keys

# Pre-mark key #3 as exhausted (from check_api_keys.py results)
if len(GOOGLE_API_KEYS) > 2:
    exhausted_keys.add(GOOGLE_API_KEYS[2])
    print(f"⚠️ Pre-marking key #3 as exhausted: {GOOGLE_API_KEYS[2][:10]}...")

genai.configure(api_key=GOOGLE_API_KEYS[0])

def get_llm_instance(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
        max_tokens=2000,
        max_retries=0,  # Disable LangChain retries, we handle it ourselves
    )

# Global llm instance (initialize with first key)
llm = get_llm_instance(GOOGLE_API_KEYS[0])

def get_next_api_key():
    """Cycle through the list of API keys, skipping exhausted ones."""
    global current_api_key_index, exhausted_keys
    
    attempts = 0
    max_attempts = len(GOOGLE_API_KEYS) * 2
    
    while attempts < max_attempts:
        api_key = GOOGLE_API_KEYS[current_api_key_index]
        current_api_key_index = (current_api_key_index + 1) % len(GOOGLE_API_KEYS)
        
        # Skip exhausted keys
        if api_key not in exhausted_keys:
            print(f"Using API Key: {api_key[:10]}... (Index: {current_api_key_index - 1})")
            return api_key
        else:
            print(f"Skipping exhausted key: {api_key[:10]}...")
        
        attempts += 1
    
    # All keys exhausted - clear and retry
    print("⚠️ All API keys exhausted. Clearing exhausted list and retrying...")
    exhausted_keys.clear()
    return GOOGLE_API_KEYS[0]

def mark_key_exhausted(api_key):
    """Mark a key as exhausted when it hits 429."""
    global exhausted_keys
    exhausted_keys.add(api_key)
    print(f"❌ Key {api_key[:10]}... marked as exhausted")

def make_gemini_call_with_key(prompt, api_key, model=None, max_retries=2, delay=0.5):
    """Helper function to make a single Gemini API call with a specific API key."""
    genai.configure(api_key=api_key)
    for attempt in range(max_retries):
        try:
            if model:
                model_instance = genai.GenerativeModel(model)
                response = model_instance.generate_content(prompt)
                time.sleep(delay)
                return response.text if hasattr(response, 'text') else response.content
            else:
                current_llm = get_llm_instance(api_key)
                response = current_llm.invoke(prompt)
                time.sleep(delay)
                return response.content
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "resource exhausted" in error_str.lower():
                print(f"⚠️ Quota exceeded with key {api_key[:10]}...")
                mark_key_exhausted(api_key)
                return None  # Signal to try another key
            else:
                print(f"❌ Error with {api_key[:10]}...: {error_str[:100]}")
                return f"Error: {str(e)}"
    return None

def parallel_gemini_calls(prompts, ingredients, model=None, delay=0.5):
    """Parallelize Gemini API calls for a list of prompts, ensuring correct mapping to ingredients."""
    results = []
    # Create a mapping of prompts to ingredients
    prompt_to_ingredient = {id(prompt): ingredient for prompt, ingredient in zip(prompts, ingredients)}
    with ThreadPoolExecutor(max_workers=len(GOOGLE_API_KEYS)) as executor:
        # Map each prompt to an API key and a future
        future_to_prompt = {
            executor.submit(make_gemini_call_with_key, prompt, GOOGLE_API_KEYS[i % len(GOOGLE_API_KEYS)], model, 4, delay): prompt
            for i, prompt in enumerate(prompts)
        }
        # Collect results as they complete
        temp_results = {}
        for future in as_completed(future_to_prompt):
            prompt = future_to_prompt[future]
            ingredient = prompt_to_ingredient[id(prompt)]
            try:
                result = future.result()
                temp_results[ingredient] = result
            except Exception as e:
                temp_results[ingredient] = f"Error: {str(e)}"
        
        # Reconstruct results in the original order of ingredients
        for ingredient in ingredients:
            if ingredient in temp_results:
                results.append((prompt_to_ingredient[id(prompts[ingredients.index(ingredient)])], temp_results[ingredient]))
    
    return results

def make_gemini_call(prompt, model=None, max_retries=None, delay=0.5):
    """Make a Gemini API call with retries across different API keys."""
    if max_retries is None:
        max_retries = len(GOOGLE_API_KEYS)
    
    for attempt in range(max_retries):
        api_key = get_next_api_key()
        genai.configure(api_key=api_key)
        
        try:
            if model:
                model_instance = genai.GenerativeModel(model)
                response = model_instance.generate_content(prompt)
            else:
                current_llm = get_llm_instance(api_key)
                response = current_llm.invoke(prompt)
            
            time.sleep(delay)
            return response
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "resource exhausted" in error_str.lower():
                print(f"⚠️ Quota exceeded with key {api_key[:10]}... (Attempt {attempt + 1}/{max_retries})")
                mark_key_exhausted(api_key)
                if attempt < max_retries - 1:
                    print("🔄 Trying next key...")
                    time.sleep(1)
                    continue
                else:
                    raise Exception(f"All {len(GOOGLE_API_KEYS)} API keys exhausted. Please wait or add more keys.")
            else:
                print(f"❌ API Error: {error_str[:200]}")
                raise e
    raise Exception("Failed to complete API call after all retries")

# Hugging Face API Setup
HF_API_URL = os.getenv("HF_API_URL", "https://lfotro8pc3ooznh6.us-east-1.aws.endpoints.huggingface.cloud")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    print("⚠️ Warning: HF_API_TOKEN not found in .env file. Hugging Face features will be disabled.")
HF_HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {HF_API_TOKEN}" if HF_API_TOKEN else "",
    "Content-Type": "application/json"
}

def make_hf_call(payload):
    """Make a single Hugging Face API call."""
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload).json()
        if "error" in response:
            raise Exception(f"Hugging Face API error: {response['error']}")
        return response.get("generated_text", "No assessment available"), response.get("personalized_health_assessment", "No personalized assessment available")
    except Exception as e:
        return f"Error: {str(e)}", "Error in personalized assessment"

def parallel_hf_calls(payloads, ingredients, delay=0.5):
    """Parallelize Hugging Face API calls for a list of payloads, ensuring correct mapping to ingredients."""
    results = []
    # Create a mapping of payloads to ingredients
    payload_to_ingredient = {id(payload): ingredient for payload, ingredient in zip(payloads, ingredients)}
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Map each payload to a future
        future_to_payload = {
            executor.submit(make_hf_call, payload): payload
            for payload in payloads
        }
        # Collect results as they complete
        temp_results = {}
        for future in as_completed(future_to_payload):
            payload = future_to_payload[future]
            ingredient = payload_to_ingredient[id(payload)]
            try:
                safety_assessment, personalized_assessment = future.result()
                temp_results[ingredient] = (safety_assessment, personalized_assessment)
            except Exception as e:
                temp_results[ingredient] = (f"Error: {str(e)}", "Error in personalized assessment")
            time.sleep(delay)
        
        # Reconstruct results in the original order of ingredients
        for ingredient in ingredients:
            if ingredient in temp_results:
                results.append((payload_to_ingredient[id(payloads[ingredients.index(ingredient)])], temp_results[ingredient][0], temp_results[ingredient][1]))
    
    return results
# Apyflux API Setup
APYFLUX_URL = 'https://gateway.apyflux.com/search'
APYFLUX_HEADERS = {
    'x-app-id': '1948be40-3210-4d66-bb38-b660249ef2dc',
    'x-client-id': 'IkgJ0Vy71QguXXgxyWnmQNjjl8r2',
    'x-api-key': 'RP0Xae9lcNezS/XluNHwpxuyrewJYjmPBsSSVDE4pp4='
}

# RAG System Setup for Food Codes
loader = PyPDFLoader("cleanrag.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEYS[0]
)
filtered_splits = filter_complex_metadata(splits)
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

prompt = PromptTemplate.from_template("""
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

Question: {input}
Context: {context}
Answer:
""")
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Directory Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(SAVE_DIR, exist_ok=True)
DB_PATH = os.path.join(SAVE_DIR, "foodvisor.db")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
SPECIAL_CODES_LOG_PATH = os.path.join(LOGS_DIR, "special_codes.log")

# Initialize Database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        personal_details TEXT,
        maintenance_calories INTEGER,
        protein_requirement INTEGER,
        allergies TEXT,
        medical_info TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS food_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        food_id TEXT,
        food_name TEXT,
        serving_size REAL,
        nutrients TEXT,
        timestamp TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

# Session State
predefined_allergies = "nil"
predefined_medical_conditions = "nil"
personalDetails = ""
extractedText1 = ""
extractedText2 = ""
important_text_extracted_from_camera = ""
form_data = {
    "weight": "",
    "age": "",
    "height": "",
    "gender": "",
    "activity": "",
    "goal": "",
    "allergyInput": "",
    "medicalInfoInput": "",
    "currentStep": 1
}

messages = [
    {
        "role": "system",
        "content": (
            f"You are a medical assistant who works as a doctor with knowledge about allergies, dietary restrictions, and serious health conditions. "
            f"Users will provide their personal details- Age, weight, and height, as well as any allergies or medical conditions they have. "
            f"Your response should start with 'No' or 'Yes' if it's a question, followed by a brief explanation and a moderation suggestion. "
            f"Personal details: {personalDetails}. Predefined allergies: {predefined_allergies}. Predefined medical conditions: {predefined_medical_conditions}."
        )
    },
    {
        "role": "assistant",
        "content": "Hey, how can I help you today?"
    }
]

# Helper Functions
def log_special_code(code, rag_output):
    """Log special coded ingredients and their RAG outputs to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Code: {code}, RAG Output: {rag_output}\n"
    with open(SPECIAL_CODES_LOG_PATH, 'a') as log_file:
        log_file.write(log_entry)

def get_special_code_logs():
    """Retrieve the logs of special coded ingredients and their RAG outputs."""
    if not os.path.exists(SPECIAL_CODES_LOG_PATH):
        return []
    with open(SPECIAL_CODES_LOG_PATH, 'r') as log_file:
        return log_file.readlines()

def image_to_base64(image_data):
    """Convert image data (base64 string) to base64 format usable by Gemini API."""
    return image_data.split(',')[1]

def extract_text_with_gemini(image_data, report_type):
    """Extract and format text from an image using Gemini API."""
    image_base64 = image_to_base64(image_data)

    if report_type == "allergy":
        prompt = [
            {"text": """
            Extract all visible text from the image, which contains an allergy report. Structure the data in the following format:
            - Allergen: [name], Test Result: [result], Reference Range: [range]
            Example:
            - Allergen: Milk, Test Result: Positive Class 3, Reference Range: Negative
            If no allergy data is found, return an empty string. Return ONLY the structured data, no additional text.
            """},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
        ]
    else:  # medical
        prompt = [
            {"text": """
            Extract all visible text from the image, which contains a medical report. Structure the data in the following format:
            - Test: [name], Result: [value], Normal Range: [range], Status: [Abnormal/Normal]
            - Condition: [name], Type: [chronic/acute/etc.], Status: [diagnosed/self-reported]
            Example:
            - Test: Blood Glucose, Result: 150 mg/dL, Normal Range: 70-110 mg/dL, Status: Abnormal
            - Condition: Diabetes, Type: Chronic, Status: Diagnosed
            If no medical data is found, return an empty string. Return ONLY the structured data, no additional text.
            """},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
        ]

    try:
        response = make_gemini_call(prompt, model="gemini-2.5-flash")
        return response.text.strip()
    except Exception as e:
        print(f"Error extracting {report_type} report with Gemini: {str(e)}")
        return ""

def correct_ingredients_with_gemini(extracted_text):
    """Correct spelling and format ingredients using Gemini API, preserving special coded ingredients."""
    prompt = f"""
    The following text contains a list of food ingredients extracted via OCR, which may have spelling errors or improper formatting:
    {extracted_text}

    Please:
    1. Correct any spelling mistakes in the ingredients, but preserve any special coded ingredients (e.g., E100, INS 100, E 100, INS100) exactly as they appear without modification.
    2. Format the ingredients as a comma-separated list (e.g., "sugar, salt, water, E100").
    3. Remove any irrelevant text or OCR artifacts.
    4. Return ONLY the corrected ingredient list, no additional text.
    """
    try:
        response = make_gemini_call(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error correcting ingredients with Gemini: {str(e)}")
        return extracted_text

def clean_and_structure_extracted_data(extractedText1, extractedText2):
    try:
        allergy_prompt = f"""
        The following text contains allergy information extracted from user reports:
        {extractedText1}

        Please process this data and:
        1. Identify all allergy-related information
        2. Remove any OCR errors or irrelevant text
        3. Map severity based on class:
           - Class 1: Mild
           - Class 2: Mild
           - Class 3: Moderate
           - Class 4: Severe
           - Class 5: Severe
           - Class 6: Very Severe
        4. Structure the output in this format:
           - Allergen: [specific allergen]
             Type: [food/drug/environmental/etc.]
             Severity: [mild/moderate/severe/very severe]
             Reactions: [list of reactions, if any; otherwise empty list]

        Example Output:
        - Allergen: Peanuts
          Type: Food
          Severity: Severe
          Reactions: []

        Return ONLY the structured data, no additional commentary.
        """

        medical_prompt = f"""
        The following contains medical data from reports AND manual user input:
        {extractedText2}

        Analyze this data and:

        1. For LAB RESULTS:
           - STRICTLY Include ONLY values outside reference ranges and are VERY ABNORMAL
           - Show actual value vs normal range
           - STRICTLY Ignore borderline/normal results
           - Format as:
             - Test: [test name]
               Result: [value] (normal: [range])
               Significance: [clinical importance]

        2. For MANUALLY ENTERED CONDITIONS (like "I have diabetes"):
           - Include all clearly stated conditions
           - Format as:
             - Condition: [condition name]
               Type: [chronic/acute/etc.]
               Status: [self-reported/diagnosed]
               Notes: [any additional details]

        3. Final Output Rules:
           - Group lab abnormalities first
           - Then list manually entered conditions
           - Use exactly this format:

        === ABNORMAL LAB RESULTS ===
        - Test: [test name]
          Result: [value] (normal: [range])
          Significance: [clinical importance]

        === REPORTED CONDITIONS ===
        - Condition: [condition name]
          Type: [chronic/acute/etc.]
          Status: [self-reported/diagnosed]
          Notes: [any additional details]

        If no lab results or conditions are found, return empty sections:

        === ABNORMAL LAB RESULTS ===
        === REPORTED CONDITIONS ===

        Return NOTHING ELSE - no headers, no explanations.
        """
        
        cleaned_allergy = make_gemini_call(allergy_prompt).content
        cleaned_medical = make_gemini_call(medical_prompt).content
        
        print("\n=== Cleaned Allergy Data ===")
        print(cleaned_allergy)
        print("\n=== Cleaned Medical Data ===")
        print(cleaned_medical)
        
        return cleaned_allergy, cleaned_medical
    except Exception as e:
        print(f"Error in cleaning data: {str(e)}")
        return extractedText1, extractedText2

def process_barcode_image(barcode_data, is_image=True):
    """Process a barcode image or barcode number and fetch product details from OpenFoodFacts."""
    try:
        if is_image:
            # Decode base64 image
            image_base64 = image_to_base64(barcode_data)
            image_bytes = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None, "Failed to decode image. Ensure the uploaded file is a valid image."
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            barcodes = pyzbar.decode(img_gray)
            
            if not barcodes:
                return None, "No barcode found in the image. Please ensure the barcode is clear and visible."
            
            barcode = barcodes[0]
            barcode_data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            print(f"Found {barcode_type} barcode: {barcode_data}")
        else:
            # Use provided barcode number directly
            barcode_type = "Unknown"
            print(f"Processing barcode number: {barcode_data}")

        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode_data}.json"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return None, f"Failed to fetch product data from OpenFoodFacts: {str(e)}"
        
        data = response.json()
        if data.get("status") != 1:
            return None, "No product found for this barcode in OpenFoodFacts database."
        
        product = data.get("product", {})
        product_details = {
            "product_name": product.get("product_name", "N/A"),
            "brand": product.get("brands", "N/A"),
            "quantity": product.get("quantity", "N/A"),
            "ingredients": product.get("ingredients_text", "N/A"),
            "allergens": product.get("allergens", "N/A"),
            "nutrients": product.get("nutriments", {}),
            "country": product.get("countries", "N/A")
        }
        
        normalized_nutrients = {
            "energy-kcal_100g": f"{product_details['nutrients'].get('energy-kcal_100g', 'N/A')} kcal" if product_details['nutrients'].get('energy-kcal_100g') else "N/A",
            "sugars_100g": f"{product_details['nutrients'].get('sugars_100g', 'N/A')} g" if product_details['nutrients'].get('sugars_100g') else "N/A",
            "fat_100g": f"{product_details['nutrients'].get('fat_100g', 'N/A')} g" if product_details['nutrients'].get('fat_100g') else "N/A",
            "saturated-fat_100g": f"{product_details['nutrients'].get('saturated-fat_100g', 'N/A')} g" if product_details['nutrients'].get('saturated-fat_100g') else "N/A",
            "trans-fat_100g": f"{product_details['nutrients'].get('trans-fat_100g', 'N/A')} g" if product_details['nutrients'].get('trans-fat_100g') else "N/A",
            "sodium_100g": f"{product_details['nutrients'].get('sodium_100g', 'N/A')} mg" if product_details['nutrients'].get('sodium_100g') else "N/A",
            "carbohydrates_100g": f"{product_details['nutrients'].get('carbohydrates_100g', 'N/A')} g" if product_details['nutrients'].get('carbohydrates_100g') else "N/A",
            "proteins_100g": f"{product_details['nutrients'].get('proteins_100g', 'N/A')} g" if product_details['nutrients'].get('proteins_100g') else "N/A"
        }
        
        return {
            "barcode": barcode_data,
            "type": barcode_type,
            "product_details": product_details,
            "normalized_nutrients": normalized_nutrients
        }, None
    except Exception as e:
        print(f"Error processing barcode: {str(e)}")
        return None, f"Error processing barcode: {str(e)}"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/saveImage', methods=['POST'])
def save_image():
    global extractedText1, extractedText2, personalDetails, important_text_extracted_from_camera
    try:
        data = request.json
        personalDetails = data.get('personalDetails', '')
        allergy_report = data.get('allergyReport', '')
        medical_report = data.get('medicalReport', '')
        uploaded_image_data = data.get('imageSrc')

        if allergy_report:
            extractedText1 = extract_text_with_gemini(allergy_report, "allergy")
            print("\nExtracted Allergy Report Text (Gemini):", extractedText1)

        if medical_report:
            extractedText2 = extract_text_with_gemini(medical_report, "medical")
            print("\nExtracted Medical Report Text (Gemini):", extractedText2)

        if uploaded_image_data:
            # Save the image temporarily without processing
            image_data = base64.b64decode(image_to_base64(uploaded_image_data))
            image_path = os.path.join(SAVE_DIR, 'uploaded_image.jpg')
            with open(image_path, 'wb') as f:
                f.write(image_data)
            # Return the image path for further processing
            return jsonify({
                'success': True,
                'image_path': image_path,
                'extracted_text1': extractedText1,
                'extracted_text2': extractedText2
            }), 200
        else:
            return jsonify({
                'success': True,
                'extracted_text1': extractedText1,
                'extracted_text2': extractedText2
            }), 200
    except Exception as e:
        print("Error in /saveImage:", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/processImageAdjustments', methods=['POST'])
def process_image_adjustments():
    try:
        data = request.json
        image_data = data.get('imageSrc')
        brightness = float(data.get('brightness', 0))
        contrast = float(data.get('contrast', 1))
        crop = data.get('crop', None)
        rotation = float(data.get('rotation', 0))
        preview_only = data.get('preview', False)  # New flag for preview mode

        # Decode base64 image
        image_base64 = image_to_base64(image_data)
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'}), 400

        # Apply brightness and contrast adjustments
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

        # Apply cropping if provided
        if crop:
            x, y, width, height = crop['x'], crop['y'], crop['width'], crop['height']
            img = img[y:y+height, x:x+width]

        # Apply rotation if provided
        if rotation != 0:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

        # Convert adjusted image back to base64 for response
        _, buffer = cv2.imencode('.jpg', img)
        adjusted_image_base64 = base64.b64encode(buffer).decode()

        # Skip text extraction for preview mode
        if preview_only:
            return jsonify({
                'success': True,
                'adjusted_image': f'data:image/jpeg;base64,{adjusted_image_base64}'
            }), 200

        # Extract text using Tesseract (for final apply)
        temp_image_path = os.path.join(SAVE_DIR, 'adjusted_image.jpg')
        cv2.imwrite(temp_image_path, img)
        extracted_text = pytesseract.image_to_string(Image.open(temp_image_path))
        corrected_text = correct_ingredients_with_gemini(extracted_text)
        os.remove(temp_image_path)

        return jsonify({
            'success': True,
            'adjusted_image': f'data:image/jpeg;base64,{adjusted_image_base64}',
            'extracted_text': corrected_text
        }), 200
    except Exception as e:
        print(f"Error in /processImageAdjustments: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/ingredients', methods=['POST'])
def ingredients():
    try:
        ingredients = request.json.get('ingredients')
        print("Ingredients:", ingredients)

        with open(os.path.join(SAVE_DIR, 'ingredients.txt'), 'w') as f:
            f.write(ingredients)

        with open(os.path.join(SAVE_DIR, 'ingredients.txt'), 'r') as f:
            extracted_text = f.read()

        corrected_ingredients = correct_ingredients_with_gemini(extracted_text)
        print("\nCorrected Ingredients (Gemini):", corrected_ingredients)

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT allergies, medical_info FROM user_data ORDER BY id DESC LIMIT 1")
        user_data = c.fetchone()
        conn.close()

        allergies = user_data[0] if user_data else "None reported"
        medical_info = user_data[1] if user_data else "None reported"

        combined_text = (
            f"Allergies: {allergies}. "
            f"Medical conditions: {medical_info}. "
            f"Personal details: {personalDetails}. "
            f"Ingredients: {corrected_ingredients}. "
            f"Important extracted text: {important_text_extracted_from_camera}."
        )
        print("\nCombined Text:", combined_text)

        ingredient_analysis = analyze_ingredients(combined_text)
        return jsonify({
            'success': True,
            'extracted_text': corrected_ingredients,
            'ingredient_analysis': ingredient_analysis
        }), 200
    except Exception as e:
        print("Error in /ingredients:", e)
        return jsonify({'success': False, 'error': str(e)}), 500

def analyze_ingredients(combined_text):
    return "Analysis placeholder"

@app.route('/foodsafety', methods=['POST'])
def foodsafety():
    try:
        data = request.json
        ingredients = data.get('ingredients')
        model_type = data.get('model_type', 'pro')
        print("Ingredients for Food Safety Analysis:", ingredients)
        print("Model Type:", model_type)

        # Ensure ingredients are provided
        if not ingredients:
            return jsonify({'success': False, 'error': 'Ingredients are required'}), 400

        # Parse ingredients using the new function
        ingredient_list = parse_ingredients(ingredients)
        print("Parsed Ingredients:", ingredient_list)

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT allergies, medical_info FROM user_data ORDER BY id DESC LIMIT 1")
        user_data = c.fetchone()
        conn.close()

        allergies = user_data[0] if user_data else "None reported"
        medical_info = user_data[1] if user_data else "None reported"
        print("\nAllergies for Food Safety:", allergies)
        print("Medical Info for Food Safety:", medical_info)

        food_code_pattern = re.compile(r'^(E\s?\d{3}[a-zA-Z]?|INS\s?\d{3}[a-zA-Z]?|\d{3})$', re.IGNORECASE)
        special_codes = []
        special_codes_with_details = []
        ingredient_safety_assessments = []

        # Identify special codes
        for ingredient in ingredient_list:
            # Clean the ingredient for special code matching
            cleaned_ingredient = ingredient
            # Remove text in parentheses for matching purposes
            if "(" in ingredient and ")" in ingredient:
                cleaned_ingredient = re.sub(r'\s*\([^)]+\)', '', ingredient).strip()
            if food_code_pattern.match(cleaned_ingredient):
                normalized_code = cleaned_ingredient.replace(" ", "")
                print(f"Found special code: {normalized_code}")
                special_codes.append(normalized_code)

        if special_codes:
            # Parallel Gemini calls for RAG safety queries
            safety_queries = [f"Is {code} safe for consumption:" for code in special_codes]
            safety_results = parallel_gemini_calls(safety_queries, special_codes)
            
            # Parallel Gemini calls for personalized assessments
            personalized_prompts = [
                f"""
                Provide a personalized health assessment for the food additive '{code}' based on the user's medical and allergy data.

                User's Medical Conditions: {medical_info if medical_info else 'None'}
                User's Allergies: {allergies if allergies else 'None'}

                Format the response as:
                Personalized Health Assessment: [assessment]

                Return ONLY the formatted response, no additional text.
                """
                for code in special_codes
            ]
            personalized_results = parallel_gemini_calls(personalized_prompts, special_codes)

            # Process results
            for (safety_query, rag_safety_info), (pers_prompt, pers_result), code in zip(safety_results, personalized_results, special_codes):
                try:
                    log_special_code(code, rag_safety_info)
                    personalized_assessment = pers_result.split("Personalized Health Assessment: ")[1]
                except Exception as e:
                    print(f"Error processing {code}: {str(e)}")
                    rag_safety_info = "Unable to retrieve safety information."
                    log_special_code(code, rag_safety_info)
                    personalized_assessment = "Unable to assess due to lack of information."

                special_codes_with_details.append({
                    "additive": code,
                    "personalized_health_assessment": personalized_assessment
                })

                ingredient_safety_assessments.append({
                    "ingredient": code,
                    "safety_assessment": f"Food Code Safety (RAG): {rag_safety_info}",
                    "personalized_health_assessment": personalized_assessment
                })

        # Parallelize assessments for non-special-code ingredients
        non_special_ingredients = []
        for ingredient in ingredient_list:
            cleaned_ingredient = re.sub(r'\s*\([^)]+\)', '', ingredient).strip()
            if not food_code_pattern.match(cleaned_ingredient):
                # Use the full ingredient name (including parentheses) for assessment
                non_special_ingredients.append(ingredient)

        if non_special_ingredients:
            if model_type == "pro":
                # Prepare Hugging Face payloads
                hf_payloads = [
                    {
                        "inputs": f"[INST] Give safety assessment about the food ingredient '{ingredient}'. Highlight if it is harmful for any group of people or banned in any country due to its toxic nature. Provide a personalized health assessment based on the user's medical and allergy data, if it's generally safe then prompt it's safe and no concerns. [/INST]",
                        "medical_data": medical_info if medical_info else "None",
                        "allergy_data": allergies if allergies else "None",
                        "parameters": {}
                    }
                    for ingredient in non_special_ingredients
                ]
                # Parallel Hugging Face calls with ingredient mapping
                hf_results = parallel_hf_calls(hf_payloads, non_special_ingredients)

                # Process Hugging Face results, fall back to Gemini if needed
                for (_, safety_assessment, personalized_assessment), ingredient in zip(hf_results, non_special_ingredients):
                    if "Error" in safety_assessment:
                        print(f"Falling back to Gemini for {ingredient}: {safety_assessment}")
                        gemini_prompt = f"""
                        Give a safety assessment for the food ingredient '{ingredient}'. Highlight if it is harmful for any group of people or banned in any country due to its toxic nature. Provide a personalized health assessment based on the user's medical and allergy data.
                        User's Medical Conditions: {medical_info if medical_info else 'None'}
                        User's Allergies: {allergies if allergies else 'None'}
                        Format the response as:
                        Safety Assessment: [assessment]
                        Personalized Health Assessment: [assessment]
                        Return ONLY the formatted response, no additional text.
                        """
                        gemini_results = parallel_gemini_calls([gemini_prompt], [ingredient])
                        gemini_response = gemini_results[0][1]
                        try:
                            safety_assessment = gemini_response.split("Safety Assessment: ")[1].split("\n")[0]
                            personalized_assessment = gemini_response.split("Personalized Health Assessment: ")[1]
                        except Exception as e:
                            safety_assessment = "Error in Gemini fallback"
                            personalized_assessment = "Error in Gemini fallback"
                    ingredient_safety_assessments.append({
                        "ingredient": ingredient,
                        "safety_assessment": safety_assessment,
                        "personalized_health_assessment": personalized_assessment
                    })
            else:
                # Parallel Gemini calls for non-special ingredients
                gemini_prompts = [
                    f"""
                    Give a safety assessment for the food ingredient '{ingredient}'. Highlight if it is harmful for any group of people or banned in any country due to its toxic nature. Provide a personalized health assessment based on the user's medical and allergy data.
                    User's Medical Conditions: {medical_info if medical_info else 'None'}
                    User's Allergies: {allergies if allergies else 'None'}
                    Format the response as:
                    Safety Assessment: [assessment]
                    Personalized Health Assessment: [assessment]
                    Return ONLY the formatted response, no additional text.
                    """
                    for ingredient in non_special_ingredients
                ]
                gemini_results = parallel_gemini_calls(gemini_prompts, non_special_ingredients)

                for (_, gemini_response), ingredient in zip(gemini_results, non_special_ingredients):
                    try:
                        safety_assessment = gemini_response.split("Safety Assessment: ")[1].split("\n")[0]
                        personalized_assessment = gemini_response.split("Personalized Health Assessment: ")[1]
                    except Exception as e:
                        safety_assessment = "Error in Gemini response"
                        personalized_assessment = "Error in Gemini response"
                    ingredient_safety_assessments.append({
                        "ingredient": ingredient,
                        "safety_assessment": safety_assessment,
                        "personalized_health_assessment": personalized_assessment
                    })

        prompt = f"""
        You are a food safety and nutrition analysis expert. Analyze the following ingredients for a 100g serving and return a structured JSON object with food safety and nutritional data.

        Ingredients: {ingredients}
        User's Medical Conditions: {medical_info if medical_info else 'None'}
        User's Allergies: {allergies if allergies else 'None'}

        Return the analysis in this JSON format:
        - "processing_level": A string (e.g., "Highly Processed", "Minimally Processed", "Unprocessed")
        - "hydrogenated_oil": A string (e.g., "Present", "Not Present")
        - "additives": An array of objects, each with "name" (e.g., "E621") and "description" (e.g., "Monosodium Glutamate, may cause headaches")
        - "energy": A string (e.g., "250 kcal")
        - "total_sugars": A string (e.g., "15g")
        - "total_fat": A string (e.g., "10g")
        - "saturated_fat": A string (e.g., "5g")
        - "trans_fat": A string (e.g., "0g")
        - "cholesterol": A string (e.g., "20mg")
        - "sodium": A string (e.g., "300mg")
        - "nutrients": An array of objects, each with "name" (e.g., "Protein") and "amount" (e.g., "5g")
        - "ingredients": A string listing all ingredients (e.g., "Flour, Sugar, Butter, E621")

        If any information is unavailable, use "N/A" for strings or an empty array for additives and nutrients.
        Ensure the response is valid JSON. Return ONLY the JSON object, with no additional text.
        """

        response = make_gemini_call(prompt)
        raw_response = response.content
        print("Raw Gemini API Response for /foodsafety:", raw_response)

        try:
            foodsafety_analysis = json.loads(raw_response)
        except json.JSONDecodeError as e:
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                foodsafety_analysis = json.loads(json_match.group(0))
            else:
                return jsonify({'success': False, 'error': 'Gemini API response is not valid JSON'}), 500

        concerns_and_likes = generate_concerns_and_likes(foodsafety_analysis, medical_info, allergies, special_codes_with_details)

        messages.append({
            "type": "foodsafety",
            "ingredients_FS": ingredients,
            "foodsafetyAnalysis": foodsafety_analysis,
            "ingredientSafetyAssessments": ingredient_safety_assessments,
            "specialCodes": special_codes,
            "specialCodesWithDetails": special_codes_with_details,
            "concernsAndLikes": concerns_and_likes
        })

        return jsonify({
            'success': True,
            'foodsafety_analysis': foodsafety_analysis,
            'ingredient_safety_assessments': ingredient_safety_assessments,
            'special_codes': special_codes,
            'special_codes_with_details': special_codes_with_details,
            'special_code_logs': get_special_code_logs(),
            'concerns_and_likes': concerns_and_likes
        }), 200
    except Exception as e:
        print("Error in /foodsafety:", e)
        return jsonify({'success': False, 'error': str(e)}), 500

def generate_concerns_and_likes(foodsafety_analysis, medical_conditions, allergies, special_codes_with_details):
    concerns = []
    likes = []
    recommendation = ""

    SUGAR_THRESHOLD = 20
    FAT_THRESHOLD = 20
    SATURATED_FAT_THRESHOLD = 5
    TRANS_FAT_THRESHOLD = 0
    CHOLESTEROL_THRESHOLD = 30
    SODIUM_THRESHOLD = 400

    def extract_numeric(value):
        if isinstance(value, str):
            numeric = re.sub(r'[^\d.]', '', value)
            try:
                return float(numeric)
            except ValueError:
                return 0
        return 0

    if foodsafety_analysis.get("total_sugars", "N/A") != "N/A":
        sugar_value = extract_numeric(foodsafety_analysis["total_sugars"])
        if sugar_value > SUGAR_THRESHOLD:
            concerns.append({"label": "Total Sugars", "value": f"{foodsafety_analysis['total_sugars']} - High sugar content"})
        else:
            likes.append({"label": "Total Sugars", "value": f"{foodsafety_analysis['total_sugars']} - Reasonable sugar level"})

    if foodsafety_analysis.get("total_fat", "N/A") != "N/A":
        fat_value = extract_numeric(foodsafety_analysis["total_fat"])
        if fat_value > FAT_THRESHOLD:
            concerns.append({"label": "Total Fat", "value": f"{foodsafety_analysis['total_fat']} - High fat content"})
        else:
            likes.append({"label": "Total Fat", "value": f"{foodsafety_analysis['total_fat']} - Acceptable fat level"})

    if foodsafety_analysis.get("saturated_fat", "N/A") != "N/A":
        sat_fat_value = extract_numeric(foodsafety_analysis["saturated_fat"])
        if sat_fat_value > SATURATED_FAT_THRESHOLD:
            concerns.append({"label": "Saturated Fat", "value": f"{foodsafety_analysis['saturated_fat']} - High saturated fat"})
        else:
            likes.append({"label": "Saturated Fat", "value": f"{foodsafety_analysis['saturated_fat']} - Low saturated fat"})

    if foodsafety_analysis.get("trans_fat", "N/A") != "N/A":
        trans_fat_value = extract_numeric(foodsafety_analysis["trans_fat"])
        if trans_fat_value > TRANS_FAT_THRESHOLD:
            concerns.append({"label": "Trans Fat", "value": f"{foodsafety_analysis['trans_fat']} - Contains trans fat"})
        else:
            likes.append({"label": "Trans Fat", "value": f"{foodsafety_analysis['trans_fat']} - Good for heart health"})

    if foodsafety_analysis.get("cholesterol", "N/A") != "N/A":
        cholesterol_value = extract_numeric(foodsafety_analysis["cholesterol"])
        if cholesterol_value > CHOLESTEROL_THRESHOLD:
            concerns.append({"label": "Cholesterol", "value": f"{foodsafety_analysis['cholesterol']} - High cholesterol"})
        else:
            likes.append({"label": "Cholesterol", "value": f"{foodsafety_analysis['cholesterol']} - Low cholesterol content"})

    if foodsafety_analysis.get("sodium", "N/A") != "N/A":
        sodium_value = extract_numeric(foodsafety_analysis["sodium"])
        if sodium_value > SODIUM_THRESHOLD:
            concerns.append({"label": "Sodium", "value": f"{foodsafety_analysis['sodium']} - High sodium content"})
        else:
            likes.append({"label": "Sodium", "value": f"{foodsafety_analysis['sodium']} - Acceptable sodium level"})

    if foodsafety_analysis.get("processing_level", "N/A") != "N/A":
        if "Highly Processed" in foodsafety_analysis["processing_level"]:
            concerns.append({"label": "Processing Level", "value": f"{foodsafety_analysis['processing_level']} - May lack nutrients"})
        else:
            likes.append({"label": "Processing Level", "value": f"{foodsafety_analysis['processing_level']} - Less processed"})

    if foodsafety_analysis.get("hydrogenated_oil", "N/A") != "N/A":
        if foodsafety_analysis["hydrogenated_oil"] == "Present":
            concerns.append({"label": "Hydrogenated Oil", "value": "Present - May increase heart disease risk"})
        else:
            likes.append({"label": "Hydrogenated Oil", "value": "Not Present - Better for heart health"})

    if foodsafety_analysis.get("additives", []):
        for additive in foodsafety_analysis["additives"]:
            if "may cause" in additive.get("description", "").lower() or "banned" in additive.get("description", "").lower():
                concerns.append({"label": "Additive Concern", "value": f"{additive['name']}: {additive['description']}"})
            else:
                likes.append({"label": "Additive", "value": f"{additive['name']}: Generally safe"})

    if allergies and allergies != "None":
        ingredients = foodsafety_analysis.get("ingredients", "N/A").lower()
        allergy_list = []
        for line in allergies.split("\n"):
            if line.startswith("- Allergen:"):
                allergen = line.split("Allergen:")[1].split("\n")[0].strip()
                allergy_list.append(allergen.lower())
        for allergen in allergy_list:
            if allergen in ingredients:
                concerns.append({"label": "Allergen Alert", "value": f"Contains {allergen}, which you are allergic to"})

    if medical_conditions and medical_conditions != "None":
        medical_list = []
        for line in medical_conditions.split("\n"):
            if line.startswith("- Condition:"):
                condition = line.split("Condition:")[1].split("\n")[0].strip().lower()
                medical_list.append(condition)
        for condition in medical_list:
            if condition == "diabetes":
                sugar_value = extract_numeric(foodsafety_analysis.get("total_sugars", "0g"))
                if sugar_value > SUGAR_THRESHOLD:
                    concerns.append({"label": "Diabetes Concern", "value": f"High sugar ({foodsafety_analysis['total_sugars']}) - Unsuitable for diabetic users"})
            if condition == "hypertension":
                sodium_value = extract_numeric(foodsafety_analysis.get("sodium", "0mg"))
                if sodium_value > SODIUM_THRESHOLD:
                    concerns.append({"label": "Hypertension Concern", "value": f"High sodium ({foodsafety_analysis['sodium']}) - Unsuitable for hypertension"})

    for code in special_codes_with_details:
        assessment = code["personalized_health_assessment"].lower()
        if "avoid" in assessment or "unsafe" in assessment or "allergic" in assessment:
            concerns.append({"label": "Special Code Concern", "value": f"{code['additive']}: {code['personalized_health_assessment']}"})

    if concerns:
        prompt = f"""
        Based on the following concerns and user health data, provide a concise recommendation (1-2 sentences) for the user regarding the analyzed food product.

        Concerns: {', '.join([f"{item['label']}: {item['value']}" for item in concerns])}
        User's Medical Conditions: {medical_conditions if medical_conditions else 'None'}
        User's Allergies: {allergies if allergies else 'None'}

        Return ONLY the recommendation, no additional text.
        """
        recommendation = make_gemini_call(prompt).content
    else:
        recommendation = "This product appears safe based on the analysis and your health profile, but consume in moderation."

    return {
        "concerns": concerns,
        "likes": likes,
        "recommendation": recommendation
    }

@app.route('/saveCombinedData', methods=['POST'])
def save_combined_data():
    global personalDetails, extractedText1, extractedText2, form_data
    try:
        weight = request.form.get('weight')
        age = request.form.get('age')
        height = request.form.get('height')
        gender = request.form.get('gender')
        activity = request.form.get('activity')
        goal = request.form.get('goal')
        maintenance_calories = request.form.get('maintenance_calories')
        protein_requirement = request.form.get('protein_requirement')
        allergy_text = request.form.get('allergyInput', '')
        medical_text = request.form.get('medicalInfoInput', '')

        form_data.update({
            "weight": weight,
            "age": age,
            "height": height,
            "gender": gender,
            "activity": activity,
            "goal": goal,
            "allergyInput": allergy_text,
            "medicalInfoInput": medical_text
        })

        personalDetails = f"Weight: {weight} kg, Age: {age} years, Height: {height} cm, Gender: {gender}, Activity Level: {activity}, Fitness Goal: {goal}"

        extractedText1 = allergy_text
        extractedText2 = medical_text

        medical_files = request.files.getlist('medicalFiles')
        for i, file in enumerate(medical_files):
            if file.filename != '':
                temp_filename = f"medical_{i}_{file.filename}"
                temp_path = os.path.join(SAVE_DIR, temp_filename)
                file.save(temp_path)

                with open(temp_path, "rb") as img_file:
                    image_base64 = base64.b64encode(img_file.read()).decode()

                prompt = [
                    {"text": """
                    Extract all visible text from the image, which contains a medical report. Structure the data in the following format:
                    - Test: [name], Result: [value], Normal Range: [range], Status: [Abnormal/Normal]
                    - Condition: [name], Type: [chronic/acute/etc.], Status: [diagnosed/self-reported]
                    Example:
                    - Test: Blood Glucose, Result: 150 mg/dL, Normal Range: 70-110 mg/dL, Status: Abnormal
                    - Condition: Diabetes, Type: Chronic, Status: Diagnosed
                    If no medical data is found, return an empty string. Return ONLY the structured data, no additional text.
                    """},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                ]
                response = make_gemini_call(prompt, model="gemini-2.5-flash")
                extracted_text = response.text.strip()
                extractedText2 += f"\n\n[Document {i+1} - {file.filename}]:\n{extracted_text}"
                os.remove(temp_path)

        allergy_files = request.files.getlist('allergyFiles')
        for i, file in enumerate(allergy_files):
            if file.filename != '':
                temp_filename = f"allergy_{i}_{file.filename}"
                temp_path = os.path.join(SAVE_DIR, temp_filename)
                file.save(temp_path)

                with open(temp_path, "rb") as img_file:
                    image_base64 = base64.b64encode(img_file.read()).decode()

                prompt = [
                    {"text": """
                    Extract all visible text from the image, which contains an allergy report. Structure the data in the following format:
                    - Allergen: [name], Test Result: [result], Reference Range: [range]
                    Example:
                    - Allergen: Milk, Test Result: Positive Class 3, Reference Range: Negative
                    If no allergy data is found, return an empty string. Return ONLY the structured data, no additional text.
                    """},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                ]
                response = make_gemini_call(prompt, model="gemini-2.5-flash")
                extracted_text = response.text.strip()
                extractedText1 += f"\n\n[Document {i+1} - {file.filename}]:\n{extracted_text}"
                os.remove(temp_path)

        print("\n=== Final Extracted Texts ===")
        print("Allergies (extractedText1):", extractedText1)
        print("Medical Info (extractedText2):", extractedText2)

        cleaned_allergy, cleaned_medical = clean_and_structure_extracted_data(extractedText1, extractedText2)

        print("\n=== Final Structured Data ===")
        print("Allergies:", cleaned_allergy)
        print("Medical Info:", cleaned_medical)

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO user_data (personal_details, maintenance_calories, protein_requirement, allergies, medical_info)
                     VALUES (?, ?, ?, ?, ?)''',
                  (personalDetails, maintenance_calories, protein_requirement, cleaned_allergy, cleaned_medical))
        user_id = c.lastrowid
        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'message': 'Data saved and processed successfully',
            'user_id': user_id,
            'num_medical_files': len(medical_files),
            'num_allergy_files': len(allergy_files)
        })
    except Exception as e:
        print("Error in save_combined_data:", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
def parse_ingredients(ingredients_str):
    """
    Parse a comma-separated ingredients string, handling nested parentheses.
    Returns a list of cleaned ingredient names.
    """
    if not ingredients_str or ingredients_str == "N/A":
        return []

    result = []
    current = ""
    paren_count = 0
    i = 0

    while i < len(ingredients_str):
        char = ingredients_str[i]

        if char == "(":
            paren_count += 1
            current += char
        elif char == ")":
            paren_count -= 1
            current += char
        elif char == "," and paren_count == 0:
            # Only split on commas outside of parentheses
            cleaned = current.strip()
            if cleaned:
                result.append(cleaned)
            current = ""
        else:
            current += char
        i += 1

    # Add the last ingredient
    cleaned = current.strip()
    if cleaned:
        result.append(cleaned)

    # Further clean up: handle cases like "Emulsifiers (E322 (Soy Lecithin), E471)"
    final_result = []
    for item in result:
        # Remove labels like "Emulsifiers", "Colourings" and extract nested ingredients
        if "(" in item and ")" in item:
            # Extract content inside the outermost parentheses
            content = item[item.find("(")+1:item.rfind(")")]
            if "," in content:
                # Split nested ingredients inside parentheses
                nested_items = parse_ingredients(content)
                final_result.extend(nested_items)
            else:
                final_result.append(content.strip())
        else:
            final_result.append(item.strip())

    return final_result

@app.route('/geminiAnalyze', methods=['POST'])
def gemini_analyze():
    try:
        data = request.json
        food_name = data.get('foodName')
        serving_size = data.get('servingSize', '100g')
        model_type = data.get('modelType', 'pro')

        if not food_name:
            return jsonify({'success': False, 'error': 'Food name is required'}), 400

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT allergies, medical_info FROM user_data ORDER BY id DESC LIMIT 1")
        user_data = c.fetchone()
        conn.close()

        allergies = user_data[0] if user_data else "None reported"
        medical_info = user_data[1] if user_data else "None reported"
        print("\nAllergies for Gemini Analyze:", allergies)
        print("Medical Info for Gemini Analyze:", medical_info)

        prompt = f"""
        You are a food safety and nutrition analysis expert. I will provide you with a food name, and you need to analyze its nutritional content and safety for a 100g serving. Return the results in a structured JSON format with the following sections:

        - "processing_level": A string describing the processing level (e.g., "Highly Processed", "Minimally Processed", "Unprocessed").
        - "hydrogenated_oil": A string indicating presence (e.g., "Present", "Not Present").
        - "additives": An array of objects, each with "name" (e.g., "E621") and "description" (e.g., "Monosodium Glutamate, may cause headaches").
        - "energy": A string (e.g., "250 kcal").
        - "total_sugars": A string (e.g., "15g").
        - "total_fat": A string (e.g., "10g").
        - "saturated_fat": A string (e.g., "5g").
        - "trans_fat": A string (e.g., "0g").
        - "cholesterol": A string (e.g., "20mg").
        - "sodium": A string (e.g., "300mg").
        - "nutrients": An array of objects, each with "name" (e.g., "Protein") and "amount" (e.g., "5g").
        - "ingredients": A string listing all ingredients (e.g., "Flour, Sugar, Butter, E621").

        If any information is unavailable, use "N/A" for strings or an empty array for additives and nutrients. Ensure the response is valid JSON. Return ONLY the JSON object, with no additional text.

        Food name: {food_name}
        Serving size: {serving_size}
        User's Medical Conditions: {medical_info if medical_info else 'None'}
        User's Allergies: {allergies if allergies else 'None'}
        """

        response = make_gemini_call(prompt)
        raw_response = response.content
        print("Raw Gemini API Response for /geminiAnalyze:", raw_response)

        try:
            foodsafety_analysis = json.loads(raw_response)
        except json.JSONDecodeError as e:
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                foodsafety_analysis = json.loads(json_match.group(0))
            else:
                return jsonify({'success': False, 'error': 'Gemini API response is not valid JSON'}), 500

        ingredients = foodsafety_analysis.get("ingredients", "N/A")
        # Use the new parse_ingredients function
        ingredient_list = parse_ingredients(ingredients)
        print("Parsed Ingredients:", ingredient_list)

        food_code_pattern = re.compile(r'^(E\s?\d{3}[a-zA-Z]?|INS\s?\d{3}[a-zA-Z]?|\d{3})$', re.IGNORECASE)
        special_codes = []
        special_codes_with_details = []
        ingredient_safety_assessments = []

        # Identify special codes
        for ingredient in ingredient_list:
            # Clean the ingredient for special code matching
            cleaned_ingredient = ingredient
            # Remove text in parentheses for matching purposes
            if "(" in ingredient and ")" in ingredient:
                cleaned_ingredient = re.sub(r'\s*\([^)]+\)', '', ingredient).strip()
            if food_code_pattern.match(cleaned_ingredient):
                normalized_code = cleaned_ingredient.replace(" ", "")
                print(f"Found special code in geminiAnalyze: {normalized_code}")
                special_codes.append(normalized_code)

        if special_codes:
            # Parallel Gemini calls for RAG safety queries
            safety_queries = [f"Is {code} safe for consumption:" for code in special_codes]
            safety_results = parallel_gemini_calls(safety_queries, special_codes)

            # Parallel Gemini calls for personalized assessments
            personalized_prompts = [
                f"""
                Provide a personalized health assessment for the food additive '{code}' based on the user's medical and allergy data.

                User's Medical Conditions: {medical_info if medical_info else 'None'}
                User's Allergies: {allergies if allergies else 'None'}

                Format the response as:
                Personalized Health Assessment: [assessment]

                Return ONLY the formatted response, no additional text.
                """
                for code in special_codes
            ]
            personalized_results = parallel_gemini_calls(personalized_prompts, special_codes)

            # Process results
            for (safety_query, rag_safety_info), (pers_prompt, pers_result), code in zip(safety_results, personalized_results, special_codes):
                try:
                    log_special_code(code, rag_safety_info)
                    personalized_assessment = pers_result.split("Personalized Health Assessment: ")[1]
                except Exception as e:
                    print(f"Error querying RAG system for {code}: {str(e)}")
                    rag_safety_info = "Unable to retrieve safety information."
                    log_special_code(code, rag_safety_info)
                    personalized_assessment = "Unable to assess due to lack of information."

                special_codes_with_details.append({
                    "additive": code,
                    "personalized_health_assessment": personalized_assessment
                })

                ingredient_safety_assessments.append({
                    "ingredient": code,
                    "safety_assessment": f"Food Code Safety (RAG): {rag_safety_info}",
                    "personalized_health_assessment": personalized_assessment
                })

        # Parallelize assessments for non-special-code ingredients
        non_special_ingredients = []
        for ingredient in ingredient_list:
            cleaned_ingredient = re.sub(r'\s*\([^)]+\)', '', ingredient).strip()
            if not food_code_pattern.match(cleaned_ingredient):
                # Use the full ingredient name (including parentheses) for assessment
                non_special_ingredients.append(ingredient)

        if non_special_ingredients:
            if model_type == "pro":
                # Prepare Hugging Face payloads
                hf_payloads = [
                    {
                        "inputs": f"[INST] Give safety assessment about the food ingredient '{ingredient}'. Highlight if it is harmful for any group of people or banned in any country due to its toxic nature. Provide a personalized health assessment based on the user's medical and allergy data, if it's generally safe then prompt it's safe and no concerns. [/INST]",
                        "medical_data": medical_info if medical_info else "None",
                        "allergy_data": allergies if allergies else "None",
                        "parameters": {}
                    }
                    for ingredient in non_special_ingredients
                ]
                # Parallel Hugging Face calls
                hf_results = parallel_hf_calls(hf_payloads, non_special_ingredients)

                # Process Hugging Face results, fall back to Gemini if needed
                for (payload, safety_assessment, personalized_assessment), ingredient in zip(hf_results, non_special_ingredients):
                    if "Error" in safety_assessment:
                        print(f"Falling back to Gemini for {ingredient}: {safety_assessment}")
                        gemini_prompt = f"""
                        Give safety assessment about the food ingredient '{ingredient}'. Highlight if it is harmful for any group of people or banned in any country due to its toxic nature. Provide a personalized health assessment based on the user's medical and allergy data, if it's generally safe then prompt it's safe and no concerns.
                        User's Medical Conditions: {medical_info if medical_info else 'None'}
                        User's Allergies: {allergies if allergies else 'None'}
                        Format the response as:
                        Safety Assessment: [assessment]
                        Personalized Health Assessment: [assessment]
                        Return ONLY the formatted response, no additional text.
                        """
                        gemini_results = parallel_gemini_calls([gemini_prompt], [ingredient])
                        gemini_response = gemini_results[0][1]
                        try:
                            safety_assessment = gemini_response.split("Safety Assessment: ")[1].split("\n")[0]
                            personalized_assessment = gemini_response.split("Personalized Health Assessment: ")[1]
                        except Exception as e:
                            safety_assessment = "Error in Gemini fallback"
                            personalized_assessment = "Error in Gemini fallback"
                    ingredient_safety_assessments.append({
                        "ingredient": ingredient,
                        "safety_assessment": safety_assessment,
                        "personalized_health_assessment": personalized_assessment
                    })
            else:
                # Parallel Gemini calls for non-special ingredients
                gemini_prompts = [
                    f"""
                    Give safety assessment about the food ingredient '{ingredient}'. Highlight if it is harmful for any group of people or banned in any country due to its toxic nature. Provide a personalized health assessment based on the user's medical and allergy data, if it's generally safe then prompt it's safe and no concerns.
                    User's Medical Conditions: {medical_info if medical_info else 'None'}
                    User's Allergies: {allergies if allergies else 'None'}
                    Format the response as:
                    Safety Assessment: [assessment]
                    Personalized Health Assessment: [assessment]
                    Return ONLY the formatted response, no additional text.
                    """
                    for ingredient in non_special_ingredients
                ]
                gemini_results = parallel_gemini_calls(gemini_prompts, non_special_ingredients)

                for (prompt, gemini_response), ingredient in zip(gemini_results, non_special_ingredients):
                    try:
                        safety_assessment = gemini_response.split("Safety Assessment: ")[1].split("\n")[0]
                        personalized_assessment = gemini_response.split("Personalized Health Assessment: ")[1]
                    except Exception as e:
                        safety_assessment = "Error in Gemini response"
                        personalized_assessment = "Error in Gemini response"
                    ingredient_safety_assessments.append({
                        "ingredient": ingredient,
                        "safety_assessment": safety_assessment,
                        "personalized_health_assessment": personalized_assessment
                    })

        concerns_and_likes = generate_concerns_and_likes(foodsafety_analysis, medical_info, allergies, special_codes_with_details)

        messages.append({
            "type": "foodsafety",
            "foodName": food_name,
            "ingredients_FS": foodsafety_analysis.get("ingredients", "N/A"),
            "foodsafetyAnalysis": foodsafety_analysis,
            "specialCodes": special_codes,
            "specialCodesWithDetails": special_codes_with_details,
            "ingredientSafetyAssessments": ingredient_safety_assessments,
            "concernsAndLikes": concerns_and_likes
        })

        return jsonify({
            'success': True,
            'foodsafety_analysis': foodsafety_analysis,
            'special_codes': special_codes,
            'special_codes_with_details': special_codes_with_details,
            'special_code_logs': get_special_code_logs(),
            'ingredient_safety_assessments': ingredient_safety_assessments,
            'concerns_and_likes': concerns_and_likes
        }), 200
    except Exception as e:
        print("Error in /geminiAnalyze:", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/processBarcode', methods=['POST'])
def process_barcode():
    try:
        data = request.json
        barcode_image = data.get('barcodeImage')
        barcode_number = data.get('barcodeNumber')
        model_type = data.get('model_type', 'pro')
        
        if not barcode_image and not barcode_number:
            return jsonify({'success': False, 'error': 'Either barcode image or barcode number is required'}), 400
        
        if barcode_image:
            result, error = process_barcode_image(barcode_image, is_image=True)
        else:
            result, error = process_barcode_image(barcode_number, is_image=False)
        
        print("Barcode processing result:", result)
        
        if error:
            return jsonify({'success': False, 'error': error}), 400
        
        ingredients = result['product_details'].get('ingredients', 'N/A')
        normalized_nutrients = result['product_details'].get('nutrients', {})
        
        if ingredients == 'N/A':
            return jsonify({'success': False, 'error': 'No ingredients found for this product'}), 400
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT allergies, medical_info FROM user_data ORDER BY id DESC LIMIT 1")
        user_data = c.fetchone()
        conn.close()

        allergies = user_data[0] if user_data else "None reported"
        medical_info = user_data[1] if user_data else "None reported"
        print("\nAllergies for Barcode Processing:", allergies)
        print("Medical Info for Barcode Processing:", medical_info)
        
        corrected_ingredients = correct_ingredients_with_gemini(ingredients)
        
        foodsafety_analysis = {
            "processing_level": "N/A",
            "hydrogenated_oil": "Present" if "hydrogenated" in corrected_ingredients.lower() else "Not Present",
            "additives": [],
            "energy": normalized_nutrients.get('energy-kcal_100g', 'N/A'),
            "total_sugars": normalized_nutrients.get('sugars_100g', 'N/A'),
            "total_fat": normalized_nutrients.get('fat_100g', 'N/A'),
            "saturated_fat": normalized_nutrients.get('saturated-fat_100g', 'N/A'),
            "trans_fat": normalized_nutrients.get('trans-fat_100g', 'N/A'),
            "cholesterol": "N/A",
            "sodium": normalized_nutrients.get('sodium_100g', 'N/A'),
            "nutrients": [
                {"name": "Carbohydrates", "amount": normalized_nutrients.get('carbohydrates_100g', 'N/A')},
                {"name": "Protein", "amount": normalized_nutrients.get('proteins_100g', 'N/A')}
            ],
            "ingredients": corrected_ingredients
        }
        
        ingredient_list = [ingredient.strip() for ingredient in corrected_ingredients.split(',') if ingredient.strip()]
        food_code_pattern = re.compile(r'^(E\s?\d{3}[a-zA-Z]?|INS\s?\d{3}[a-zA-Z]?|\d{3})$', re.IGNORECASE)
        special_codes = []
        special_codes_with_details = []
        ingredient_safety_assessments = []
        
        for ingredient in ingredient_list:
            if food_code_pattern.match(ingredient):
                normalized_code = ingredient.replace(" ", "")
                special_codes.append(normalized_code)

        if special_codes:
            # Parallel Gemini calls for RAG safety queries
            safety_queries = [f"Is {code} safe for consumption:" for code in special_codes]
            safety_results = parallel_gemini_calls(safety_queries, special_codes)
            # Parallel Gemini calls for personalized assessments
            personalized_prompts = [
                f"""
                Provide a personalized health assessment for the food additive '{code}' based on the user's medical and allergy data.
                User's Medical Conditions: {medical_info if medical_info else 'None'}
                User's Allergies: {allergies if allergies else 'None'}
                Format the response as:
                Personalized Health Assessment: [assessment]
                Return ONLY the formatted response, no additional text.
                """
                for code in special_codes
            ]
            # In /processBarcode route, after safety_results
            personalized_results = parallel_gemini_calls(personalized_prompts, special_codes)
            # Process results
            for (safety_query, rag_safety_info), (pers_prompt, pers_result), code in zip(safety_results, personalized_results, special_codes):
                try:
                    log_special_code(code, rag_safety_info)
                    personalized_assessment = pers_result.split("Personalized Health Assessment: ")[1]
                except Exception as e:
                    print(f"Error querying RAG system for {code}: {str(e)}")
                    rag_safety_info = "Unable to retrieve safety information."
                    log_special_code(code, rag_safety_info)
                    personalized_assessment = "Unable to assess due to lack of information."

                special_codes_with_details.append({
                    "additive": code,
                    "personalized_health_assessment": personalized_assessment
                })
                ingredient_safety_assessments.append({
                    "ingredient": code,
                    "safety_assessment": f"Food Code Safety (RAG): {rag_safety_info}",
                    "personalized_health_assessment": personalized_assessment
                })

        # Parallelize assessments for non-special-code ingredients
        non_special_ingredients = [ingredient for ingredient in ingredient_list if not food_code_pattern.match(ingredient)]
        if non_special_ingredients:
            if model_type == "pro":
                # Prepare Hugging Face payloads
                hf_payloads = [
                    {
                        "inputs": f"[INST] Give safety assessment about the food ingredient '{ingredient}'. Highlight if it is harmful for any group of people or banned in any country due to its toxic nature. Provide a personalized health assessment based on the user's medical and allergy data, if it's generally safe then prompt it's safe and no concerns. [/INST]",
                        "medical_data": medical_info if medical_info else "None",
                        "allergy_data": allergies if allergies else "None",
                        "parameters": {}
                    }
                    for ingredient in non_special_ingredients
                ]
                # Parallel Hugging Face calls
                hf_results = parallel_hf_calls(hf_payloads, non_special_ingredients)

                # Process Hugging Face results, fall back to Gemini if needed
                for (payload, safety_assessment, personalized_assessment), ingredient in zip(hf_results, non_special_ingredients):
                    if "Error" in safety_assessment:
                        print(f"Falling back to Gemini for {ingredient}: {safety_assessment}")
                        gemini_prompt = f"""
                        Give safety assessment about the food ingredient '{ingredient}'. Highlight if it is harmful for any group of people or banned in any country due to its toxic nature. Provide a personalized health assessment based on the user's medical and allergy data, if it's generally safe then prompt it's safe and no concerns.
                        User's Medical Conditions: {medical_info if medical_info else 'None'}
                        User's Allergies: {allergies if allergies else 'None'}
                        Format the response as:
                        Safety Assessment: [assessment]
                        Personalized Health Assessment: [assessment]
                        Return ONLY the formatted response, no additional text.
                        """
                        gemini_results = parallel_gemini_calls([gemini_prompt], [ingredient])
                        gemini_response = gemini_results[0][1]
                        try:
                            safety_assessment = gemini_response.split("Safety Assessment: ")[1].split("\n")[0]
                            personalized_assessment = gemini_response.split("Personalized Health Assessment: ")[1]
                        except Exception as e:
                            safety_assessment = "Error in Gemini fallback"
                            personalized_assessment = "Error in Gemini fallback"
                    ingredient_safety_assessments.append({
                        "ingredient": ingredient,
                        "safety_assessment": safety_assessment,
                        "personalized_health_assessment": personalized_assessment
                    })
            else:
                # Parallel Gemini calls for non-special ingredients
                gemini_prompts = [
                    f"""
                    Give safety assessment about the food ingredient '{ingredient}'. Highlight if it is harmful for any group of people or banned in any country due to its toxic nature. Provide a personalized health assessment based on the user's medical and allergy data, if it's generally safe then prompt it's safe and no concerns.
                    User's Medical Conditions: {medical_info if medical_info else 'None'}
                    User's Allergies: {allergies if allergies else 'None'}
                    Format the response as:
                    Safety Assessment: [assessment]
                    Personalized Health Assessment: [assessment]
                    Return ONLY the formatted response, no additional text.
                    """
                    for ingredient in non_special_ingredients
                ]
                gemini_results = parallel_gemini_calls(gemini_prompts, non_special_ingredients)

                for (prompt, gemini_response), ingredient in zip(gemini_results, non_special_ingredients):
                    try:
                        safety_assessment = gemini_response.split("Safety Assessment: ")[1].split("\n")[0]
                        personalized_assessment = gemini_response.split("Personalized Health Assessment: ")[1]
                    except Exception as e:
                        safety_assessment = "Error in Gemini response"
                        personalized_assessment = "Error in Gemini response"
                    ingredient_safety_assessments.append({
                        "ingredient": ingredient,
                        "safety_assessment": safety_assessment,
                        "personalized_health_assessment": personalized_assessment
                    })
        
        concerns_and_likes = generate_concerns_and_likes(foodsafety_analysis, medical_info, allergies, special_codes_with_details)
        
        messages.append({
            "type": "foodsafety",
            "barcode": result['barcode'],
            "product_name": result['product_details']['product_name'],
            "brand": result['product_details']['brand'],
            "quantity": result['product_details']['quantity'],
            "country": result['product_details']['country'],
            "ingredients_FS": corrected_ingredients,
            "foodsafetyAnalysis": foodsafety_analysis,
            "ingredientSafetyAssessments": ingredient_safety_assessments,
            "specialCodes": special_codes,
            "specialCodesWithDetails": special_codes_with_details,
            "concernsAndLikes": concerns_and_likes
        })

        return jsonify({
            'success': True,
            'barcode': result['barcode'],
            'product_details': result['product_details'],
            'foodsafety_analysis': foodsafety_analysis,
            'ingredient_safety_assessments': ingredient_safety_assessments,
            'special_codes': special_codes,
            'special_codes_with_details': special_codes_with_details,
            'special_code_logs': get_special_code_logs(),
            'concerns_and_likes': concerns_and_likes
        }), 200
    except Exception as e:
        error_msg = f"Error in /processBarcode: {str(e)}"
        print(error_msg)
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        print("\nUser Message:", user_message)

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM user_data ORDER BY id DESC LIMIT 1")
        user_data = c.fetchone()
        conn.close()

        if user_data:
            personal_details = user_data[1]
            maintenance_calories = user_data[2]
            protein_requirement = user_data[3]
            allergies = user_data[4]
            medical_info = user_data[5]
        else:
            personal_details = "Not provided"
            maintenance_calories = 0
            protein_requirement = 0
            allergies = "None reported"
            medical_info = "None reported"

        context = {
            "personal_details": personal_details,
            "maintenance_calories": maintenance_calories,
            "protein_requirement": protein_requirement,
            "allergies": allergies,
            "medical_info": medical_info,
            "important_text": important_text_extracted_from_camera,
            "predefined_allergies": predefined_allergies,
            "predefined_conditions": predefined_medical_conditions
        }

        if re.match(r'^(hi|hello|hey|howdy|hola)$', user_message.lower()):
            prompt = f"""You are a friendly food safety assistant. Respond warmly to this greeting:
            User said: "{user_message}"
            
            {format_context(context)}
            
            Response guidelines:
            - Use a welcoming tone
            - Ask how you can help with food safety today
            - Keep it conversational (no markdown)
            """
        elif "ingredient" in user_message.lower() or "safe" in user_message.lower() or "allergy" in user_message.lower():
            prompt = f"""Analyze this food safety query:
            User asked: "{user_message}"
            
            {format_context(context)}
            
            Response format:
            - Start with 'Yes' or 'No' for safety questions
            - Provide concise explanation (max 200 words)
            - Include dietary recommendations
            - Mention health risks
            - Use bullet points for complex answers
            """
        else:
            prompt = f"""Respond to this general query:
            User asked: "{user_message}"
            
            {format_context(context)}
            
            Response guidelines:
            - Be helpful and conversational
            - If unsure, ask for clarification
            - Keep it friendly and informative
            """
        response = make_gemini_call(prompt)
        assistant_reply = response.content
        print("\nAssistant Reply:", assistant_reply)

        messages.append({
            "role": "user",
            "content": user_message
        })
        messages.append({
            "role": "assistant",
            "content": assistant_reply
        })

        return jsonify({'success': True, 'reply': assistant_reply}), 200
    except Exception as e:
        error_msg = f"Error in /chat: {str(e)}"
        print(error_msg)
        return jsonify({'success': False, 'error': error_msg}), 500

def format_context(context):
    return "\n".join([
        f"Personal Details: {context['personal_details']}",
        f"Maintenance Calories: {context['maintenance_calories']} kcal",
        f"Protein Requirement: {context['protein_requirement']} g",
        f"Allergies: {context['allergies']}",
        f"Medical Conditions: {context['medical_info']}",
        f"Important Text: {context['important_text'] or 'No image text extracted'}",
        f"System Knowledge: Allergies database, FSSAI guidelines, WHO standards"
    ])

@app.route('/getSessionData', methods=['GET'])
def get_session_data():
    global form_data
    return jsonify({
        'success': True,
        'weight': form_data['weight'],
        'age': form_data['age'],
        'height': form_data['height'],
        'gender': form_data['gender'],
        'activity': form_data['activity'],
        'goal': form_data['goal'],
        'allergyInput': form_data['allergyInput'],
        'medicalInfoInput': form_data['medicalInfoInput'],
        'currentStep': form_data['currentStep']
    })
@app.route('/clearSession', methods=['POST'])
def clear_session():
    """Clear all session data and reset the database to initial state."""
    try:
        # Clear Flask session data
        session.clear()

        # Reset global variables (if necessary)
        global personalDetails, extractedText1, extractedText2, form_data
        personalDetails = ""
        extractedText1 = ""
        extractedText2 = ""
        form_data = {
            "weight": "",
            "age": "",
            "height": "",
            "gender": "",
            "activity": "",
            "goal": "",
            "allergyInput": "",
            "medicalInfoInput": "",
            "currentStep": 1
        }

        # Clear the database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Drop and recreate tables to reset the database
        c.execute("DROP TABLE IF EXISTS user_data")
        c.execute("DROP TABLE IF EXISTS food_logs")
        conn.commit()
        conn.close()

        # Reinitialize the database
        init_db()

        return jsonify({"success": True, "message": "Session and database cleared successfully"}), 200
    except Exception as e:
        print(f"Error in /clearSession: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/user/details', methods=['GET'])
def get_user_details():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM user_data ORDER BY id DESC LIMIT 1")
        user_data = c.fetchone()
        conn.close()

        if not user_data:
            return jsonify({'success': False, 'error': 'No user data found'}), 404

        return jsonify({
            'success': True,
            'personal_details': user_data[1],
            'maintenance_calories': user_data[2],
            'protein_requirement': user_data[3],
            'allergies': user_data[4],
            'medical_info': user_data[5]
        }), 200
    except Exception as e:
        print(f"Error in /user/details: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def search_food(query, page=1, per_page=5):
    params = {'value': query, 'page': page, 'per_page': per_page}
    try:
        response = requests.get(APYFLUX_URL, headers=APYFLUX_HEADERS, params=params, timeout=10)
        response.raise_for_status()
        return process_apyflux_response(response.json(), query, page, per_page)
    except requests.RequestException as e:
        return {"error": f"Apyflux API call failed: {str(e)}"}

def process_apyflux_response(data, query, page, per_page):
    if not data.get('items'):
        return {"error": "No food items found"}

    filtered_items = []
    for item in data['items']:
        score = max(
            fuzz.partial_ratio(query.lower(), item['food_name'].lower()),
            fuzz.partial_ratio(query.lower(), item['common_names'].lower())
        )
        if score > 85 or query.lower() in item['food_name'].lower():
            serving_size = item['calories_calculated_for']
            nutrients = item['nutrients']
            normalized_nutrients = {
                "calories": round((nutrients['calories'] / serving_size) * 100, 2),
                "fats": round((nutrients['fats'] / serving_size) * 100, 2),
                "carbs": round((nutrients['carbs'] / serving_size) * 100, 2),
                "protein": round((nutrients['protein'] / serving_size) * 100, 2)
            }
            filtered_items.append({
                "food_id": item['food_unique_id'],
                "food_name": item['food_name'],
                "serving_size": serving_size,
                "serving_type": item['serving_type'],
                "nutrients": nutrients,
                "normalized_per_100g": normalized_nutrients
            })

    filtered_items.sort(
        key=lambda x: (
            query.lower() in x['food_name'].lower(),
            max(
                fuzz.partial_ratio(query.lower(), x['food_name'].lower()),
                fuzz.partial_ratio(query.lower(), x['food_name'].lower())
            )
        ),
        reverse=True
    )

    total_results = len(filtered_items)
    total_pages = (total_results + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_items = filtered_items[start:end]

    return {
        "results": paginated_items,
        "total_results": total_results,
        "total_pages": total_pages,
        "current_page": page
    }
# Add this helper function to compute a score based on food safety analysis
def compute_food_score(foodsafety_analysis, concerns_and_likes, allergies, medical_info, special_codes_with_details):
    score = 100  # Start with a perfect score

    # Helper to extract numeric values from strings like "15g" or "300mg"
    def extract_numeric(value):
        if isinstance(value, str) and value != "N/A":
            numeric = re.sub(r'[^\d.]', '', value)
            try:
                return float(numeric)
            except ValueError:
                return 0
        return 0

    # Deduct points for high levels of unhealthy nutrients
    thresholds = {
        "total_sugars": 20,  # g
        "total_fat": 20,     # g
        "saturated_fat": 5,  # g
        "trans_fat": 0,      # g
        "cholesterol": 30,   # mg
        "sodium": 400        # mg
    }

    for nutrient, threshold in thresholds.items():
        value = extract_numeric(foodsafety_analysis.get(nutrient, "N/A"))
        if value > threshold:
            if threshold == 0:  # Special case for trans_fat
                # If trans_fat is present at all, apply a flat deduction
                deduction = 15  # Flat deduction since any trans_fat is bad
            else:
                # Calculate excess as a ratio and determine deduction
                excess = (value - threshold) / threshold
                deduction = min(15, 5 + (excess * 10))
            score -= deduction

    # Deduct points for processing level
    processing_level = foodsafety_analysis.get("processing_level", "N/A")
    if processing_level == "Highly Processed":
        score -= 10
    elif processing_level == "Minimally Processed":
        score -= 5

    # Deduct points for hydrogenated oil
    if foodsafety_analysis.get("hydrogenated_oil", "N/A") == "Present":
        score -= 10

    # Deduct points for additives with negative effects
    for additive in foodsafety_analysis.get("additives", []):
        description = additive.get("description", "").lower()
        if "may cause" in description or "banned" in description:
            score -= 5

    # Deduct points for allergens
    if allergies and allergies != "None reported":
        ingredients = foodsafety_analysis.get("ingredients", "N/A").lower()
        allergy_list = []
        for line in allergies.split("\n"):
            if line.startswith("- Allergen:"):
                allergen = line.split("Allergen:")[1].split("\n")[0].strip().lower()
                allergy_list.append(allergen)
        for allergen in allergy_list:
            if allergen in ingredients:
                score -= 20  # Significant deduction for allergen presence

    # Deduct points for medical condition conflicts
    if medical_info and medical_info != "None reported":
        medical_list = []
        for line in medical_info.split("\n"):
            if line.startswith("- Condition:"):
                condition = line.split("Condition:")[1].split("\n")[0].strip().lower()
                medical_list.append(condition)
        for condition in medical_list:
            if condition == "diabetes":
                sugar_value = extract_numeric(foodsafety_analysis.get("total_sugars", "0g"))
                if sugar_value > thresholds["total_sugars"]:
                    score -= 15
            if condition == "hypertension":
                sodium_value = extract_numeric(foodsafety_analysis.get("sodium", "0mg"))
                if sodium_value > thresholds["sodium"]:
                    score -= 15

    # Deduct points for special codes with negative assessments
    for code in special_codes_with_details:
        assessment = code["personalized_health_assessment"].lower()
        if "avoid" in assessment or "unsafe" in assessment or "allergic" in assessment:
            score -= 10

    # Ensure score doesn't go below 0
    return max(0, round(score))
@app.route('/food/search', methods=['GET'])
def food_search_endpoint():
    try:
        query = request.args.get('query', '')
        page = int(request.args.get('page', 1))
        per_page = 5
        model_type = request.args.get('model_type', 'pro')  # Add model_type parameter
        if not query:
            return jsonify({'success': False, 'error': 'Query parameter required'}), 400

        # Step 1: Fetch food items from Apyflux API
        result = search_food(query, page, per_page)
        if "error" in result:
            return jsonify({'success': False, 'error': result["error"]}), 404

        # Step 2: Fetch user data for personalized assessments
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT allergies, medical_info FROM user_data ORDER BY id DESC LIMIT 1")
        user_data = c.fetchone()
        conn.close()

        allergies = user_data[0] if user_data else "None reported"
        medical_info = user_data[1] if user_data else "None reported"
        print("\nAllergies for Food Search:", allergies)
        print("Medical Info for Food Search:", medical_info)

        # Step 3: Enhance each food item with ingredients and safety analysis
        enhanced_results = []
        for item in result['results']:
            food_name = item['food_name']
            serving_size = item['serving_size']
            
            # Fetch ingredients using Gemini
            prompt = f"""
            You are a food safety and nutrition analysis expert. Provide the ingredients list for the following food item in a comma-separated format (e.g., "sugar, salt, water, E100"). If ingredients are unavailable, return "N/A".

            Food name: {food_name}
            Serving size: {serving_size}

            Return ONLY the ingredients list, no additional text.
            """
            response = make_gemini_call(prompt)
            ingredients = response.content.strip()
            print(f"Ingredients for {food_name}: {ingredients}")

            # If ingredients are not available, skip safety analysis
            if ingredients == "N/A":
                enhanced_results.append({
                    "food_id": item['food_id'],
                    "food_name": item['food_name'],
                    "serving_size": item['serving_size'],
                    "serving_type": item['serving_type'],
                    "nutrients": item['nutrients'],
                    "normalized_per_100g": item['normalized_per_100g'],
                    "ingredients": "N/A",
                    "foodsafety_analysis": None,
                    "ingredient_safety_assessments": [],
                    "special_codes": [],
                    "special_codes_with_details": [],
                    "concerns_and_likes": None
                })
                continue

            # Correct ingredients using Gemini (same as in other routes)
            corrected_ingredients = correct_ingredients_with_gemini(ingredients)

            # Build foodsafety_analysis object (similar to /processBarcode)
            foodsafety_analysis = {
                "processing_level": "N/A",
                "hydrogenated_oil": "Present" if "hydrogenated" in corrected_ingredients.lower() else "Not Present",
                "additives": [],
                "energy": f"{item['normalized_per_100g']['calories']} kcal",
                "total_sugars": f"{item['normalized_per_100g']['carbs']} g",  # Approximate sugars as carbs (Apyflux limitation)
                "total_fat": f"{item['normalized_per_100g']['fats']} g",
                "saturated_fat": "N/A",  # Apyflux doesn't provide this
                "trans_fat": "N/A",
                "cholesterol": "N/A",
                "sodium": "N/A",
                "nutrients": [
                    {"name": "Carbohydrates", "amount": f"{item['normalized_per_100g']['carbs']} g"},
                    {"name": "Protein", "amount": f"{item['normalized_per_100g']['protein']} g"}
                ],
                "ingredients": corrected_ingredients
            }

            # Parse ingredients for safety analysis
            ingredient_list = parse_ingredients(corrected_ingredients)
            print(f"Parsed Ingredients for {food_name}: {ingredient_list}")

            food_code_pattern = re.compile(r'^(E\s?\d{3}[a-zA-Z]?|INS\s?\d{3}[a-zA-Z]?|\d{3})$', re.IGNORECASE)
            special_codes = []
            special_codes_with_details = []
            ingredient_safety_assessments = []

            # Identify special codes
            for ingredient in ingredient_list:
                cleaned_ingredient = re.sub(r'\s*\([^)]+\)', '', ingredient).strip()
                if food_code_pattern.match(cleaned_ingredient):
                    normalized_code = cleaned_ingredient.replace(" ", "")
                    special_codes.append(normalized_code)

            if special_codes:
                # Parallel Gemini calls for RAG safety queries
                safety_queries = [f"Is {code} safe for consumption:" for code in special_codes]
                safety_results = parallel_gemini_calls(safety_queries, special_codes)

                # Parallel Gemini calls for personalized assessments
                personalized_prompts = [
                    f"""
                    Provide a personalized health assessment for the food additive '{code}' based on the user's medical and allergy data.
                    User's Medical Conditions: {medical_info if medical_info else 'None'}
                    User's Allergies: {allergies if allergies else 'None'}
                    Format the response as:
                    Personalized Health Assessment: [assessment]
                    Return ONLY the formatted response, no additional text.
                    """
                    for code in special_codes
                ]
                personalized_results = parallel_gemini_calls(personalized_prompts, special_codes)

                # Process results
                for (safety_query, rag_safety_info), (pers_prompt, pers_result), code in zip(safety_results, personalized_results, special_codes):
                    try:
                        log_special_code(code, rag_safety_info)
                        personalized_assessment = pers_result.split("Personalized Health Assessment: ")[1]
                    except Exception as e:
                        print(f"Error querying RAG system for {code}: {str(e)}")
                        rag_safety_info = "Unable to retrieve safety information."
                        log_special_code(code, rag_safety_info)
                        personalized_assessment = "Unable to assess due to lack of information."

                    special_codes_with_details.append({
                        "additive": code,
                        "personalized_health_assessment": personalized_assessment
                    })
                    ingredient_safety_assessments.append({
                        "ingredient": code,
                        "safety_assessment": f"Food Code Safety (RAG): {rag_safety_info}",
                        "personalized_health_assessment": personalized_assessment
                    })

            # Parallelize assessments for non-special-code ingredients
            non_special_ingredients = [ingredient for ingredient in ingredient_list if not food_code_pattern.match(re.sub(r'\s*\([^)]+\)', '', ingredient).strip())]
            if non_special_ingredients:
                if model_type == "pro":
                    # Prepare Hugging Face payloads
                    hf_payloads = [
                        {
                            "inputs": f"[INST] Give safety assessment about the food ingredient '{ingredient}'. Highlight if it is harmful for any group of people or banned in any country due to its toxic nature. Provide a personalized health assessment based on the user's medical and allergy data, if it's generally safe then prompt it's safe and no concerns. [/INST]",
                            "medical_data": medical_info if medical_info else "None",
                            "allergy_data": allergies if allergies else "None",
                            "parameters": {}
                        }
                        for ingredient in non_special_ingredients
                    ]
                    # Parallel Hugging Face calls
                    hf_results = parallel_hf_calls(hf_payloads, non_special_ingredients)

                    # Process Hugging Face results, fall back to Gemini if needed
                    for (payload, safety_assessment, personalized_assessment), ingredient in zip(hf_results, non_special_ingredients):
                        if "Error" in safety_assessment:
                            print(f"Falling back to Gemini for {ingredient}: {safety_assessment}")
                            gemini_prompt = f"""
                            Give safety assessment about the food ingredient '{ingredient}'. Highlight if it is harmful for any group of people or banned in any country due to its toxic nature. Provide a personalized health assessment based on the user's medical and allergy data, if it's generally safe then prompt it's safe and no concerns.
                            User's Medical Conditions: {medical_info if medical_info else 'None'}
                            User's Allergies: {allergies if allergies else 'None'}
                            Format the response as:
                            Safety Assessment: [assessment]
                            Personalized Health Assessment: [assessment]
                            Return ONLY the formatted response, no additional text.
                            """
                            gemini_results = parallel_gemini_calls([gemini_prompt], [ingredient])
                            gemini_response = gemini_results[0][1]
                            try:
                                safety_assessment = gemini_response.split("Safety Assessment: ")[1].split("\n")[0]
                                personalized_assessment = gemini_response.split("Personalized Health Assessment: ")[1]
                            except Exception as e:
                                safety_assessment = "Error in Gemini fallback"
                                personalized_assessment = "Error in Gemini fallback"
                        ingredient_safety_assessments.append({
                            "ingredient": ingredient,
                            "safety_assessment": safety_assessment,
                            "personalized_health_assessment": personalized_assessment
                        })
                else:
                    # Parallel Gemini calls for non-special ingredients
                    gemini_prompts = [
                        f"""
                        Give safety assessment about the food ingredient '{ingredient}'. Highlight if it is harmful for any group of people or banned in any country due to its toxic nature. Provide a personalized health assessment based on the user's medical and allergy data, if it's generally safe then prompt it's safe and no concerns.
                        User's Medical Conditions: {medical_info if medical_info else 'None'}
                        User's Allergies: {allergies if allergies else 'None'}
                        Format the response as:
                        Safety Assessment: [assessment]
                        Personalized Health Assessment: [assessment]
                        Return ONLY the formatted response, no additional text.
                        """
                        for ingredient in non_special_ingredients
                    ]
                    gemini_results = parallel_gemini_calls(gemini_prompts, non_special_ingredients)

                    for (prompt, gemini_response), ingredient in zip(gemini_results, non_special_ingredients):
                        try:
                            safety_assessment = gemini_response.split("Safety Assessment: ")[1].split("\n")[0]
                            personalized_assessment = gemini_response.split("Personalized Health Assessment: ")[1]
                        except Exception as e:
                            safety_assessment = "Error in Gemini response"
                            personalized_assessment = "Error in Gemini response"
                        ingredient_safety_assessments.append({
                            "ingredient": ingredient,
                            "safety_assessment": safety_assessment,
                            "personalized_health_assessment": personalized_assessment
                        })

            # Generate concerns and likes
            concerns_and_likes = generate_concerns_and_likes(foodsafety_analysis, medical_info, allergies, special_codes_with_details)

            # Add to enhanced results
            enhanced_results.append({
                "food_id": item['food_id'],
                "food_name": item['food_name'],
                "serving_size": item['serving_size'],
                "serving_type": item['serving_type'],
                "nutrients": item['nutrients'],
                "normalized_per_100g": item['normalized_per_100g'],
                "ingredients": corrected_ingredients,
                "foodsafety_analysis": foodsafety_analysis,
                "ingredient_safety_assessments": ingredient_safety_assessments,
                "special_codes": special_codes,
                "special_codes_with_details": special_codes_with_details,
                "concerns_and_likes": concerns_and_likes
            })

        # Update result['results'] with enhanced data
        result['results'] = enhanced_results

        return jsonify({
            'success': True,
            'data': {
                'results': result['results'],
                'total_pages': result['total_pages'],
                'current_page': result['current_page']
            }
        }), 200
    except Exception as e:
        print(f"Error in /food/search: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/food/track', methods=['POST'])
def track_food():
    try:
        data = request.json
        user_id = 1
        food_id = data.get('food_id')
        food_name = data.get('food_name')
        serving_size = data.get('serving_size')
        nutrients = data.get('nutrients')
        food_serving_size = data.get('food_serving_size')
        timestamp = data.get('timestamp')

        if not all([food_id, food_name, serving_size, nutrients, food_serving_size]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        scale = serving_size / food_serving_size
        scaled_nutrients = {
            "calories": round(nutrients['calories'] * scale, 2),
            "fats": round(nutrients['fats'] * scale, 2),
            "carbs": round(nutrients['carbs'] * scale, 2),
            "protein": round(nutrients['protein'] * scale, 2)
        }

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO food_logs (user_id, food_id, food_name, serving_size, nutrients, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (user_id, food_id, food_name, serving_size, json.dumps(scaled_nutrients), timestamp))
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'Food logged successfully', 'nutrients': scaled_nutrients}), 200
    except Exception as e:
        print(f"Error in /food/track: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/food/logs', methods=['GET'])
def get_food_logs():
    try:
        user_id = 1
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, food_name, serving_size, nutrients, timestamp FROM food_logs WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
        logs = c.fetchall()
        conn.close()

        formatted_logs = []
        for log in logs:
            formatted_logs.append({
                'id': log[0],
                'food_name': log[1],
                'serving_size': log[2],
                'nutrients': json.loads(log[3]),
                'timestamp': log[4]
            })

        return jsonify({'success': True, 'logs': formatted_logs}), 200
    except Exception as e:
        print(f"Error in /food/logs: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/food/clear', methods=['POST'])
def clear_food_logs():
    try:
        user_id = 1
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM food_logs WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'Food logs cleared successfully'}), 200
    except Exception as e:
        print(f"Error in /food/clear: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/input', methods=['POST'])
def get_input():
    return redirect(url_for('askme'))

@app.route('/askme')
def askme():
    return render_template('askme.html')

@app.route('/foodsafety')
def foodsafety_page():
    return render_template('foodsafety.html')

@app.route('/updatePersonalDetails', methods=['POST'])
def update_personal_details():
    global personalDetails
    try:
        data = request.json
        personalDetails = data.get('personalDetails', '')
        print("\nUpdated Personal Details:", personalDetails)
        return jsonify({'success': True}), 200
    except Exception as e:
        print("Error:", e)
        return jsonify({'success': False, 'error': str(e)}), 500
# Add the /geminiScore route to handle the scoring request
@app.route('/geminiScore', methods=['POST'])
def gemini_score():
    try:
        data = request.json
        foodsafety_analysis = data.get('foodsafetyAnalysis')
        concerns_and_likes = data.get('concernsAndLikes')
        allergies = data.get('allergies')
        medical_info = data.get('medicalInfo')
        special_codes_with_details = data.get('specialCodesWithDetails', [])

        if not foodsafety_analysis or not concerns_and_likes:
            return jsonify({'success': False, 'error': 'Missing required analysis data'}), 400

        # Compute the score
        score = compute_food_score(foodsafety_analysis, concerns_and_likes, allergies, medical_info, special_codes_with_details)

        return jsonify({
            'success': True,
            'score': score
        }), 200
    except Exception as e:
        print(f"Error in /geminiScore: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)