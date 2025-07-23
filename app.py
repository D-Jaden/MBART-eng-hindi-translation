from flask import Flask, request, jsonify
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables to store model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the MBart model and tokenizer"""
    global model, tokenizer
    
    try:
        logger.info("Loading MBart-50 model and tokenizer...")
        
        # Load model and tokenizer
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def translate_text(text, source_lang="en_XX", target_lang="hi_IN"):
    """Translate text from source language to target language"""
    try:
        # Set source language
        tokenizer.src_lang = source_lang
        
        # Tokenize the input text
        encoded = tokenizer(text, return_tensors="pt")
        
        # Move to same device as model
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Generate translation
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        
        # Decode the generated tokens
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        return translated_text
        
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "English to Hindi Translation API is running"
    })

@app.route('/translate', methods=['POST'])
def translate():
    """Main translation endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing 'text' field in request body"
            }), 400
        
        input_text = data['text'].strip()
        
        if not input_text:
            return jsonify({
                "error": "Text cannot be empty"
            }), 400
        
        # Optional parameters
        source_lang = data.get('source_lang', 'en_XX')  # English
        target_lang = data.get('target_lang', 'hi_IN')  # Hindi
        
        # Translate the text
        translated_text = translate_text(input_text, source_lang, target_lang)
        
        return jsonify({
            "success": True,
            "original_text": input_text,
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang
        })
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages"""
    # MBart-50 supported languages (subset for common ones)
    languages = {
        "en_XX": "English",
        "hi_IN": "Hindi",
        "ar_AR": "Arabic",
        "cs_CZ": "Czech",
        "de_DE": "German",
        "es_XX": "Spanish",
        "et_EE": "Estonian",
        "fi_FI": "Finnish",
        "fr_XX": "French",
        "gu_IN": "Gujarati",
        "he_IL": "Hebrew",
        "it_IT": "Italian",
        "ja_XX": "Japanese",
        "kk_KZ": "Kazakh",
        "ko_KR": "Korean",
        "lt_LT": "Lithuanian",
        "lv_LV": "Latvian",
        "my_MM": "Myanmar",
        "ne_NP": "Nepali",
        "nl_XX": "Dutch",
        "ro_RO": "Romanian",
        "ru_RU": "Russian",
        "si_LK": "Sinhala",
        "tr_TR": "Turkish",
        "vi_VN": "Vietnamese",
        "zh_CN": "Chinese"
    }
    
    return jsonify({
        "supported_languages": languages
    })

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information"""
    return jsonify({
        "message": "English to Hindi Translation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /translate": "Translate text from English to Hindi",
            "GET /languages": "Get supported languages",
            "GET /health": "Health check"
        },
        "model": "facebook/mbart-large-50-many-to-many-mmt",
        "example_usage": {
            "url": "/translate",
            "method": "POST",
            "body": {
                "text": "Hello, how are you?",
                "source_lang": "en_XX",
                "target_lang": "hi_IN"
            }
        }
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=7860, debug=False)
