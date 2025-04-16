# aadish.py

import requests
import logging
import sys
import time
import os

# Configure logging - only essential messages
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s:%(message)s')

# Use localhost for development
API_URL = 'https://aiaadish.vercel.app/api/chat'
#API_URL = 'http://localhost:3000/api/chat'
# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are an advanced and highly capable AI assistant named Aadish, created by Aadish; you are designed to interact with users in a clear, friendly, and professional manner while always adjusting your responses based on the complexity of their problem—whether that means being concise or providing detailed, step-by-step explanations when necessary—and you must ensure that your answers are accurate, coherent, and ethically sound, reflecting your identity and upholding the principles set by your creator. Use appropriate emojis 😊"""

# Model definitions
AADISH_MODELS = [
    {
        "name": "llama4",
        "id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "description": "Llama4: General AI model. Use for versatile tasks.",
        "usage": 'model="meta-llama/llama-4-scout-17b-16e-instruct"'
    },
    {
        "name": "llama3.3",
        "id": "llama-3.3-70b-versatile",
        "description": "Llama3.3: General AI model. Use for versatile tasks.",
        "usage": 'model="llama-3.3-70b-versatile"'
    },
    {
        "name": "mistral",
        "id": "mistral-saba-24b",
        "description": "Mistral: Advanced language model with strong performance.",
        "usage": 'model="mistral-saba-24b"'
    },
    {
        "name": "compund",
        "id": "compound-beta",
        "description": "Compund: Has search ability and code execution.",
        "usage": 'model="compound-beta"'
    },
    {
        "name": "compundmini",
        "id": "compound-beta-mini",
        "description": "CompundMini: Lightweight version of Compund.",
        "usage": 'model="compound-beta-mini"'
    },
    {
        "name": "gemma",
        "id": "gemma2-9b-it",
        "description": "Gemma: General AI model.",
        "usage": 'model="gemma2-9b-it"'
    },
    {
        "name": "deepseek",
        "id": "deepseek-r1-distill-llama-70b",
        "description": "Deepseek: Has thinking and reasoning abilities.",
        "usage": 'model="deepseek-r1-distill-llama-70b"'
    },
    {
        "name": "qwen",
        "id": "qwen-qwq-32b",
        "description": "Qwen: Has thinking and reasoning abilities.",
        "usage": 'model="qwen-qwq-32b"'
    },
]

AADISH_COMMANDS = [
    "aadish(message, model='model_id', system=None)",
    "aadishresponse()",
    "aadishtalk(model='model_id', system=None)",
    "aadishcommands()",
    "aadishmodels()"
]

# Configurable parameters with default values
class AadishConfig:
    def __init__(self):
        self._temp = 1.0
        self._topp = 1.0
        self._tokens = 1024

    @property
    def temp(self):
        return self._temp

    @temp.setter
    def temp(self, value):
        if not isinstance(value, (int, float)) or value < 0 or value > 2:
            raise ValueError("Temperature must be a number between 0 and 2")
        self._temp = float(value)

    @property
    def topp(self):
        return self._topp

    @topp.setter
    def topp(self, value):
        if not isinstance(value, (int, float)) or value <= 0 or value > 1:
            raise ValueError("Top P must be a number between 0 and 1")
        self._topp = float(value)

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        if not isinstance(value, int) or value < 1:
            raise ValueError("Max tokens must be a positive integer")
        self._tokens = value

# Create a global config instance
config = AadishConfig()

# Export config attributes at module level for easy access
temp = config.temp
topp = config.topp
tokens = config.tokens

class AadishConnectionError(Exception):
    pass

class AadishServerError(Exception):
    pass

_last_response = None

def handle_api_error(response):
    """Handle API errors with specific error messages"""
    try:
        error_data = response.json()
        if 'error' in error_data:
            if isinstance(error_data['error'], dict):
                return f"API Error: {error_data['error'].get('message', 'Unknown error')}"
            return f"API Error: {error_data['error']}"
    except:
        if response.status_code == 401:
            return "Authentication error: Invalid API key"
        elif response.status_code == 404:
            return "Model not found or not accessible"
        elif response.status_code == 429:
            return "Rate limit exceeded. Please try again later"
        return f"HTTP Error {response.status_code}: {response.text}"

def _validate_model(model: str) -> str:
    """
    Validate and convert model name to ID if needed.
    Returns the valid model ID or raises ValueError.
    """
    # If it's already a valid model ID, return it
    valid_ids = [m["id"] for m in AADISH_MODELS]
    if model in valid_ids:
        return model
        
    # Try to convert name to ID
    for m in AADISH_MODELS:
        if m["name"] == model:
            return m["id"]
            
    raise ValueError(f"Invalid model: {model}. Valid models are: {', '.join([m['name'] for m in AADISH_MODELS])}")

def aadish(message: str, model: str = "compound-beta", system: str = None) -> None:
    global _last_response
    try:
        if not message or not message.strip():
            error_msg = "Message cannot be empty"
            print(f"Error: {error_msg}", file=sys.stderr)
            _last_response = error_msg
            return

        # Convert model name to ID if it matches a known model name
        model_id = model
        for m in AADISH_MODELS:
            if m['name'] == model:
                model_id = m['id']
                break

        payload = {
            "message": message.strip(),
            "model": model_id,
            "system": system or DEFAULT_SYSTEM_PROMPT,
            "temperature": config.temp,
            "top_p": config.topp,
            "max_completion_tokens": config.tokens
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        with requests.post(
            API_URL,
            json=payload,
            headers=headers,
            stream=True,
            timeout=60
        ) as r:
            if not r.ok:
                error_msg = handle_api_error(r)
                print(f"Error: {error_msg}", file=sys.stderr)
                _last_response = error_msg
                return

            content = ""
            for chunk in r.iter_content(decode_unicode=True):
                if chunk:
                    try:
                        # Ensure proper encoding for stdout
                        if sys.stdout.encoding.lower() != 'utf-8':
                            chunk = chunk.encode('utf-8').decode(sys.stdout.encoding, errors='replace')
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                        content += chunk
                    except Exception as e:
                        print(f"\nError processing chunk: {e}", file=sys.stderr)
            
            print()  # Add final newline
            _last_response = content.strip()

    except requests.exceptions.Timeout:
        error_msg = "Connection timeout. Please check your internet connection and try again."
        print(f"Error: {error_msg}", file=sys.stderr)
        _last_response = error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "Connection error. Please check your internet connection and the API URL."
        print(f"Error: {error_msg}", file=sys.stderr)
        _last_response = error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"Error: {error_msg}", file=sys.stderr)
        _last_response = error_msg

def aadishresponse() -> str:
    """
    Returns the last response received from the server.
    """
    return _last_response

def aadishtalk(model: str = "compound-beta", system: str = None):
    """
    Starts an interactive loop with customizable system prompt.
    
    Args:
        model: The model to use (name or ID)
        system: Optional system prompt to override the default
    """
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    print("Made by Aadish with the help of AI. Yes, you can search real-time results! (Press Ctrl+C to exit)")
    
    model_id = model
    for m in AADISH_MODELS:
        if m['name'] == model:
            model_id = m['id']
            break
            
    history = []
    try:
        while True:
            user_input = input(f"{BLUE}You:{RESET} ")
            if user_input.strip() == "":
                continue
            history.append({"role": "user", "content": user_input})
            
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "message": user_input,
                "model": model_id,
                "history": history[:-1],
                "system": system or DEFAULT_SYSTEM_PROMPT,
                "temperature": config.temp,
                "top_p": config.topp,
                "max_completion_tokens": config.tokens
            }
            
            try:
                sys.stdout.write(f"{MAGENTA}Aadish:{RESET} ")
                sys.stdout.flush()
                
                response_started = False
                with requests.post(
                    API_URL,
                    json=payload,
                    headers=headers,
                    stream=True,
                    timeout=60
                ) as r:
                    if not r.ok:
                        error_msg = handle_api_error(r)
                        print(f"Error: {error_msg}", file=sys.stderr)
                        continue

                    content = ""
                    for chunk in r.iter_content(decode_unicode=True):
                        if chunk:
                            try:
                                if not response_started:
                                    response_started = True
                                # Ensure proper encoding for stdout
                                if sys.stdout.encoding.lower() != 'utf-8':
                                    chunk = chunk.encode('utf-8').decode(sys.stdout.encoding, errors='replace')
                                sys.stdout.write(chunk)
                                sys.stdout.flush()
                                content += chunk
                            except Exception as e:
                                print(f"\nError processing chunk: {e}", file=sys.stderr)
                    
                    if not content.strip():
                        print("\nNo response received", file=sys.stderr)
                    
                    print()
                    history.append({"role": "assistant", "content": content.strip()})
                    
            except requests.exceptions.Timeout:
                print("Error: Connection timeout. Please check your internet connection.", file=sys.stderr)
            except requests.exceptions.ConnectionError:
                print("Error: Connection error. Please check your internet connection.", file=sys.stderr)
            except Exception as e:
                print(f"Error: {str(e)}", file=sys.stderr)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting chat.")

def aadishcommands():
    """
    Prints all available commands for the aadish module.
    """
    print("Available commands:")
    for cmd in AADISH_COMMANDS:
        print(f"  - {cmd}")

def aadishmodels():
    """
    Prints all available AI models with details.
    """
    print("Available AI models:")
    for m in AADISH_MODELS:
        print(f"\nName: {m['name']}\nID: {m['id']}\nDescription: {m['description']}\nUsage: {m['usage']}")
