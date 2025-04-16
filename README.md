# Aadish Chat API

This project provides an AI chat API using Node.js (Express) and a Python client for easy interaction with multiple AI models.

## Features
- Streamed AI chat responses
- Supports multiple advanced models
- In-memory conversation history (Python client)
- Simple Python client for scripting and interactive use

## Setup

### 1. Clone the repository
```sh
git clone <your-repo-url>
cd <project-folder>
```

### 2. Install dependencies
```sh
cd vercel
npm install
```

### 3. Configure environment variables
Create a `.env` file in the `vercel` directory (see template below):
```
API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
PROD_URL=https://your-production-url.com
```

### 4. Run the server (development)
```sh
npm run dev
```
The server will start on `http://localhost:3000` by default.

---

## Python Client Usage

The Python client (`aadish.py`) provides several functions for interacting with the API.

### Functions & Examples

#### 1. `aadish(message: str, model: str = "compound-beta", custom_system_prompt: str = None)`
Send a message to the Aadish AI server and print the streamed reply. Optionally specify a model and custom system prompt.
```python
from aadish import aadish
# Basic usage
aadish("Hello, how are you?", model="llama4")

# With custom system prompt
aadish("Hello", model="llama4", custom_system_prompt="You are a helpful assistant that speaks like Shakespeare.")
```

#### 2. `aadishresponse()`
Get the last response received from the server.
```python
from aadish import aadish, aadishresponse
aadish("Tell me a joke.")
print(aadishresponse())
```

#### 3. `aadishtalk(model: str = "compound-beta", custom_system_prompt: str = None)`
Start an interactive chat loop with the AI. Optionally specify a model and custom system prompt.
```python
from aadish import aadishtalk
# Basic usage
aadishtalk(model="llama3.3")

# With custom system prompt
aadishtalk(model="llama3.3", custom_system_prompt="You are a Python coding expert that provides concise answers.")
```

#### 4. `aadishcommands()`
Print all available commands for the Aadish module.
```python
from aadish import aadishcommands
aadishcommands()
```

#### 5. `aadishmodels()`
Print all available AI models with details.
```python
from aadish import aadishmodels
aadishmodels()
```

---

## Available Models

| Name        | ID                                   | Description                                 | Usage Example                                 |
|-------------|--------------------------------------|---------------------------------------------|-----------------------------------------------|
| llama4      | meta-llama/llama-4-scout-17b-16e-instruct | General AI model. Versatile tasks.          | model="meta-llama/llama-4-scout-17b-16e-instruct" |
| llama3.3    | llama-3.3-70b-versatile              | General AI model. Versatile tasks.          | model="llama-3.3-70b-versatile"              |
| compund     | compound-beta                        | Has search ability and code execution.      | model="compound-beta"                        |
| compundmini | compound-beta-mini                   | Lightweight version of Compund.             | model="compound-beta-mini"                   |
| gemma       | gemma2-9b-it                         | General AI model.                           | model="gemma2-9b-it"                         |
| deepseek    | deepseek-r1-distill-llama-70b        | Thinking and reasoning abilities.           | model="deepseek-r1-distill-llama-70b"        |
| qwen        | qwen-qwq-32b                         | Thinking and reasoning abilities.           | model="qwen-qwq-32b"                         |

---

## API Endpoints

- `GET /` — Welcome message
- `POST /api/chat` — Chat with the AI (see code for request format)

## Notes
- **Do not commit your API keys.**
- Conversation history is not persisted between sessions.
- For local development, you can point the Python client to `http://localhost:3000/api/chat`.

---
Made by Aadish.
