const express = require('express');
const dotenv = require('dotenv');
const { ChatGroq } = require('@langchain/groq');
const { HumanMessage, SystemMessage, AIMessage } = require("@langchain/core/messages");
const { BufferMemory, ChatMessageHistory } = require('langchain/memory');
const { ConversationChain } = require('langchain/chains');
const { ChatPromptTemplate, MessagesPlaceholder } = require("@langchain/core/prompts");
const cors = require('cors');

// Only load .env in development
if (process.env.NODE_ENV !== 'production') {
    dotenv.config({ path: require('path').resolve(__dirname, '../.env') });
}

const app = express();
app.use(cors());
app.use(express.json());

const API_KEY = process.env.GROQ_API_KEY || process.env.API_KEY;

if (!API_KEY) {
    throw new Error('GROQ_API_KEY is required');
}

// Initialize logger with production-safe logging
const logger = {
    info: (msg) => console.log(`INFO: ${msg}`),
    error: (msg) => console.error(`ERROR: ${msg}`),
    warning: (msg) => console.warn(`WARN: ${msg}`),
    debug: (msg) => process.env.NODE_ENV !== 'production' ? console.log(`DEBUG: ${msg}`) : null
};

// Valid model IDs
const VALID_MODELS = [
    "compound-beta",
    "compound-beta-mini",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b",
    "qwen-qwq-32b"
];

// Health check endpoint
app.get('/', (req, res) => {
    res.status(200).json({ status: 'ok', message: 'Aadish Chat API is running' });
});

app.post('/api/chat', async (req, res) => {
    let responseStarted = false;
    let responseEnded = false;

    // Cleanup function to ensure we don't leave hanging connections
    const cleanup = () => {
        if (!responseEnded) {
            responseEnded = true;
            if (!responseStarted) {
                res.status(500).json({ error: 'Internal server error' });
            } else {
                res.end();
            }
        }
    };

    try {
        const { message, model, system, history, temperature, top_p, max_completion_tokens } = req.body;

        // Validate required parameters
        if (!message || typeof message !== 'string' || message.trim() === '') {
            return res.status(400).json({ error: 'Message is required and cannot be empty' });
        }

        // Validate and sanitize config
        const validatedConfig = {
            message: message.trim(),
            model: VALID_MODELS.includes(model) ? model : "compound-beta",
            system: typeof system === 'string' && system.trim() ? system.trim() : "You are a helpful AI assistant.",
            temperature: Number.isFinite(parseFloat(temperature)) ? Math.max(0, Math.min(2, parseFloat(temperature))) : 1.0,
            topP: Number.isFinite(parseFloat(top_p)) ? Math.max(0, Math.min(1, parseFloat(top_p))) : 1.0,
            maxTokens: Number.isInteger(parseInt(max_completion_tokens)) ? Math.max(1, parseInt(max_completion_tokens)) : 1024
        };

        // Set up streaming response
        res.writeHead(200, {
            'Content-Type': 'text/plain; charset=utf-8',
            'Transfer-Encoding': 'chunked',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        });
        responseStarted = true;

        // Configure chat model
        const llm = new ChatGroq({
            apiKey: API_KEY,
            model: validatedConfig.model,
            temperature: validatedConfig.temperature,
            maxTokens: validatedConfig.maxTokens,
            topP: validatedConfig.topP,
            streaming: true,
            callbacks: [{
                handleLLMNewToken(token) {
                    if (!responseEnded && token) {
                        try {
                            res.write(token);
                        } catch (err) {
                            logger.error(`Failed to write token: ${err.message}`);
                            cleanup();
                        }
                    }
                },
                handleLLMEnd() {
                    cleanup();
                },
                handleLLMError(error) {
                    logger.error(`LLM Error: ${error.message}`);
                    if (!responseEnded) {
                        res.write(`Error: ${error.message || 'Unknown error'}`);
                        cleanup();
                    }
                }
            }]
        });

        // Create prompt template
        const chatPrompt = ChatPromptTemplate.fromMessages([
            new SystemMessage(validatedConfig.system),
            ...(history && Array.isArray(history) ? history.map(msg => {
                if (msg.role === 'user') return new HumanMessage(msg.content);
                if (msg.role === 'assistant') return new AIMessage(msg.content);
                return null;
            }).filter(Boolean) : []),
            new HumanMessage(validatedConfig.message)
        ]);

        // Generate the response
        const response = await chatPrompt.invoke({
            llm,
        });

        if (!responseEnded) {
            cleanup();
        }

    } catch (error) {
        logger.error(`Server Error: ${error.message}`);
        console.error('Stack trace:', error);
        cleanup();
    }
});

module.exports = app;
