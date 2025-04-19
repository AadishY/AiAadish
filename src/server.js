const express = require('express');
const dotenv = require('dotenv');
const { ChatGroq } = require('@langchain/groq');
const { HumanMessage, SystemMessage, AIMessage } = require("@langchain/core/messages");
const { BufferMemory, ChatMessageHistory } = require('langchain/memory');
const { ConversationChain } = require('langchain/chains');
const { ChatPromptTemplate, MessagesPlaceholder } = require("@langchain/core/prompts");
const cors = require('cors');

dotenv.config({ path: require('path').resolve(__dirname, '../.env') });

const app = express();
app.use(cors());
app.use(express.json());

const API_KEY = process.env.GROQ_API_KEY || process.env.API_KEY;

if (!API_KEY) {
    throw new Error('GROQ_API_KEY is required. Please check your .env file.');
}

// Initialize logger
const logger = {
    info: (msg) => console.log(`INFO: ${msg}`),
    error: (msg) => console.error(`ERROR: ${msg}`),
    warning: (msg) => console.warn(`WARN: ${msg}`),
    debug: (msg) => console.log(`DEBUG: ${msg}`)
};

// Create a chat prompt template with memory
const chatPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("system_message"),
    new MessagesPlaceholder("history"),
    new HumanMessage("{input}")
]);

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

app.get('/', (req, res) => {
    res.send('Welcome to the Aadish Chat API! Use POST /api/chat to interact.');
});

app.post('/api/chat', async (req, res) => {
    try {
        const { 
            message, 
            model, 
            system, 
            history,
            temperature,
            top_p,
            max_completion_tokens 
        } = req.body;

        if (!message || typeof message !== 'string' || message.trim() === '') {
            return res.status(400).json({ error: 'Message is required and cannot be empty' });
        }

        const modelId = model || "compound-beta";
        if (!VALID_MODELS.includes(modelId)) {
            return res.status(400).json({ error: 'Invalid model ID' });
        }

        // Set headers for streaming response
        res.writeHead(200, {
            'Content-Type': 'text/plain; charset=utf-8',
            'Transfer-Encoding': 'chunked',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        });

        logger.info(`Processing chat request with model: ${modelId}`);
        
        // Configure LangChain model
        const llm = new ChatGroq({
            apiKey: API_KEY,
            model: modelId,
            temperature: typeof temperature === 'number' ? temperature : 1.0,
            maxTokens: typeof max_completion_tokens === 'number' ? max_completion_tokens : 1024,
            topP: typeof top_p === 'number' ? top_p : 1.0,
            streaming: true,
            callbacks: [{
                handleLLMNewToken(token) {
                    if (token) {
                        const encodedToken = Buffer.from(token, 'utf8').toString('utf8');
                        res.write(encodedToken);
                        logger.debug(`Token generated: ${token.slice(0, 30)}...`);
                    }
                },
                handleLLMEnd() {
                    logger.info('Chat completion finished successfully');
                    res.end();
                },
                handleLLMError(error) {
                    logger.error(`LLM Error: ${error.message || 'Unknown error'}`);
                    if (!res.writableEnded) {
                        res.write('Error: ' + (error.message || 'Unknown error'));
                        res.end();
                    }
                }
            }]
        });

        // Initialize chat history if provided
        let chatHistory;
        if (history && Array.isArray(history)) {
            logger.info(`Initializing chat history with ${history.length} previous messages`);
            const previousMessages = history.map(msg => {
                if (!msg?.role || !msg?.content) return null;
                if (msg.role === 'user') return new HumanMessage(msg.content);
                if (msg.role === 'assistant') return new AIMessage(msg.content);
                if (msg.role === 'system') return new SystemMessage(msg.content);
                return null;
            }).filter(Boolean);
            
            chatHistory = new ChatMessageHistory(previousMessages);
        } else {
            logger.debug('Starting new chat history');
            chatHistory = new ChatMessageHistory();
        }

        // Create memory instance
        const memory = new BufferMemory({
            chatHistory,
            returnMessages: true,
            memoryKey: "history",
        });

        // Create conversation chain
        const chain = new ConversationChain({
            llm,
            memory,
            prompt: chatPrompt,
        });

        // Run the chain with the current message
        await chain.call({
            input: message,
            system_message: [new SystemMessage(system || "You are a helpful AI assistant.")]
        });

    } catch (error) {
        logger.error(`Server Error: ${error.message}`);
        console.error('Stack trace:', error);
        if (!res.writableEnded) {
            const errorMessage = error.message || 'Unknown server error';
            res.write('\nServer Error: ' + errorMessage);
            res.end();
        }
    }
});

module.exports = app;
