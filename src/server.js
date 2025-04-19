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

        // Set headers for streaming response with proper encoding
        res.writeHead(200, {
            'Content-Type': 'text/plain; charset=utf-8',
            'Transfer-Encoding': 'chunked',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache'
        });

        // Configure LangChain model with parameters
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
                        // Ensure UTF-8 encoding for emojis
                        const encodedToken = Buffer.from(token, 'utf8').toString('utf8');
                        res.write(encodedToken);
                    }
                },
                handleLLMEnd() {
                    res.end();
                },
                handleLLMError(error) {
                    console.error('LLM Error:', error);
                    if (!res.writableEnded) {
                        res.write('Error: ' + (error.message || 'Unknown error'));
                        res.end();
                    }
                }
            }]
        });

        if (history && Array.isArray(history)) {
            const messages = [
                new SystemMessage(system || "You are a helpful AI assistant."),
                ...(history.map(msg => {
                    if (msg.role === 'user') return new HumanMessage(msg.content);
                    if (msg.role === 'assistant') return new AIMessage(msg.content);
                    return null;
                }).filter(Boolean)),
                new HumanMessage(message)
            ];
            
            await llm.invoke(messages);
        } else {
            await llm.invoke([
                new SystemMessage(system || "You are a helpful AI assistant."),
                new HumanMessage(message)
            ]);
        }

    } catch (error) {
        console.error('Server Error:', error);
        if (!res.writableEnded) {
            const errorMessage = error.message || 'Unknown server error';
            res.write('\nServer Error: ' + errorMessage);
            res.end();
        }
    }
});

module.exports = app;
