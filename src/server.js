const express = require('express');
const dotenv = require('dotenv');
const { ChatGroq } = require('@langchain/groq');
const { HumanMessage, SystemMessage, AIMessage } = require("@langchain/core/messages");
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
    res.json({
        status: 'success',
        message: 'Welcome to the Aadish Chat API!',
        endpoints: {
            chat: {
                method: 'POST',
                path: '/api/chat',
                description: 'Send chat messages to interact with the AI'
            }
        }
    });
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
            max_completion_tokens,
            format = 'text' // New parameter for response format
        } = req.body;

        if (!message || typeof message !== 'string' || message.trim() === '') {
            return res.status(400).json({ 
                status: 'error',
                error: 'Message is required and cannot be empty' 
            });
        }

        const modelId = model || "compound-beta";

        // Set headers based on format
        if (format === 'text') {
            res.writeHead(200, {
                'Content-Type': 'text/plain; charset=utf-8',
                'Transfer-Encoding': 'chunked',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache'
            });
        } else {
            res.writeHead(200, {
                'Content-Type': 'application/json; charset=utf-8',
                'Transfer-Encoding': 'chunked',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache'
            });
        }

        let responseText = '';

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
                        const encodedToken = Buffer.from(token, 'utf8').toString('utf8');
                        if (format === 'text') {
                            res.write(encodedToken);
                        } else {
                            responseText += encodedToken;
                        }
                    }
                },
                handleLLMEnd() {
                    if (format === 'json') {
                        const response = {
                            status: 'success',
                            response: responseText,
                            model: modelId,
                            usage: {
                                input_tokens: message.length / 4, // Approximate
                                output_tokens: responseText.length / 4 // Approximate
                            }
                        };
                        res.write(JSON.stringify(response));
                    }
                    res.end();
                },
                handleLLMError(error) {
                    console.error('LLM Error:', error);
                    if (!res.writableEnded) {
                        const errorResponse = {
                            status: 'error',
                            error: error.message || 'Unknown error',
                            model: modelId
                        };
                        if (format === 'json') {
                            res.write(JSON.stringify(errorResponse));
                        } else {
                            res.write('Error: ' + errorResponse.error);
                        }
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
            const errorResponse = {
                status: 'error',
                error: error.message || 'Unknown server error',
                timestamp: new Date().toISOString()
            };
            
            if (req.body.format === 'json') {
                res.write(JSON.stringify(errorResponse));
            } else {
                res.write('\nServer Error: ' + errorResponse.error);
            }
            res.end();
        }
    }
});

module.exports = app;
