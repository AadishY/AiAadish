const app = require('./server');

// Start the server when running locally
if (process.env.NODE_ENV !== 'production') {
    const PORT = process.env.PORT || 3000;
    let server;

    try {
        server = app.listen(PORT, () => {
            console.log(`Server is running on port ${PORT}`);
        });

        // Handle server-specific errors
        server.on('error', (error) => {
            if (error.code === 'EADDRINUSE') {
                console.error(`Port ${PORT} is already in use`);
            } else {
                console.error('Server error:', error);
            }
            process.exit(1);
        });

        // Graceful shutdown handling
        const shutdown = () => {
            console.log('Received shutdown signal. Closing server...');
            server.close(() => {
                console.log('Server closed');
                process.exit(0);
            });

            // Force close after 10s
            setTimeout(() => {
                console.error('Could not close connections in time, forcefully shutting down');
                process.exit(1);
            }, 10000);
        };

        process.on('SIGTERM', shutdown);
        process.on('SIGINT', shutdown);
    } catch (error) {
        console.error('Failed to start server:', error);
        process.exit(1);
    }
}

// Export the Express app as a serverless function
module.exports = app;
