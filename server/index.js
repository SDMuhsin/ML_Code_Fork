const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const dotenv = require('dotenv');

dotenv.config();

const authRoutes = require('./routes/auth');
const submissionRoutes = require('./routes/submissions');
const problemRoutes = require('./routes/problems');
const executeRoutes = require('./routes/execute');
const discussionRoutes = require('./routes/discussions');
const kataRoutes = require('./kata/routes');

const app = express();
const PORT = process.env.PORT || 5001;
// This server runs code you send it, so by default it only listens on this machine.
const HOST = process.env.HOST || '127.0.0.1';

// Only the local frontend may call the API. An open CORS policy here would let
// any website you visit POST code to the execution endpoints.
const allowedOrigins = (process.env.CORS_ORIGINS || 'http://localhost:8080,http://127.0.0.1:8080')
    .split(',').map(s => s.trim()).filter(Boolean);
app.use(cors({
    origin: (origin, cb) => cb(null, !origin || allowedOrigins.includes(origin)),
}));
// CORS alone does not stop a "simple" cross-site POST from reaching the server,
// so refuse any request that carries a foreign Origin header.
app.use((req, res, next) => {
    const origin = req.headers.origin;
    if (origin && !allowedOrigins.includes(origin)) return res.status(403).json({ error: 'Forbidden origin' });
    next();
});
app.use(express.json({ limit: '5mb' }));

// Database Connection. The kata section does not need it; the rest of the app does.
mongoose.connect(process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/mlcode', { serverSelectionTimeoutMS: 4000 })
    .then(() => console.log('MongoDB Connected'))
    .catch(err => console.log(`MongoDB not available (${err.message}). Kata still works; sign-in and the ML problem set need MongoDB.`));

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/submissions', submissionRoutes);
app.use('/api/problems', problemRoutes);
app.use('/api/execute', executeRoutes);
app.use('/api/discussions', discussionRoutes);
app.use('/api/kata', kataRoutes);

app.get('/', (req, res) => {
    res.send('ML Code API is running');
});

app.listen(PORT, HOST, () => {
    console.log(`Server running on http://${HOST}:${PORT}`);
});
