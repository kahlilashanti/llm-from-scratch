// forward_pass.js

const fs = require('fs');

// --- Load token mappings and embeddings ---
const tokenToId = JSON.parse(fs.readFileSync('token_to_id.json', 'utf-8'));
const idToToken = JSON.parse(fs.readFileSync('id_to_token.json', 'utf-8'));
const rawEmbeddings = JSON.parse(fs.readFileSync('embeddings.json', 'utf-8'));

const embeddings = {};
for (const id in rawEmbeddings) {
    embeddings[id] = Object.values(rawEmbeddings[id]);
}

const vocabSize = Object.keys(tokenToId).length;
const embedSize = 32;

// --- Helper Functions ---
function randomMatrix(rows, cols) {
    return Array.from({ length: rows }, () => randomVector(cols));
}

function randomVector(size) {
    return Array.from({ length: size }, () => (Math.random() * 2 - 1));
}

function matmul(vec, matrix) {
    return matrix.map(row => dot(vec, row));
}

function dot(a, b) {
    return a.reduce((sum, val, idx) => sum + val * b[idx], 0);
}

function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map(x => x / sum);
}

// --- Initialize Transformer Weights ---
const Wq = randomMatrix(embedSize, embedSize);
const Wk = randomMatrix(embedSize, embedSize);
const Wv = randomMatrix(embedSize, embedSize);
const Wout = randomMatrix(embedSize, embedSize);

// --- Transformer Self-Attention ---
function selfAttention(tokens) {
    const embeds = tokens.map(t => {
        const id = tokenToId[t];
        if (id === undefined) {
            console.error(`Unknown token: '${t}'`);
            process.exit(1);
        }
        return embeddings[id];
    });

    const queries = embeds.map(e => matmul(e, Wq));
    const keys = embeds.map(e => matmul(e, Wk));
    const values = embeds.map(e => matmul(e, Wv));

    const attended = [];
    for (let i = 0; i < queries.length; i++) {
        const scores = keys.map(k => dot(queries[i], k) / Math.sqrt(embedSize));
        const attn = softmax(scores);

        const context = Array(embedSize).fill(0);
        for (let j = 0; j < attn.length; j++) {
            for (let k = 0; k < embedSize; k++) {
                context[k] += attn[j] * values[j][k];
            }
        }
        attended.push(matmul(context, Wout));
    }

    return attended;
}

// --- Simple Forward Pass ---
function predictNextToken(contextTokens) {
    const outputVectors = selfAttention(contextTokens);

    // Use the last output vector to predict next token
    const lastVector = outputVectors[outputVectors.length - 1];

    // Simple classifier: project to vocab size
    const classifierW = randomMatrix(embedSize, vocabSize);
    const logits = matmul(lastVector, classifierW);

    const probs = softmax(logits);

    // Pick the most likely token
    const maxIdx = probs.indexOf(Math.max(...probs));

    return idToToken[maxIdx];
}

// --- Example Usage ---
const exampleTokens = ['Apple', 'calories'];
console.log("Context tokens:", exampleTokens);

const nextToken = predictNextToken(exampleTokens);
console.log("Predicted next token:", nextToken);
