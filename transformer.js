// transformer.js
const fs = require('fs');

// Load token mappings and embeddings
const tokenToId = JSON.parse(fs.readFileSync('token_to_id.json', 'utf-8'));
const idToToken = JSON.parse(fs.readFileSync('id_to_token.json', 'utf-8'));

// Fix embeddings loading
const rawEmbeddings = JSON.parse(fs.readFileSync('embeddings.json', 'utf-8'));
const embeddings = {};
for (const id in rawEmbeddings) {
    embeddings[id] = Object.values(rawEmbeddings[id]);
}

const vocabSize = Object.keys(tokenToId).length;
const embedSize = 32;

// Helper: Create random weight matrix
function randomMatrix(rows, cols) {
    return Array.from({ length: rows }, () => randomVector(cols));
}

function randomVector(size) {
    return Array.from({ length: size }, () => (Math.random() * 2 - 1));
}

// Initialize weights
const Wq = randomMatrix(embedSize, embedSize);
const Wk = randomMatrix(embedSize, embedSize);
const Wv = randomMatrix(embedSize, embedSize);
const Wout = randomMatrix(embedSize, embedSize);

// Matrix multiplication helper
function matmul(vec, matrix) {
    return matrix.map(row => dot(vec, row));
}

// Dot product helper
function dot(a, b) {
    return a.reduce((sum, val, idx) => sum + val * b[idx], 0);
}

// Softmax helper
function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map(x => x / sum);
}

// Self-attention block
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

// Example usage
const exampleTokens = ['Apple', 'calories'];
const output = selfAttention(exampleTokens);

console.log("Output vectors:", output);

// DONE.
