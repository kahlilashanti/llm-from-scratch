// generate.js

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
const embedSize = 64;

const TEMPERATURE = 0.7; // <<< Easy to tweak
const TOP_K = 5;         // <<< Only pick from top 5 most likely
const TOP_P = 0.9;       // <<< Cumulative probability 90%

// --- Helper Functions ---
function matmul(vec, matrix) {
    return matrix.map(row => dot(vec, row));
}

function dot(a, b) {
    return a.reduce((sum, val, idx) => sum + val * b[idx], 0);
}

function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

// --- Top-k + Top-p (nucleus) sampling ---
function sampleTopKTopP(probs, topK = 5, topP = 0.9, temperature = 1.0) {
    const sorted = probs
        .map((p, idx) => ({ p, idx }))
        .sort((a, b) => b.p - a.p);

    let candidates = sorted.slice(0, topK);

    // Normalize top-k
    let totalP = candidates.reduce((sum, x) => sum + x.p, 0);
    candidates = candidates.map(x => ({ ...x, p: x.p / totalP }));

    // Apply temperature
    if (temperature !== 1.0) {
        candidates = candidates.map(x => ({
            ...x,
            p: Math.pow(x.p, 1 / temperature)
        }));
        const newSum = candidates.reduce((sum, x) => sum + x.p, 0);
        candidates = candidates.map(x => ({ ...x, p: x.p / newSum }));
    }

    // Apply nucleus (top-p)
    let cumulative = 0;
    const nucleus = [];
    for (let c of candidates) {
        cumulative += c.p;
        nucleus.push(c);
        if (cumulative >= topP) break;
    }

    const r = Math.random();
    let acc = 0;
    for (let c of nucleus) {
        acc += c.p;
        if (r < acc) {
            return c.idx;
        }
    }
    return nucleus[nucleus.length - 1].idx; // fallback
}

// --- Clean Join ---
function cleanJoin(tokens) {
    return tokens
        .map(t => t.trim())
        .join(' ')
        .replace(/\s+\)/g, ')')
        .replace(/\(\s+/g, '(')
        .replace(/\s{2,}/g, ' ');
}

// --- Load Trained Classifier Weights ---
const classifierW = JSON.parse(fs.readFileSync('classifierW.json', 'utf-8'));

// --- Self-Attention Setup ---
const Wq = randomMatrix(embedSize, embedSize);
const Wk = randomMatrix(embedSize, embedSize);
const Wv = randomMatrix(embedSize, embedSize);
const Wout = randomMatrix(embedSize, embedSize);

function randomMatrix(rows, cols) {
    return Array.from({ length: rows }, () => randomVector(cols));
}

function randomVector(size) {
    return Array.from({ length: size }, () => (Math.random() * 2 - 1) * 0.01);
}

function selfAttention(tokens) {
    const embeds = tokens.map(t => embeddings[tokenToId[t]]);

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

// --- Text Generation ---
function generate(startTokens, maxLength = 20) {
    const output = [...startTokens];

    for (let step = 0; step < maxLength; step++) {
        const outputVectors = selfAttention(output);
        const lastVector = outputVectors[outputVectors.length - 1];

        const logits = classifierW[0].map((_, colIdx) =>
            lastVector.reduce((sum, val, rowIdx) => sum + val * classifierW[rowIdx][colIdx], 0)
        );

        const probs = softmax(logits);

        const idx = sampleTopKTopP(probs, TOP_K, TOP_P, TEMPERATURE);
        const nextToken = idToToken[idx];

        output.push(nextToken);
    }

    return output;
}

// --- Run Example ---
const start = ['Apple', 'calories'];
const generated = generate(start, 30); // longer sample to show more structure

console.log("Generated text:");
console.log(cleanJoin(generated));
