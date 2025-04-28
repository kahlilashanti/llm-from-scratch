// train.js

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
const learningRate = 0.01;

// --- Helper Functions ---
function randomMatrix(rows, cols) {
    return Array.from({ length: rows }, () => randomVector(cols));
}

function randomVector(size) {
    return Array.from({ length: size }, () => (Math.random() * 2 - 1) * 0.01);
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
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

function crossEntropyLoss(probs, trueIdx) {
    return -Math.log(probs[trueIdx] + 1e-9);
}

// --- Initialize Weights ---
let Wq = randomMatrix(embedSize, embedSize);
let Wk = randomMatrix(embedSize, embedSize);
let Wv = randomMatrix(embedSize, embedSize);
let Wout = randomMatrix(embedSize, embedSize);
let classifierW = randomMatrix(embedSize, vocabSize);

// --- Self-Attention ---
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

// --- Training Data Prep ---
const rawText = fs.readFileSync('food-calories.txt', 'utf-8');
const lines = rawText.split('\n').filter(Boolean);
const trainingPairs = [];

for (let line of lines) {
    const tokens = line.split(/[\s,]+/).filter(Boolean);
    for (let i = 0; i < tokens.length - 1; i++) {
        trainingPairs.push({
            context: tokens.slice(0, i + 1),
            target: tokens[i + 1],
        });
    }
}

// --- Training Loop ---
console.log(`Training on ${trainingPairs.length} samples...`);

for (let epoch = 0; epoch < 200; epoch++) {
    let totalLoss = 0;

    for (let sample of trainingPairs) {
        const { context, target } = sample;

        const outputVectors = selfAttention(context);
        const lastVector = outputVectors[outputVectors.length - 1];

        const logits = classifierW[0].map((_, colIdx) =>
            lastVector.reduce((sum, val, rowIdx) => sum + val * classifierW[rowIdx][colIdx], 0)
        );

        const probs = softmax(logits);

        const targetId = tokenToId[target];
        if (targetId === undefined) {
            continue; // skip unknown token
        }
        const loss = crossEntropyLoss(probs, targetId);
        totalLoss += loss;

        // --- Manual Gradient Descent ---
        const grad = probs.slice();
        grad[targetId] -= 1; // dL/dz for softmax + cross-entropy

        // Update classifier weights
        for (let i = 0; i < vocabSize; i++) {
            for (let j = 0; j < embedSize; j++) {
                classifierW[j][i] -= learningRate * grad[i] * lastVector[j];
            }
        }

        // --- NEW: Update Wq, Wk, Wv, Wout slightly ---
        for (let i = 0; i < embedSize; i++) {
            for (let j = 0; j < embedSize; j++) {
                Wq[i][j] -= learningRate * 0.01 * (Math.random() * 2 - 1);
                Wk[i][j] -= learningRate * 0.01 * (Math.random() * 2 - 1);
                Wv[i][j] -= learningRate * 0.01 * (Math.random() * 2 - 1);
                Wout[i][j] -= learningRate * 0.01 * (Math.random() * 2 - 1);
            }
        }
    }

    if (epoch % 10 === 0) {
        console.log(`Epoch ${epoch}: Loss = ${totalLoss.toFixed(4)}`);
    }
}

console.log("Training complete!");

// --- Save classifier weights (optional if needed later) ---
fs.writeFileSync('classifierW.json', JSON.stringify(classifierW));
