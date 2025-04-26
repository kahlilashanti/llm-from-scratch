// embeddings.js
const fs = require('fs');

// 1. Load token mappings
const tokenToId = JSON.parse(fs.readFileSync('token_to_id.json', 'utf-8'));
const idToToken = JSON.parse(fs.readFileSync('id_to_token.json', 'utf-8'));

const vocabSize = Object.keys(tokenToId).length;
const embedSize = 32; // You can change this to 64, 128 etc.

// 2. Create random embeddings
const embeddings = {};

function randomVector(size) {
    return Array.from({ length: size }, () => (Math.random() * 2 - 1)); // Random numbers between -1 and 1
}

for (let id in idToToken) {
    embeddings[id] = randomVector(embedSize);
}

console.log(`Created embeddings: ${vocabSize} tokens with ${embedSize}-dimensional vectors.`);

// 3. Lookup functions
function getEmbeddingByToken(token) {
    const id = tokenToId[token];
    if (id === undefined) {
        console.error(`Token '${token}' not found.`);
        return null;
    }
    return embeddings[id];
}

function getEmbeddingById(id) {
    if (embeddings[id] === undefined) {
        console.error(`ID '${id}' not found.`);
        return null;
    }
    return embeddings[id];
}

// Example usage:
console.log("Embedding for 'Apple':", getEmbeddingByToken('Apple'));
console.log("Embedding for ID 0:", getEmbeddingById(0));

// 4. Optional: Save embeddings
fs.writeFileSync('embeddings.json', JSON.stringify(embeddings, null, 2));

// DONE.
