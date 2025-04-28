// embeddings.js

const fs = require('fs');

// --- Load token mappings ---
const tokenToId = JSON.parse(fs.readFileSync('token_to_id.json', 'utf-8'));
const idToToken = JSON.parse(fs.readFileSync('id_to_token.json', 'utf-8'));

const vocabSize = Object.keys(tokenToId).length;
const embedSize = 64; // <<<<< UPGRADED to 64 dimensions

// --- Helper ---
function randomVector(size) {
    return Array.from({ length: size }, () => (Math.random() * 2 - 1) * 0.01);
}

// --- Create Embeddings ---
const embeddings = {};

for (let token in tokenToId) {
    embeddings[tokenToId[token]] = randomVector(embedSize);
}

// --- Save ---
fs.writeFileSync('embeddings.json', JSON.stringify(embeddings));

console.log(`Created embeddings: ${vocabSize} tokens with ${embedSize}-dimensional vectors.`);
console.log(`Embedding for 'Apple':`);
console.log(embeddings[tokenToId['Apple']]);
console.log(`Embedding for ID 0:`);
console.log(embeddings[0]);
