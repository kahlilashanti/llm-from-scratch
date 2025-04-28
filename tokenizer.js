// tokenizer.js

const fs = require('fs');

// --- Load text ---
const rawText = fs.readFileSync('food-calories.txt', 'utf-8');

// --- Tokenize text ---
let tokens = rawText
    .split(/[\s,]+/)       // split on spaces and commas
    .map(t => t.trim())
    .filter(t => t.length > 0);

// --- Special handling: keep [END] token as-is ---
tokens = tokens.map(t => {
    if (t === '[END]') return t;
    return t;
});

// --- Build vocabulary ---
const vocab = [...new Set(tokens)];
const tokenToId = {};
const idToToken = {};

vocab.forEach((token, idx) => {
    tokenToId[token] = idx;
    idToToken[idx] = token;
});

// --- Save mappings ---
fs.writeFileSync('token_to_id.json', JSON.stringify(tokenToId));
fs.writeFileSync('id_to_token.json', JSON.stringify(idToToken));

console.log(`Loaded ${rawText.length} characters.`);
console.log(`Tokenized into ${tokens.length} tokens.`);
console.log(`Vocabulary size: ${vocab.length}`);
console.log(`Example token: '${vocab[0]}' --> ID: ${tokenToId[vocab[0]]}`);
console.log(`Example ID: 0 --> Token: '${idToToken[0]}'`);
