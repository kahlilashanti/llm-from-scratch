// tokenizer.js
const fs = require('fs');

// 1. Load the text
const rawText = fs.readFileSync('food-calories.txt', 'utf-8');
console.log(`Loaded ${rawText.length} characters.`);

// 2. Simple Tokenizer (split on whitespace and basic punctuation)
function simpleTokenizer(text) {
    const tokens = text.split(/([\s,.!?;:\n]+)/).filter(Boolean);
    return tokens;
}

const tokens = simpleTokenizer(rawText);
console.log(`Tokenized into ${tokens.length} tokens.`);

// 3. Build Vocabulary
const vocabSet = new Set(tokens);
const vocab = Array.from(vocabSet);

console.log(`Vocabulary size: ${vocab.length}`);

// 4. Build Token -> ID and ID -> Token maps
const tokenToId = {};
const idToToken = {};
vocab.forEach((token, idx) => {
    tokenToId[token] = idx;
    idToToken[idx] = token;
});

// 5. Optional: Save vocab (JSON)
fs.writeFileSync('token_to_id.json', JSON.stringify(tokenToId, null, 2));
fs.writeFileSync('id_to_token.json', JSON.stringify(idToToken, null, 2));

// 6. Usage example:
console.log(`Example token: '${tokens[0]}' --> ID: ${tokenToId[tokens[0]]}`);
console.log(`Example ID: 0 --> Token: '${idToToken[0]}'`);

// DONE.
