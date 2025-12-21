#!/usr/bin/env node
/**
 * Helper script for cross-language validation of extractTokenFeatures.
 * 
 * Reads kotogram tokens from stdin (one token per line, JSON-escaped),
 * outputs extracted features as JSON array.
 * 
 * Usage:
 *   echo '["⌈ˢ猫ᵖnoun⌉", "⌈ˢをᵖparticle⌉"]' | node scripts/extract_features_ts.mjs
 */

import { extractTokenFeatures } from '../dist/kotogram.js';

// Read all input from stdin
let input = '';
process.stdin.setEncoding('utf8');

process.stdin.on('data', chunk => {
    input += chunk;
});

process.stdin.on('end', () => {
    try {
        const tokens = JSON.parse(input);
        const results = tokens.map(token => extractTokenFeatures(token));
        console.log(JSON.stringify(results));
    } catch (e) {
        console.error('Error:', e.message);
        process.exit(1);
    }
});
