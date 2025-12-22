let customConfig = [];
let hasIgnoresFile = false;
try {
  require.resolve('./eslint.ignores.cjs');
  hasIgnoresFile = true;
} catch {
  // eslint.ignores.cjs doesn't exist
}

if (hasIgnoresFile) {
  const ignores = require('./eslint.ignores.cjs');
  customConfig.push({ ignores });
}

customConfig.push({
  files: ['tests-ts/**/*.ts'],
  rules: {
    '@typescript-eslint/no-floating-promises': 'off'
  }
});

module.exports = [...require('gts'), ...customConfig];
