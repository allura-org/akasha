use anyhow::Result;

/// Minimal wrapper around `tokenizers::Tokenizer` for incremental / final decoding.
/// Based on the helper in candle-examples, reimplemented here to avoid that dependency.
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn get_token(&self, text: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(text).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }
}

/// Decode a fixed slice of token IDs to a string.
pub fn decode_tokens(tokenizer: &tokenizers::Tokenizer, tokens: &[u32]) -> Result<String> {
    tokenizer
        .decode(tokens, true)
        .map_err(|e| anyhow::anyhow!("decode failed: {e}"))
}
