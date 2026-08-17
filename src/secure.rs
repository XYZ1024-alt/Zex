use anyhow::{Context, Result, bail};
use argon2::Argon2;
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use chacha20poly1305::{
    XChaCha20Poly1305, XNonce,
    aead::{Aead, AeadCore, KeyInit, OsRng, Payload},
};
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

#[derive(Clone, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct EncryptionKey(String);

impl EncryptionKey {
    pub fn new(value: String) -> Option<Self> {
        (!value.trim().is_empty()).then_some(Self(value))
    }

    pub fn from_environment() -> Option<Self> {
        std::env::var("ZEX_MEMORY_ENCRYPTION_KEY")
            .ok()
            .and_then(Self::new)
    }

    fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for EncryptionKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("EncryptionKey([redacted])")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EncryptedContent {
    pub(crate) format: u8,
    pub(crate) nonce: String,
}

pub(crate) struct Cipher {
    key: Zeroizing<[u8; 32]>,
}

impl std::fmt::Debug for Cipher {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Cipher([redacted])")
    }
}

impl Cipher {
    pub(crate) fn new(secret: &EncryptionKey, context: &str) -> Result<Self> {
        let mut key = Zeroizing::new([0u8; 32]);
        Argon2::default()
            .hash_password_into(secret.expose().as_bytes(), context.as_bytes(), &mut *key)
            .map_err(|error| anyhow::anyhow!("failed to derive encryption key: {error}"))?;
        Ok(Self { key })
    }

    pub(crate) fn encrypt(
        &self,
        associated_data: &str,
        content: &str,
    ) -> Result<(String, EncryptedContent)> {
        let cipher = XChaCha20Poly1305::new_from_slice(&*self.key)
            .map_err(|_| anyhow::anyhow!("failed to initialize encryption"))?;
        let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
        let encrypted = cipher
            .encrypt(
                &nonce,
                Payload {
                    msg: content.as_bytes(),
                    aad: associated_data.as_bytes(),
                },
            )
            .map_err(|_| anyhow::anyhow!("failed to encrypt content"))?;
        Ok((
            BASE64.encode(encrypted),
            EncryptedContent {
                format: 1,
                nonce: BASE64.encode(nonce),
            },
        ))
    }

    pub(crate) fn decrypt(
        &self,
        associated_data: &str,
        content: &str,
        encryption: &EncryptedContent,
    ) -> Result<String> {
        if encryption.format != 1 {
            bail!(
                "unsupported content encryption format {}",
                encryption.format
            );
        }
        let nonce = BASE64
            .decode(&encryption.nonce)
            .context("failed to decode encryption nonce")?;
        if nonce.len() != 24 {
            bail!("encryption nonce has the wrong length");
        }
        let nonce = XNonce::from_slice(&nonce);
        let encrypted = BASE64
            .decode(content)
            .context("failed to decode encrypted content")?;
        let cipher = XChaCha20Poly1305::new_from_slice(&*self.key)
            .map_err(|_| anyhow::anyhow!("failed to initialize decryption"))?;
        let decrypted = cipher
            .decrypt(
                nonce,
                Payload {
                    msg: &encrypted,
                    aad: associated_data.as_bytes(),
                },
            )
            .map_err(|_| anyhow::anyhow!("content authentication failed"))?;
        String::from_utf8(decrypted).context("decrypted content is not UTF-8")
    }
}
