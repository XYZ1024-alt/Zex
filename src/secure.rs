use anyhow::{Context, Result, bail};
use argon2::Argon2;
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use blake2::{
    Blake2bMac,
    digest::{KeyInit as MacKeyInit, Mac as _, consts::U32},
};
use chacha20poly1305::{
    XChaCha20Poly1305, XNonce,
    aead::{Aead, AeadCore, OsRng, Payload},
};
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

/// Marks a fingerprint as keyed. Stores written before keying contain bare hex
/// digests, and this tag is what lets migration tell the two apart.
pub(crate) const KEYED_FINGERPRINT_PREFIX: &str = "k1:";

type FingerprintMac = Blake2bMac<U32>;

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
    fingerprint_key: Zeroizing<[u8; 32]>,
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
        // Expanded from the content key instead of a second Argon2 pass: the
        // expensive derivation is already paid for, and a domain-separated MAC
        // keeps the two uses from sharing key material.
        let fingerprint_key = Zeroizing::new(keyed_digest(&key, b"zex/memory/fingerprint/v1")?);
        Ok(Self {
            key,
            fingerprint_key,
        })
    }

    /// Keyed digest for deduplication metadata. Fingerprints stay in plaintext
    /// so the dedupe index can be rebuilt at open without decrypting the whole
    /// store — which means an unkeyed digest would let anyone holding the file
    /// confirm a guess about what was observed. Keying removes that while
    /// keeping the value stable within the session.
    pub(crate) fn fingerprint(&self, bytes: &[u8]) -> Result<String> {
        let digest = keyed_digest(&self.fingerprint_key, bytes)?;
        let hex = digest[..16]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        Ok(format!("{KEYED_FINGERPRINT_PREFIX}{hex}"))
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

fn keyed_digest(key: &[u8; 32], bytes: &[u8]) -> Result<[u8; 32]> {
    let mut mac = <FingerprintMac as MacKeyInit>::new_from_slice(key)
        .map_err(|_| anyhow::anyhow!("failed to initialize the memory fingerprint key"))?;
    mac.update(bytes);
    Ok(mac.finalize().into_bytes().into())
}
