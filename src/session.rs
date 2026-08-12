use std::{
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};

use crate::agent::Message;

#[derive(Debug, Clone)]
pub struct SessionStore {
    directory: PathBuf,
}

impl SessionStore {
    pub fn new(directory: PathBuf) -> Self {
        Self { directory }
    }

    pub async fn save(&self, messages: &[Message]) -> Result<PathBuf> {
        tokio::fs::create_dir_all(&self.directory)
            .await
            .with_context(|| {
                format!(
                    "failed to create session directory {}",
                    self.directory.display()
                )
            })?;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before the Unix epoch")?
            .as_millis();
        let path = self.directory.join(format!("{timestamp}.json"));
        let content =
            serde_json::to_vec_pretty(messages).context("failed to serialize the session")?;
        tokio::fs::write(&path, content)
            .await
            .with_context(|| format!("failed to save session {}", path.display()))?;
        Ok(path)
    }

    pub async fn load_latest(&self) -> Result<Option<Vec<Message>>> {
        let mut entries = match tokio::fs::read_dir(&self.directory).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to read session directory {}",
                        self.directory.display()
                    )
                });
            }
        };
        let mut latest: Option<PathBuf> = None;

        while let Some(entry) = entries
            .next_entry()
            .await
            .context("failed to inspect saved sessions")?
        {
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }

            if latest
                .as_ref()
                .is_none_or(|current| file_name(&path) > file_name(current.as_path()))
            {
                latest = Some(path);
            }
        }

        let Some(path) = latest else {
            return Ok(None);
        };
        let content = tokio::fs::read(&path)
            .await
            .with_context(|| format!("failed to read session {}", path.display()))?;
        let messages = serde_json::from_slice(&content)
            .with_context(|| format!("failed to parse session {}", path.display()))?;
        Ok(Some(messages))
    }
}

fn file_name(path: &Path) -> &std::ffi::OsStr {
    path.file_name().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::{
        process,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use crate::agent::Message;

    use super::SessionStore;

    #[tokio::test]
    async fn loads_the_most_recent_saved_session() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("zex-session-{}-{unique}", process::id()));
        let store = SessionStore::new(directory.clone());
        let first = vec![Message::User {
            content: "first".to_owned(),
        }];
        let second = vec![Message::User {
            content: "second".to_owned(),
        }];

        store.save(&first).await.unwrap();
        tokio::time::sleep(Duration::from_millis(2)).await;
        store.save(&second).await.unwrap();

        assert_eq!(store.load_latest().await.unwrap(), Some(second));
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }
}
