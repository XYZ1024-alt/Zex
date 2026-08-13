use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;

use super::{ThinkingCapabilities, ThinkingLevel};

const MODELS_DEV_URL: &str = "https://models.dev/api.json";
const CACHE_FILE: &str = "models-dev-cache.json";
const REFRESH_TIMEOUT: Duration = Duration::from_secs(8);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelsDevCatalog {
    models: BTreeMap<(String, String), ThinkingCapabilities>,
    model_providers: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelsDevLoad {
    Refreshed,
    Cached,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelsDevProviderAlias {
    pub id: String,
    pub api: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiProvider {
    id: Option<String>,
    api: Option<String>,
    #[serde(default)]
    models: BTreeMap<String, ApiModel>,
}

#[derive(Debug, Deserialize)]
struct ApiModel {
    reasoning: Option<bool>,
    #[serde(default)]
    reasoning_options: Vec<ReasoningOption>,
    interleaved: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ReasoningOption {
    Effort {
        #[serde(default)]
        values: Vec<String>,
    },
    Toggle,
    BudgetTokens {
        #[serde(rename = "min")]
        _min: Option<u64>,
        #[serde(rename = "max")]
        _max: Option<u64>,
    },
}

impl ModelsDevCatalog {
    pub async fn load(
        cache_directory: &Path,
    ) -> (Self, Vec<ModelsDevProviderAlias>, ModelsDevLoad) {
        let cache_path = cache_directory.join(CACHE_FILE);
        if cfg!(test) {
            return match read_cache(&cache_path).await {
                Ok((catalog, aliases)) => (catalog, aliases, ModelsDevLoad::Cached),
                Err(_) => (Self::default(), Vec::new(), ModelsDevLoad::Unavailable),
            };
        }
        match refresh(&cache_path).await {
            Ok((catalog, aliases)) => (catalog, aliases, ModelsDevLoad::Refreshed),
            Err(_) => match read_cache(&cache_path).await {
                Ok((catalog, aliases)) => (catalog, aliases, ModelsDevLoad::Cached),
                Err(_) => (Self::default(), Vec::new(), ModelsDevLoad::Unavailable),
            },
        }
    }

    pub fn capabilities(&self, provider_id: &str, model_id: &str) -> Option<ThinkingCapabilities> {
        self.models
            .get(&(provider_id.to_owned(), model_id.to_owned()))
            .cloned()
            .or_else(|| {
                let providers = self.model_providers.get(model_id)?;
                (providers.len() == 1)
                    .then(|| {
                        let provider_id = providers.first()?;
                        self.models
                            .get(&(provider_id.clone(), model_id.to_owned()))
                            .cloned()
                    })
                    .flatten()
            })
    }

    pub fn provider_aliases(content: &[u8]) -> Result<Vec<ModelsDevProviderAlias>> {
        let api: BTreeMap<String, ApiProvider> =
            serde_json::from_slice(content).context("invalid models.dev response")?;
        Ok(api
            .into_iter()
            .map(|(provider_key, provider)| ModelsDevProviderAlias {
                id: provider.id.unwrap_or(provider_key),
                api: provider.api,
            })
            .collect())
    }

    fn parse(content: &[u8]) -> Result<Self> {
        let api: BTreeMap<String, ApiProvider> =
            serde_json::from_slice(content).context("invalid models.dev response")?;
        let mut catalog = Self::default();
        for (provider_key, provider) in api {
            let provider_id = provider.id.unwrap_or(provider_key);
            for (model_id, model) in provider.models {
                let capabilities = capabilities_from_model(model);
                catalog
                    .model_providers
                    .entry(model_id.clone())
                    .or_default()
                    .insert(provider_id.clone());
                catalog
                    .models
                    .insert((provider_id.clone(), model_id), capabilities);
            }
        }
        Ok(catalog)
    }
}

async fn refresh(cache_path: &Path) -> Result<(ModelsDevCatalog, Vec<ModelsDevProviderAlias>)> {
    let client = Client::builder()
        .timeout(REFRESH_TIMEOUT)
        .build()
        .context("failed to build models.dev client")?;
    let response = client
        .get(MODELS_DEV_URL)
        .send()
        .await
        .context("failed to refresh models.dev")?
        .error_for_status()
        .context("models.dev returned an error")?;
    let content = response
        .bytes()
        .await
        .context("failed to read models.dev response")?;
    let catalog = ModelsDevCatalog::parse(&content)?;
    let aliases = ModelsDevCatalog::provider_aliases(&content)?;
    let _ = write_cache(cache_path, &content).await;
    Ok((catalog, aliases))
}

async fn read_cache(cache_path: &Path) -> Result<(ModelsDevCatalog, Vec<ModelsDevProviderAlias>)> {
    let content = tokio::fs::read(cache_path)
        .await
        .with_context(|| format!("failed to read {}", cache_path.display()))?;
    Ok((
        ModelsDevCatalog::parse(&content)?,
        ModelsDevCatalog::provider_aliases(&content)?,
    ))
}

async fn write_cache(cache_path: &Path, content: &[u8]) -> Result<()> {
    if let Some(parent) = cache_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let temporary_path = temporary_cache_path(cache_path);
    tokio::fs::write(&temporary_path, content)
        .await
        .with_context(|| format!("failed to write {}", temporary_path.display()))?;
    match tokio::fs::rename(&temporary_path, cache_path).await {
        Ok(()) => Ok(()),
        Err(_error) if cfg!(windows) && cache_path.exists() => {
            tokio::fs::remove_file(cache_path)
                .await
                .with_context(|| format!("failed to replace {}", cache_path.display()))?;
            tokio::fs::rename(&temporary_path, cache_path)
                .await
                .with_context(|| format!("failed to replace {}", cache_path.display()))
        }
        Err(error) => {
            Err(error).with_context(|| format!("failed to cache {}", cache_path.display()))
        }
    }
}

fn temporary_cache_path(cache_path: &Path) -> PathBuf {
    cache_path.with_extension("json.tmp")
}

fn capabilities_from_model(model: ApiModel) -> ThinkingCapabilities {
    let supports_interleaved_thinking = model.interleaved.is_some();
    if model.reasoning != Some(true) {
        return ThinkingCapabilities::disabled(supports_interleaved_thinking);
    }

    if model.reasoning_options.is_empty() {
        return ThinkingCapabilities {
            supports_interleaved_thinking,
            ..ThinkingCapabilities::default()
        };
    }
    let effort_values = model.reasoning_options.into_iter().find_map(|option| {
        if let ReasoningOption::Effort { values } = option {
            Some(values)
        } else {
            None
        }
    });
    let Some(values) = effort_values else {
        return ThinkingCapabilities::disabled(supports_interleaved_thinking);
    };

    let mut supported = vec![ThinkingLevel::Off];
    let mut map = BTreeMap::new();
    for value in values {
        let Some(level) = parse_effort_level(&value) else {
            continue;
        };
        if level != ThinkingLevel::Off {
            map.insert(level, value);
        }
        supported.push(level);
    }
    if supported.iter().all(|level| *level == ThinkingLevel::Off) {
        return ThinkingCapabilities::disabled(supports_interleaved_thinking);
    }
    ThinkingCapabilities::from_discovered(supported, map, supports_interleaved_thinking)
}

fn parse_effort_level(value: &str) -> Option<ThinkingLevel> {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" | "off" => Some(ThinkingLevel::Off),
        "minimal" => Some(ThinkingLevel::Minimal),
        "low" => Some(ThinkingLevel::Low),
        "medium" => Some(ThinkingLevel::Medium),
        "high" => Some(ThinkingLevel::High),
        "xhigh" => Some(ThinkingLevel::XHigh),
        "max" => Some(ThinkingLevel::Max),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelsDevCatalog, ThinkingLevel};

    #[test]
    fn maps_effort_options_and_disables_toggle_only_models() {
        let catalog = ModelsDevCatalog::parse(
            br#"{
                "openai": {
                    "models": {
                        "codex": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": ["none", "high", "max"]}
                            ],
                            "interleaved": {"field": "reasoning_content"}
                        },
                        "toggle": {
                            "reasoning": true,
                            "reasoning_options": [{"type": "toggle"}]
                        }
                    }
                }
            }"#,
        )
        .unwrap();

        let codex = catalog.capabilities("openai", "codex").unwrap();
        assert_eq!(
            codex.available_levels(),
            vec![ThinkingLevel::Off, ThinkingLevel::High, ThinkingLevel::Max]
        );
        assert_eq!(codex.min_level, ThinkingLevel::High);
        assert_eq!(codex.max_level, ThinkingLevel::Max);
        assert!(codex.supports_interleaved_thinking);
        assert_eq!(
            codex
                .reasoning_effort_map
                .get(&ThinkingLevel::Max)
                .map(String::as_str),
            Some("max")
        );

        let toggle = catalog.capabilities("openai", "toggle").unwrap();
        assert_eq!(toggle.available_levels(), vec![ThinkingLevel::Off]);
        assert!(!toggle.supports_reasoning_effort);
    }

    #[test]
    fn uses_safe_defaults_when_reasoning_has_no_options() {
        let catalog = ModelsDevCatalog::parse(
            br#"{
                "anthropic": {
                    "models": {
                        "claude": {
                            "reasoning": true,
                            "reasoning_options": []
                        }
                    }
                }
            }"#,
        )
        .unwrap();

        assert_eq!(
            catalog
                .capabilities("anthropic", "claude")
                .unwrap()
                .available_levels(),
            vec![
                ThinkingLevel::Off,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High
            ]
        );
    }

    #[test]
    fn falls_back_to_a_globally_unique_model_id_only() {
        let catalog = ModelsDevCatalog::parse(
            br#"{
                "one": {"models": {"unique": {"reasoning": false}}},
                "two": {"models": {"shared": {"reasoning": false}}},
                "three": {"models": {"shared": {"reasoning": false}}}
            }"#,
        )
        .unwrap();

        assert!(catalog.capabilities("gateway", "unique").is_some());
        assert!(catalog.capabilities("gateway", "shared").is_none());
    }

    #[test]
    fn reads_provider_ids_and_api_aliases() {
        let aliases = ModelsDevCatalog::provider_aliases(
            br#"{
                "openai": {
                    "id": "openai",
                    "api": "https://api.openai.com/v1",
                    "models": {}
                }
            }"#,
        )
        .unwrap();

        assert_eq!(aliases[0].id, "openai");
        assert_eq!(aliases[0].api.as_deref(), Some("https://api.openai.com/v1"));
    }
}
