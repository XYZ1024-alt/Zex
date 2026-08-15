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
    models: BTreeMap<(String, String), ModelInfo>,
    canonical_models: BTreeMap<String, ModelInfo>,
    unqualified_models: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelLimit {
    pub context: u64,
    pub output: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ModelInfo {
    capabilities: ThinkingCapabilities,
    limit: Option<ModelLimit>,
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
    #[serde(default)]
    id: String,
    reasoning: Option<bool>,
    #[serde(default)]
    reasoning_options: Vec<ReasoningOption>,
    interleaved: Option<serde_json::Value>,
    #[serde(default)]
    limit: Option<ApiLimit>,
}

#[derive(Debug, Deserialize)]
struct ApiLimit {
    context: Option<u64>,
    #[serde(default)]
    output: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ReasoningOption {
    Effort {
        #[serde(default)]
        values: Vec<Option<String>>,
    },
    Toggle,
    BudgetTokens {
        #[serde(rename = "min")]
        _min: Option<i64>,
        #[serde(rename = "max")]
        _max: Option<i64>,
    },
}

impl ModelsDevCatalog {
    #[cfg(test)]
    pub(crate) fn from_json(content: &[u8]) -> Result<Self> {
        Self::parse(content)
    }

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
            .map(|info| info.capabilities.clone())
            .or_else(|| self.unique_model_capabilities(model_id))
    }

    pub fn limits(&self, provider_id: &str, model_id: &str) -> Option<ModelLimit> {
        self.models
            .get(&(provider_id.to_owned(), model_id.to_owned()))
            .and_then(|info| info.limit)
            .or_else(|| self.unique_model_limit(model_id))
    }

    fn unqualified_matches(&self, model_id: &str) -> Option<&BTreeSet<String>> {
        self.unqualified_models.get(model_id).or_else(|| {
            model_id
                .rsplit_once('/')
                .and_then(|(_, id)| self.unqualified_models.get(id))
        })
    }

    fn unique_model_capabilities(&self, model_id: &str) -> Option<ThinkingCapabilities> {
        let matches = self.unqualified_matches(model_id)?;
        let mut capabilities = matches
            .iter()
            .filter_map(|canonical_id| self.canonical_models.get(canonical_id))
            .map(|info| &info.capabilities);
        let first = capabilities.next()?.clone();
        Some(capabilities.fold(first, merge_capabilities))
    }

    fn unique_model_limit(&self, model_id: &str) -> Option<ModelLimit> {
        let matches = self.unqualified_matches(model_id)?;
        let mut limits = matches
            .iter()
            .filter_map(|canonical_id| self.canonical_models.get(canonical_id))
            .filter_map(|info| info.limit);
        let first = limits.next()?;
        Some(limits.fold(first, merge_limits))
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
                let model_aliases = model_aliases(&model_id, &model.id);
                let info = model_info_from_model(model);
                let canonical_id = canonical_model_id(&provider_id, &model_id);
                for alias in model_aliases {
                    catalog
                        .unqualified_models
                        .entry(alias)
                        .or_default()
                        .insert(canonical_id.clone());
                }
                catalog
                    .canonical_models
                    .entry(canonical_id)
                    .or_insert_with(|| info.clone());
                catalog.models.insert((provider_id.clone(), model_id), info);
            }
        }
        Ok(catalog)
    }
}

fn canonical_model_id(provider_id: &str, model_id: &str) -> String {
    if model_id.contains('/') {
        model_id.to_owned()
    } else {
        format!("{provider_id}/{model_id}")
    }
}

fn model_aliases(model_key: &str, model_id: &str) -> BTreeSet<String> {
    let mut aliases = BTreeSet::new();
    for value in [model_key, model_id] {
        aliases.insert(value.to_owned());
        if let Some((_, unqualified)) = value.rsplit_once('/') {
            aliases.insert(unqualified.to_owned());
        }
    }
    aliases
}

fn merge_capabilities(
    mut merged: ThinkingCapabilities,
    candidate: &ThinkingCapabilities,
) -> ThinkingCapabilities {
    if !candidate.supports_reasoning_effort {
        return merged;
    }
    if !merged.supports_reasoning_effort {
        return candidate.clone();
    }

    merged.supported.extend(candidate.supported.iter().copied());
    merged.supported.sort_unstable();
    merged.supported.dedup();
    merged
        .reasoning_effort_map
        .extend(candidate.reasoning_effort_map.clone());
    merged.min_level = merged
        .supported
        .iter()
        .copied()
        .find(|level| *level != ThinkingLevel::Off)
        .unwrap_or(ThinkingLevel::Off);
    merged.max_level = merged
        .supported
        .iter()
        .copied()
        .rev()
        .find(|level| *level != ThinkingLevel::Off)
        .unwrap_or(ThinkingLevel::Off);
    merged.supports_interleaved_thinking |= candidate.supports_interleaved_thinking;
    merged
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

fn merge_limits(merged: ModelLimit, candidate: ModelLimit) -> ModelLimit {
    ModelLimit {
        context: merged.context.min(candidate.context),
        output: match (merged.output, candidate.output) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        },
    }
}

fn model_info_from_model(model: ApiModel) -> ModelInfo {
    let limit = model.limit.as_ref().and_then(|limit| {
        limit.context.map(|context| ModelLimit {
            context,
            output: limit.output,
        })
    });
    ModelInfo {
        capabilities: capabilities_from_model(model),
        limit,
    }
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
    for value in values.into_iter().flatten() {
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
    fn accepts_negative_budget_token_sentinels() {
        ModelsDevCatalog::parse(
            br#"{
                "nvidia": {
                    "models": {
                        "reasoning-model": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "budget_tokens", "min": -1, "max": 32768}
                            ]
                        }
                    }
                }
            }"#,
        )
        .unwrap();
    }

    #[test]
    fn ignores_null_effort_values() {
        let catalog = ModelsDevCatalog::parse(
            br#"{
                "sarvam": {
                    "models": {
                        "reasoning-model": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": [null, "low", "medium", "high"]}
                            ]
                        }
                    }
                }
            }"#,
        )
        .unwrap();

        assert_eq!(
            catalog
                .capabilities("sarvam", "reasoning-model")
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
    fn falls_back_when_matching_model_capabilities_agree() {
        let catalog = ModelsDevCatalog::parse(
            br#"{
                "one": {"models": {"unique": {"reasoning": false}}},
                "two": {"models": {"shared": {"reasoning": false}}},
                "three": {"models": {"shared": {"reasoning": false}}}
            }"#,
        )
        .unwrap();

        assert!(catalog.capabilities("gateway", "unique").is_some());
        assert_eq!(
            catalog
                .capabilities("gateway", "shared")
                .unwrap()
                .available_levels(),
            vec![ThinkingLevel::Off]
        );
    }

    #[test]
    fn resolves_namespaced_model_ids_when_capabilities_agree() {
        let catalog = ModelsDevCatalog::parse(
            br#"{
                "one": {
                    "models": {
                        "openai/gpt-5.4-mini": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": ["none", "low", "medium", "high", "xhigh"]}
                            ]
                        }
                    }
                },
                "two": {
                    "models": {
                        "openai/gpt-5.4-mini": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": ["none", "low", "medium", "high", "xhigh"]}
                            ]
                        }
                    }
                }
            }"#,
        )
        .unwrap();

        assert_eq!(
            catalog
                .capabilities("gateway", "gpt-5.4-mini")
                .unwrap()
                .available_levels(),
            vec![
                ThinkingLevel::Off,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
                ThinkingLevel::XHigh
            ]
        );
    }

    #[test]
    fn resolves_namespaced_model_id_despite_provider_specific_variants() {
        let catalog = ModelsDevCatalog::parse(
            br#"{
                "limited": {
                    "models": {
                        "gpt-5.4-mini": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": ["low", "medium", "high"]}
                            ]
                        }
                    }
                },
                "one": {
                    "models": {
                        "openai/gpt-5.4-mini": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": ["none", "low", "medium", "high", "xhigh"]}
                            ]
                        }
                    }
                },
                "two": {
                    "models": {
                        "openai/gpt-5.4-mini": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": ["none", "low", "medium", "high", "xhigh"]}
                            ]
                        }
                    }
                }
            }"#,
        )
        .unwrap();

        assert_eq!(
            catalog
                .capabilities("gateway", "openai/gpt-5.4-mini")
                .unwrap()
                .max_level,
            ThinkingLevel::XHigh
        );
    }

    #[test]
    fn merges_namespaced_model_capabilities() {
        let catalog = ModelsDevCatalog::parse(
            br#"{
                "one": {
                    "models": {
                        "vendor-a/gpt-5.4-mini": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": ["low", "medium", "high"]}
                            ]
                        }
                    }
                },
                "two": {
                    "models": {
                        "vendor-b/gpt-5.4-mini": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": ["low", "medium", "high", "xhigh"]}
                            ]
                        }
                    }
                }
            }"#,
        )
        .unwrap();

        assert_eq!(
            catalog
                .capabilities("gateway", "gpt-5.4-mini")
                .unwrap()
                .available_levels(),
            vec![
                ThinkingLevel::Off,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
                ThinkingLevel::XHigh
            ]
        );
    }

    #[test]
    fn merges_direct_and_namespaced_provider_variants() {
        let catalog = ModelsDevCatalog::parse(
            br#"{
                "limited": {
                    "models": {
                        "gpt-5.4-mini": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": ["minimal", "low", "medium", "high"]}
                            ]
                        }
                    }
                },
                "extended": {
                    "models": {
                        "openai/gpt-5.4-mini": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": ["none", "low", "medium", "high", "xhigh", "max"]}
                            ]
                        }
                    }
                }
            }"#,
        )
        .unwrap();

        assert_eq!(
            catalog
                .capabilities("gateway", "gpt-5.4-mini")
                .unwrap()
                .available_levels(),
            ThinkingLevel::ALL
        );
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

    #[test]
    fn parses_context_limits_and_merges_unqualified_matches_conservatively() {
        let catalog = ModelsDevCatalog::parse(
            br#"{
                "openai": {
                    "models": {
                        "gpt-5": {
                            "reasoning": false,
                            "limit": {"context": 400000, "output": 128000}
                        }
                    }
                },
                "azure": {
                    "models": {
                        "gpt-5": {
                            "reasoning": false,
                            "limit": {"context": 272000}
                        },
                        "other": {"reasoning": false}
                    }
                }
            }"#,
        )
        .unwrap();

        let exact = catalog.limits("openai", "gpt-5").unwrap();
        assert_eq!(exact.context, 400_000);
        assert_eq!(exact.output, Some(128_000));

        // Unknown provider falls back to unqualified matches and merges to
        // the smallest window so the budget never overestimates.
        let merged = catalog.limits("gateway", "gpt-5").unwrap();
        assert_eq!(merged.context, 272_000);
        assert_eq!(merged.output, Some(128_000));

        assert!(catalog.limits("azure", "other").is_none());
        assert!(catalog.limits("azure", "missing").is_none());
    }
}
