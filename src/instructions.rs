use anyhow::{Context, Result};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
const MAX_FILE_BYTES: u64 = 512 * 1024;

#[derive(Clone, Debug)]
pub struct SkillCatalog {
    pub(crate) entries: BTreeMap<String, PathBuf>,
}

impl SkillCatalog {
    pub fn names_and_paths(&self) -> Vec<String> {
        self.entries
            .iter()
            .map(|(name, path)| format!("- {name}: {}", path.display()))
            .collect()
    }
    pub fn path(&self, name: &str) -> Option<&Path> {
        self.entries.get(name).map(PathBuf::as_path)
    }
    pub fn into_entries(self) -> BTreeMap<String, PathBuf> {
        self.entries
    }
}

pub async fn catalog(working_dir: &Path) -> Result<SkillCatalog> {
    let home = directories::BaseDirs::new()
        .context("failed to locate user home directory")?
        .home_dir()
        .to_owned();
    let mut entries = BTreeMap::new();
    for dir in [
        home.join(".agents/skills"),
        home.join(".zex/skills"),
        working_dir.join(".zex/skills"),
    ] {
        if dir.is_dir() {
            collect_catalog(&dir, &mut entries).await?;
        }
    }
    Ok(SkillCatalog { entries })
}
pub async fn load(working_dir: &Path) -> Result<String> {
    let home = directories::BaseDirs::new()
        .context("failed to locate user home directory")?
        .home_dir()
        .to_owned();
    let mut out = vec!["You are Zex, a minimal AI agent core. Be concise and accurate. Use grep to search file contents, glob to find files, and bash only for other system commands. Use read, write, and edit for file operations. Use tool results to finish the task.".to_owned()];
    for path in [
        home.join(".zex/AGENTS.md"),
        working_dir.join(".zex/AGENTS.md"),
    ] {
        if let Some(s) = read_optional(&path).await? {
            out.push(format!("[Instructions from {}]\n{s}", path.display()));
        }
    }
    let skills = catalog(working_dir).await?.names_and_paths();
    if !skills.is_empty() {
        out.push(format!("[Available skills]\n{}", skills.join("\n")));
    }
    Ok(out.join("\n\n"))
}

async fn collect_catalog(dir: &Path, entries: &mut BTreeMap<String, PathBuf>) -> Result<()> {
    let mut pending = vec![dir.to_owned()];
    let mut visited = HashSet::new();
    while let Some(current) = pending.pop() {
        let identity = tokio::fs::canonicalize(&current)
            .await
            .unwrap_or(current.clone());
        if !visited.insert(identity) {
            continue;
        }
        let mut it = tokio::fs::read_dir(&current)
            .await
            .with_context(|| format!("failed to scan skills directory {}", current.display()))?;
        while let Some(entry) = it
            .next_entry()
            .await
            .context("failed to enumerate skills directory")?
        {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.file_name().is_some_and(|n| n == "SKILL.md")
                && let Some(name) = path
                    .parent()
                    .and_then(Path::file_name)
                    .and_then(|n| n.to_str())
            {
                entries.insert(name.to_owned(), path);
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
async fn collect_skill_indexes(dir: &Path, out: &mut Vec<String>) -> Result<()> {
    let mut pending = vec![dir.to_owned()];
    while let Some(current) = pending.pop() {
        let mut entries = tokio::fs::read_dir(&current)
            .await
            .with_context(|| format!("failed to scan skills directory {}", current.display()))?;
        while let Some(entry) = entries
            .next_entry()
            .await
            .context("failed to enumerate skills directory")?
        {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.file_name().is_some_and(|name| name == "SKILL.md") {
                let name = path
                    .parent()
                    .and_then(Path::file_name)
                    .and_then(|n| n.to_str())
                    .unwrap_or("unnamed");
                out.push(format!("- {name}: {}", path.display()));
            }
        }
    }
    Ok(())
}

pub async fn read_skill(path: &Path) -> Result<String> {
    let metadata = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("failed to inspect skill {}", path.display()))?;
    if metadata.len() > MAX_FILE_BYTES {
        anyhow::bail!("skill {} exceeds {} bytes", path.display(), MAX_FILE_BYTES);
    }
    tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("failed to read skill {}", path.display()))
}
async fn read_optional(path: &Path) -> Result<Option<String>> {
    let m = match tokio::fs::metadata(path).await {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("failed to inspect {}", path.display())),
    };
    if m.len() > MAX_FILE_BYTES {
        anyhow::bail!(
            "instruction file {} exceeds {} bytes",
            path.display(),
            MAX_FILE_BYTES
        );
    }
    Ok(Some(tokio::fs::read_to_string(path).await.with_context(
        || format!("failed to read {}", path.display()),
    )?))
}
pub fn global_zex_dir() -> PathBuf {
    directories::BaseDirs::new()
        .map(|b| b.home_dir().join(".zex"))
        .unwrap_or_else(|| PathBuf::from(".zex"))
}
