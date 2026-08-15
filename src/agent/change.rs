use std::path::PathBuf;

/// Largest file content carried in a [`FileChange`]. Larger files skip the
/// change record entirely so the event stream and view state stay small.
pub const CHANGE_MAX_BYTES: usize = 512 * 1024;

/// A file mutation performed by the `write` or `edit` tool, captured at the
/// execution site where the previous content is still known. `before` is
/// `None` when the tool created the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileChange {
    pub path: PathBuf,
    pub before: Option<String>,
    pub after: String,
}

impl FileChange {
    pub fn capture(path: PathBuf, before: Option<String>, after: String) -> Option<Self> {
        let before_len = before.as_ref().map_or(0, String::len);
        if before_len > CHANGE_MAX_BYTES || after.len() > CHANGE_MAX_BYTES {
            return None;
        }
        Some(Self {
            path,
            before,
            after,
        })
    }
}

/// `(added, removed)` changed line counts between `before` and `after`.
pub fn change_counts(change: &FileChange) -> (usize, usize) {
    let before = change.before.as_deref().unwrap_or("");
    let (_, old_changed, new_changed, _) = changed_line_ranges(before, &change.after);
    (new_changed.len(), old_changed.len())
}

/// Split `old` and `new` into a shared prefix, the changed regions, and a
/// shared suffix. Prefix trimming keeps the changed region local even when
/// both sides are whole files.
pub fn changed_line_ranges<'a>(
    old: &'a str,
    new: &'a str,
) -> (Vec<&'a str>, Vec<&'a str>, Vec<&'a str>, Vec<&'a str>) {
    let old_lines = old.lines().collect::<Vec<_>>();
    let new_lines = new.lines().collect::<Vec<_>>();
    let prefix_len = old_lines
        .iter()
        .zip(&new_lines)
        .take_while(|(old, new)| old == new)
        .count();
    let shared_remaining = old_lines
        .len()
        .saturating_sub(prefix_len)
        .min(new_lines.len().saturating_sub(prefix_len));
    let suffix_len = old_lines[prefix_len..]
        .iter()
        .rev()
        .zip(new_lines[prefix_len..].iter().rev())
        .take(shared_remaining)
        .take_while(|(old, new)| old == new)
        .count();
    let old_changed_end = old_lines.len().saturating_sub(suffix_len);
    let new_changed_end = new_lines.len().saturating_sub(suffix_len);
    (
        old_lines[..prefix_len].to_vec(),
        old_lines[prefix_len..old_changed_end].to_vec(),
        new_lines[prefix_len..new_changed_end].to_vec(),
        old_lines[old_changed_end..].to_vec(),
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{CHANGE_MAX_BYTES, FileChange, change_counts, changed_line_ranges};

    #[test]
    fn capture_drops_oversized_contents() {
        let small = "content".to_owned();
        let large = "x".repeat(CHANGE_MAX_BYTES + 1);
        assert!(
            FileChange::capture(PathBuf::from("a.txt"), Some(small.clone()), small.clone())
                .is_some()
        );
        assert!(FileChange::capture(PathBuf::from("a.txt"), Some(large.clone()), small).is_none());
        assert!(FileChange::capture(PathBuf::from("a.txt"), None, large).is_none());
    }

    #[test]
    fn counts_report_added_and_removed_lines() {
        let overwrite = FileChange {
            path: PathBuf::from("a.txt"),
            before: Some("keep\nold\nkeep\n".to_owned()),
            after: "keep\nnew\nkeep\n".to_owned(),
        };
        assert_eq!(change_counts(&overwrite), (1, 1));

        let created = FileChange {
            path: PathBuf::from("b.txt"),
            before: None,
            after: "one\ntwo\n".to_owned(),
        };
        assert_eq!(change_counts(&created), (2, 0));
    }

    #[test]
    fn changed_line_ranges_trim_shared_prefix_and_suffix() {
        let (prefix, old_changed, new_changed, suffix) =
            changed_line_ranges("a\nb\nc\nd", "a\nx\ny\nd");
        assert_eq!(prefix, vec!["a"]);
        assert_eq!(old_changed, vec!["b", "c"]);
        assert_eq!(new_changed, vec!["x", "y"]);
        assert_eq!(suffix, vec!["d"]);

        let (_, old_changed, new_changed, _) = changed_line_ranges("", "one\ntwo");
        assert!(old_changed.is_empty());
        assert_eq!(new_changed, vec!["one", "two"]);
    }
}
