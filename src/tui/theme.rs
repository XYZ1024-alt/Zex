//! Runtime theme resolved from `[theme]` config overrides.
//!
//! The TUI renders against [`theme()`], which starts as [`DEFAULT_THEME`] and
//! is replaced by [`install_theme`] once the config is loaded at startup.

use std::sync::RwLock;

use ratatui::style::Color;

use crate::config::{ThemeColor, ThemeConfig};

/// The resolved palette every render call reads from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme {
    pub background: Color,
    pub surface: Color,
    pub surface_hover: Color,
    pub surface_raised: Color,
    pub text: Color,
    pub text_strong: Color,
    pub text_dim: Color,
    pub text_faint: Color,
    pub gray_dim: Color,
    pub accent_primary: Color,
    pub accent_secondary: Color,
    pub accent_user: Color,
    pub accent_thinking: Color,
    pub accent_tool: Color,
    pub border: Color,
    pub border_active: Color,
    pub ok: Color,
    pub bad: Color,
    pub command: Color,
    pub running: Color,
    pub model_accent: Color,
    pub md_code: Color,
    pub code_bg: Color,
    pub diff_add_bg: Color,
    pub diff_del_bg: Color,
    pub wordmark_ink: Color,
}

// Fixed dark ink used only as the blend target for fades/shimmers; never
// painted as a background.
pub const BASE_INK: Color = Color::Rgb(18, 20, 24);

// Kimi Code × oh-my-pi palette on a transparent base: the terminal's own
// background shows through, surfaces are neutral cool grays, cyan leads,
// violet thinks, amber runs commands. Diffs follow GitHub's dark washes.
pub const DEFAULT_THEME: Theme = Theme {
    background: Color::Reset,
    surface: Color::Rgb(29, 33, 41),
    surface_hover: Color::Rgb(38, 43, 52),
    surface_raised: Color::Rgb(49, 54, 63),
    text: Color::Rgb(212, 215, 221),
    text_strong: Color::Rgb(242, 244, 248),
    text_dim: Color::Rgb(119, 125, 136),
    text_faint: Color::Rgb(95, 102, 115),
    gray_dim: Color::Rgb(61, 66, 74),
    accent_primary: Color::Rgb(103, 232, 249),
    accent_secondary: Color::Rgb(178, 129, 214),
    accent_user: Color::Rgb(232, 227, 217),
    accent_thinking: Color::Rgb(178, 129, 214),
    // Tool chrome deliberately blends into the gray ramp; only status colors pop.
    accent_tool: Color::Rgb(95, 102, 115),
    border: Color::Rgb(61, 66, 74),
    border_active: Color::Rgb(23, 143, 185),
    ok: Color::Rgb(137, 210, 129),
    bad: Color::Rgb(252, 58, 75),
    // Semantic accents: cyan leads, violet thinks, amber runs commands.
    command: Color::Rgb(254, 188, 56),
    running: Color::Rgb(147, 197, 253),
    model_accent: Color::Rgb(215, 135, 175),
    md_code: Color::Rgb(229, 193, 255),
    code_bg: Color::Rgb(22, 26, 31),
    diff_add_bg: Color::Rgb(18, 38, 30),
    diff_del_bg: Color::Rgb(45, 18, 20),
    // Wordmark palette: a single cyan tone for the block letters,
    // animated by the shimmer sweep.
    wordmark_ink: Color::Rgb(103, 232, 249),
};

static ACTIVE: RwLock<Theme> = RwLock::new(DEFAULT_THEME);

/// The currently installed theme.
pub(crate) fn theme() -> Theme {
    *ACTIVE
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn apply(config: Option<ThemeColor>, current: Color) -> Color {
    match config {
        Some(ThemeColor::Terminal) => Color::Reset,
        Some(ThemeColor::Rgb(red, green, blue)) => Color::Rgb(red, green, blue),
        None => current,
    }
}

/// Resolve the theme from config overrides without installing it.
fn resolve(config: &ThemeConfig) -> Theme {
    let mut theme = DEFAULT_THEME;
    theme.background = apply(config.background, theme.background);
    theme.surface = apply(config.surface, theme.surface);
    theme.surface_hover = apply(config.surface_hover, theme.surface_hover);
    theme.surface_raised = apply(config.surface_raised, theme.surface_raised);
    theme.text = apply(config.text, theme.text);
    theme.text_strong = apply(config.text_strong, theme.text_strong);
    theme.text_dim = apply(config.text_dim, theme.text_dim);
    theme.text_faint = apply(config.text_faint, theme.text_faint);
    theme.gray_dim = apply(config.gray_dim, theme.gray_dim);
    theme.accent_primary = apply(config.accent_primary, theme.accent_primary);
    theme.accent_secondary = apply(config.accent_secondary, theme.accent_secondary);
    theme.accent_user = apply(config.accent_user, theme.accent_user);
    // Aliases resolve after their base key: unset accent_thinking follows the
    // (possibly overridden) accent_secondary, unset accent_tool follows text_faint.
    theme.accent_thinking = apply(config.accent_thinking, theme.accent_secondary);
    theme.accent_tool = apply(config.accent_tool, theme.text_faint);
    theme.border = apply(config.border, theme.border);
    theme.border_active = apply(config.border_active, theme.border_active);
    theme.ok = apply(config.ok, theme.ok);
    theme.bad = apply(config.bad, theme.bad);
    theme.command = apply(config.command, theme.command);
    theme.running = apply(config.running, theme.running);
    theme.model_accent = apply(config.model_accent, theme.model_accent);
    theme.md_code = apply(config.md_code, theme.md_code);
    theme.code_bg = apply(config.code_bg, theme.code_bg);
    theme.diff_add_bg = apply(config.diff_add_bg, theme.diff_add_bg);
    theme.diff_del_bg = apply(config.diff_del_bg, theme.diff_del_bg);
    theme.wordmark_ink = apply(config.wordmark_ink, theme.wordmark_ink);
    theme
}

/// Install the theme resolved from config. Called once at startup before the
/// TUI starts rendering.
pub fn install_theme(config: &ThemeConfig) {
    let mut active = ACTIVE
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *active = resolve(config);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_keys_keep_the_default_theme() {
        assert_eq!(resolve(&ThemeConfig::default()), DEFAULT_THEME);
    }

    #[test]
    fn aliases_follow_their_overridden_base_key() {
        let config = ThemeConfig {
            text_faint: Some(ThemeColor::Rgb(1, 2, 3)),
            accent_secondary: Some(ThemeColor::Rgb(4, 5, 6)),
            ..ThemeConfig::default()
        };
        let theme = resolve(&config);
        assert_eq!(theme.accent_tool, Color::Rgb(1, 2, 3));
        assert_eq!(theme.accent_thinking, Color::Rgb(4, 5, 6));
    }

    #[test]
    fn explicit_alias_overrides_win_and_terminal_maps_to_reset() {
        let config = ThemeConfig {
            accent_secondary: Some(ThemeColor::Rgb(4, 5, 6)),
            accent_thinking: Some(ThemeColor::Rgb(7, 8, 9)),
            background: Some(ThemeColor::Terminal),
            ..ThemeConfig::default()
        };
        let theme = resolve(&config);
        assert_eq!(theme.accent_thinking, Color::Rgb(7, 8, 9));
        assert_eq!(theme.background, Color::Reset);
    }
}
