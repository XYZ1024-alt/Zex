//! Chrome glyphs and the ZEX wordmark, with legacy-console fallbacks.
//!
//! Chrome glyphs: prompt arrow, left accent rail, height-tiered wordmark art.
//! Wordmark grids use one char per pixel, rendered two terminal columns wide:
//! `S` = ink pixel, `.` = empty.

/// `"❯ "` (U+276F) normally, `"> "` on legacy Windows consoles. Always 2 columns.
pub(super) fn prompt_arrow() -> &'static str {
    if is_legacy_windows_console() {
        "> "
    } else {
        "\u{276F} "
    }
}

/// `"┃"` (U+2503) normally, `"│"` (U+2502) on legacy Windows consoles.
pub(super) fn accent_bar() -> &'static str {
    if is_legacy_windows_console() {
        "\u{2502}"
    } else {
        "\u{2503}"
    }
}

/// Hide braille art on short terminals and on legacy ConHost (no font fallback).
/// Tests always take the modern path so logo-tier assertions stay deterministic.
pub(super) fn is_legacy_windows_console() -> bool {
    if cfg!(test) {
        return false;
    }
    cfg!(windows)
        && std::env::var_os("WT_SESSION").is_none()
        && std::env::var_os("TERM_PROGRAM").is_none()
        && std::env::var_os("TERM").is_none()
}

/// Height at or above which the small logo is shown (below it, no logo).
pub(super) const SMALL_LOGO_MIN_HEIGHT: u16 = 22;
/// Height at or above which the full logo is shown (stacked layout).
pub(super) const FULL_LOGO_MIN_HEIGHT: u16 = 24;
/// Minimum terminal width for the side-by-side hero box.
pub(super) const HERO_BOX_MIN_WIDTH: u16 = 90;

/// Full-size ZEX wordmark: chunky block letters (22px x 7 rows).
/// `S` = ink pixel, `.` = empty.
const LOGO_FULL: &str = "\
SSSSSS..SSSSSS..SS..SS
SSSSSS..SSSSSS..SS..SS
...SS...SS.......SSSS.
..SS....SSSSS.....SS..
.SS.....SS.......SSSS.
SSSSSS..SSSSSS..SS..SS
SSSSSS..SSSSSS..SS..SS";

/// Compact ZEX wordmark (19px x 5 rows) for mid-height terminals.
const LOGO_SMALL: &str = "\
SSSSS..SSSSS..S...S
...S...SS......S.S.
..S....SSSS.....S..
.S.....SS......S.S.
SSSSS..SSSSS..S...S";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LogoTier {
    Full,
    Small,
    Hidden,
}

pub(super) fn logo_tier(window_height: u16) -> LogoTier {
    if is_legacy_windows_console() || window_height < SMALL_LOGO_MIN_HEIGHT {
        LogoTier::Hidden
    } else if window_height < FULL_LOGO_MIN_HEIGHT {
        LogoTier::Small
    } else {
        LogoTier::Full
    }
}

pub(super) fn logo_art(tier: LogoTier) -> Option<&'static str> {
    match tier {
        LogoTier::Full => Some(LOGO_FULL),
        LogoTier::Small => Some(LOGO_SMALL),
        LogoTier::Hidden => None,
    }
}

/// Hero box always uses the full mark when logos are allowed.
pub(super) fn hero_logo_art() -> Option<&'static str> {
    if is_legacy_windows_console() {
        None
    } else {
        Some(LOGO_FULL)
    }
}

pub(super) fn logo_line_count(art: &str) -> u16 {
    art.lines().filter(|line| !line.is_empty()).count() as u16
}

pub(super) fn logo_visual_width(art: &str) -> u16 {
    // One sprite char per pixel, each pixel rendered two columns wide.
    art.lines()
        .filter(|line| !line.is_empty())
        .map(|line| line.chars().count() * 2)
        .max()
        .unwrap_or(16) as u16
}

pub(super) fn spinner_frame(tick: u64) -> &'static str {
    const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    FRAMES[(tick as usize / 4) % FRAMES.len()]
}
