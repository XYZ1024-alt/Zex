//! Animation primitives for ambient motion: easing, sheen sweeps, breathing.
//!
//! All ambient color motion derives from the wall clock at render time, so no
//! tick state needs plumbing through `App`. State-continuous motion (scroll
//! glides, focus transitions) advances per frame in `App::tick_animations`.
//! Every animated factor freezes to its settled state under `cfg!(test)` so
//! the layout test suite stays deterministic.

use ratatui::style::Color;

/// True when motion must render its settled (final) state: unit tests.
pub(super) fn frozen() -> bool {
    cfg!(test)
}

/// Wall-clock milliseconds, the single time source for ambient motion.
pub(super) fn clock_millis() -> u64 {
    if frozen() {
        return 0;
    }
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

/// Linear RGB blend, the single primitive behind all animated color.
pub(super) fn blend_color(from: Color, to: Color, t: f64) -> Color {
    let (Color::Rgb(fr, fg, fb), Color::Rgb(tr, tg, tb)) = (from, to) else {
        return if t < 0.5 { from } else { to };
    };
    let t = t.clamp(0.0, 1.0);
    let lerp = |a: u8, b: u8| (a as f64 + (b as f64 - a as f64) * t).round() as u8;
    Color::Rgb(lerp(fr, tr), lerp(fg, tg), lerp(fb, tb))
}

/// Raised-cosine sheen brightness (0.0..=1.0) at `position`, for a highlight
/// band of half-width `band` sweeping across `0..span` once per `period_ms`.
/// Frozen to 0 (no highlight) in tests.
pub(super) fn sheen(position: f64, span: f64, period_ms: u64, band: f64) -> f64 {
    if frozen() {
        return 0.0;
    }
    let phase = (clock_millis() % period_ms) as f64 / period_ms as f64;
    let center = phase * (span + 2.0 * band) - band;
    let distance = (position - center).abs();
    if distance >= band {
        0.0
    } else {
        0.5 + 0.5 * (std::f64::consts::PI * distance / band).cos()
    }
}

/// Slow sinusoidal breathing in 0.0..=1.0 over `period_ms`. Frozen to 0
/// (base state) in tests.
pub(super) fn breath(period_ms: u64) -> f64 {
    if frozen() {
        return 0.0;
    }
    let phase = (clock_millis() % period_ms) as f64 / period_ms as f64;
    0.5 + 0.5 * (phase * std::f64::consts::TAU).sin()
}

/// Frame-rate-independent exponential approach: returns the next value when
/// `current` chases `target` with per-frame `factor` (0 < factor < 1).
pub(super) fn chase(current: f64, target: f64, factor: f64) -> f64 {
    current + (target - current) * factor
}
