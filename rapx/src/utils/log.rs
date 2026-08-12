use chrono::Local;
use fern::colors::{Color, ColoredLevelConfig};
use fern::{self, Dispatch};
use log::LevelFilter;
#[rustversion::since(1.83)]
use rustc_span::source_map::get_source_map;
#[rustversion::before(1.83)]
use rustc_span::source_map::SourceMap;
#[rustversion::before(1.83)]
use rustc_data_structures::sync::Lrc;
use rustc_span::{FileName, Pos, Span};
#[rustversion::before(1.83)]
use std::cell::RefCell;
use std::ops::Range;

#[rustversion::before(1.83)]
thread_local! {
    static RAP_SOURCE_MAP: RefCell<Option<Lrc<SourceMap>>> = RefCell::new(None);
}

#[rustversion::before(1.83)]
pub fn install_source_map(source_map: Lrc<SourceMap>) {
    RAP_SOURCE_MAP.with(|slot| *slot.borrow_mut() = Some(source_map));
}

#[rustversion::before(1.83)]
fn with_source_map<R>(callback: impl FnOnce(&SourceMap) -> R) -> R {
    RAP_SOURCE_MAP.with(|slot| {
        let source_map = slot.borrow();
        callback(
            source_map
                .as_deref()
                .expect("RAP source map must be installed before analysis"),
        )
    })
}

#[rustversion::since(1.83)]
fn with_source_map<R>(callback: impl FnOnce(&rustc_span::source_map::SourceMap) -> R) -> R {
    let source_map = get_source_map().expect("rustc source map is unavailable during analysis");
    callback(&source_map)
}

fn log_level() -> LevelFilter {
    if let Ok(s) = std::env::var("RAP_LOG") {
        match s.parse() {
            Ok(level) => return level,
            Err(err) => eprintln!("RAP_LOG is invalid: {err}"),
        }
    }
    LevelFilter::Info
}

/// Detect `RAP_LOG` environment variable first; if it's not set,
/// default to INFO level.
pub fn init_log() -> Result<(), fern::InitError> {
    let dispatch = Dispatch::new().level(log_level());

    let color_line = ColoredLevelConfig::new()
        .error(Color::Red)
        .warn(Color::Yellow)
        .info(Color::White)
        .debug(Color::Blue)
        .trace(Color::Cyan);

    let color_level = color_line.info(Color::Green);
    let stderr_dispatch = Dispatch::new()
        .format(move |callback, args, record| {
            let now = Local::now();
            callback.finish(format_args!(
                "{}{}|RAP|{}{}|: {}\x1B[0m",
                format_args!(
                    "\x1B[{}m",
                    color_line.get_color(&record.level()).to_fg_str()
                ),
                now.format("%H:%M:%S"),
                color_level.color(record.level()),
                format_args!(
                    "\x1B[{}m",
                    color_line.get_color(&record.level()).to_fg_str()
                ),
                args
            ))
        })
        .chain(std::io::stderr());

    /* Note that we cannot dispatch to stdout due to some bugs */
    dispatch.chain(stderr_dispatch).apply()?;
    Ok(())
}

#[macro_export]
macro_rules! rap_trace {
    ($($arg:tt)+) => (
        ::log::trace!(target: "RAP", $($arg)+)
    );
}

#[macro_export]
macro_rules! rap_debug {
    ($($arg:tt)+) => (
        ::log::debug!(target: "RAP", $($arg)+)
    );
}

#[macro_export]
macro_rules! rap_info {
    ($($arg:tt)+) => (
        ::log::info!(target: "RAP", $($arg)+)
    );
}

#[macro_export]
macro_rules! rap_warn {
    ($($arg:tt)+) => (
        ::log::warn!(target: "RAP", $($arg)+)
    );
}

#[macro_export]
macro_rules! rap_error {
    ($($arg:tt)+) => (
        ::log::error!(target: "RAP", $($arg)+)
    );
}

pub fn rap_error_and_exit(msg: impl AsRef<str>) -> ! {
    rap_error!("{}", msg.as_ref());
    std::process::exit(1)
}

#[inline]
pub fn span_to_source_code(span: Span) -> String {
    with_source_map(|source_map| source_map.span_to_snippet(span).unwrap())
}

#[inline]
pub fn span_to_first_line(span: Span) -> Span {
    // extend the span to an entrie line or extract the first line if it has multiple lines
    with_source_map(|source_map| source_map.span_extend_to_line(span.shrink_to_lo()))
}

#[inline]
pub fn span_to_trimmed_span(span: Span) -> Span {
    // trim out the first few whitespace
    with_source_map(|source_map| {
        span.trim_start(source_map.span_take_while(span, |c| c.is_whitespace()))
            .unwrap()
    })
}

#[inline]
#[rustversion::before(1.96)]
fn format_local_filename(filename: &FileName) -> String {
    filename.prefer_local().to_string()
}

#[inline]
#[rustversion::since(1.96)]
fn format_local_filename(filename: &FileName) -> String {
    filename.prefer_local_unconditionally().to_string()
}

#[inline]
pub fn span_to_filename(span: Span) -> String {
    let filename = with_source_map(|source_map| source_map.span_to_filename(span));
    format_local_filename(&filename)
}

#[inline]
pub fn span_to_line_number(span: Span) -> usize {
    with_source_map(|source_map| source_map.lookup_char_pos(span.lo()).line)
}

#[inline]
// this function computes the relative pos range of two spans which could be generated from two dirrerent files or not intersect with each other
// warning: we just return 0..0 to drop off the unintersected pairs
pub fn relative_pos_range(span: Span, sub_span: Span) -> Range<usize> {
    if sub_span.lo() < span.lo() || sub_span.hi() > span.hi() {
        return 0..0;
    }
    let offset = span.lo();
    let lo = (sub_span.lo() - offset).to_usize();
    let hi = (sub_span.hi() - offset).to_usize();
    lo..hi
}

pub fn are_spans_in_same_file(span1: Span, span2: Span) -> bool {
    with_source_map(|source_map| {
        let file1 = source_map.lookup_source_file(span1.lo());
        let file2 = source_map.lookup_source_file(span2.lo());
        file1.name == file2.name
    })
}
