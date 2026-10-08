//! Split one Pi assistant text segment into pieces.
//!
//! The parser owns the text. A labeler only names the shape. This module is the
//! parser, the trailing-question splitter, and the deterministic baseline labeler.

use pulldown_cmark::{Event, Options, Parser, TagEnd};

/// Unicode scalar values. Matches the Pi interaction body cap.
pub const MAX_SEGMENT_CHARS: usize = 16_000;

/// Bump when block boundaries or heading rules change.
/// The cursor-subset snapshot test fails until this matches the new boundaries hash.
pub const PIECE_PARSER_VERSION: &str = "1";

/// Bump when the baseline kind rules change.
/// The cursor-subset snapshot test fails until this matches the new kinds hash.
pub const PIECE_BASELINE_VERSION: &str = "1";

/// HTML index contract generation.
/// `1` was a `div` index and put the line number inside `pre`.
/// `2` is `nav`, `section`, `h2`, `ul`, and `data-n` on `pre`.
pub const HTML_INDEX_SHAPE_VERSION: &str = "2";

/// A heading lead-in is one line no longer than this.
pub const MAX_HEADING_CHARS: usize = 80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceKind {
    Statement,
    Options,
    Steps,
    Instructions,
    Example,
    Caveat,
    Code,
    Question,
}

impl PieceKind {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Statement => "statement",
            Self::Options => "options",
            Self::Steps => "steps",
            Self::Instructions => "instructions",
            Self::Example => "example",
            Self::Caveat => "caveat",
            Self::Code => "code",
            Self::Question => "question",
        }
    }

    pub fn from_label(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "statement" => Some(Self::Statement),
            "options" => Some(Self::Options),
            "steps" => Some(Self::Steps),
            "instructions" => Some(Self::Instructions),
            "example" => Some(Self::Example),
            "caveat" => Some(Self::Caveat),
            "code" => Some(Self::Code),
            "question" => Some(Self::Question),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PieceItem {
    /// 1-based position within the piece. Distinct from [`Piece::order`].
    pub position: u32,
    pub text: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Piece {
    /// 1-based position of this piece in the segment.
    pub order: u32,
    pub kind: PieceKind,
    pub heading: Option<String>,
    pub items: Vec<PieceItem>,
    /// UTF-8 byte offsets into the original segment.
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredSegment {
    pub pieces: Vec<Piece>,
    /// The pieces' ranges tile the segment (gaps are whitespace or horizontal rules).
    pub tiled: bool,
    /// `false` when the segment was over the character cap and was not parsed.
    pub parsed: bool,
}

/// `narration` is a `stopReason: toolUse` segment. It is parsed, and every
/// block stays a [`PieceKind::Statement`] except fences, which stay
/// [`PieceKind::Code`].
pub fn structure_segment(text: &str, narration: bool) -> StructuredSegment {
    if text.chars().count() > MAX_SEGMENT_CHARS {
        return StructuredSegment {
            pieces: vec![whole_statement(text)],
            tiled: true,
            parsed: false,
        };
    }
    let mut blocks = parse_blocks(text);
    attach_headings(&mut blocks, text);
    let mut blocks = split_questions(blocks, text);
    label_blocks(&mut blocks, text, narration);
    commit_blocks(text, blocks)
}

/// Keep `blocks` when they tile `text`. Otherwise the segment is one statement
/// covering the whole text, with `tiled` false.
fn commit_blocks(text: &str, mut blocks: Vec<Piece>) -> StructuredSegment {
    if !ranges_tile(text, &blocks) {
        return StructuredSegment {
            pieces: vec![whole_statement(text)],
            tiled: false,
            parsed: true,
        };
    }
    for (i, block) in blocks.iter_mut().enumerate() {
        block.order = (i + 1) as u32;
    }
    StructuredSegment {
        pieces: blocks,
        tiled: true,
        parsed: true,
    }
}

fn whole_statement(text: &str) -> Piece {
    Piece {
        order: 1,
        kind: PieceKind::Statement,
        heading: None,
        items: Vec::new(),
        start: 0,
        end: text.len(),
    }
}

fn parse_blocks(text: &str) -> Vec<Piece> {
    let options = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH;
    let parser = Parser::new_ext(text, options).into_offset_iter();
    let mut pieces = Vec::new();
    let mut list_depth = 0i32;
    let mut quote_depth = 0i32;
    let mut table_depth = 0i32;
    let mut item_starts: Vec<usize> = Vec::new();
    let mut current_items: Vec<PieceItem> = Vec::new();
    let mut list_start: Option<usize> = None;

    for (event, range) in parser {
        match event {
            Event::Start(tag) => match tag {
                pulldown_cmark::Tag::List(_) if quote_depth == 0 && table_depth == 0 => {
                    if list_depth == 0 {
                        list_start = Some(range.start);
                        current_items.clear();
                    }
                    list_depth += 1;
                }
                pulldown_cmark::Tag::Item if list_depth == 1 => {
                    item_starts.push(range.start);
                }
                pulldown_cmark::Tag::BlockQuote(_) if list_depth == 0 && table_depth == 0 => {
                    quote_depth += 1;
                }
                pulldown_cmark::Tag::Table(_) if list_depth == 0 && quote_depth == 0 => {
                    table_depth += 1;
                }
                _ => {}
            },
            Event::End(tag) => match tag {
                TagEnd::Heading(_) if list_depth == 0 && quote_depth == 0 && table_depth == 0 => {
                    pieces.push(Piece {
                        order: 0,
                        kind: PieceKind::Statement,
                        heading: Some(heading_text(&text[range.clone()])),
                        items: Vec::new(),
                        start: range.start,
                        end: range.end,
                    });
                }
                TagEnd::Paragraph if list_depth == 0 && quote_depth == 0 && table_depth == 0 => {
                    pieces.push(plain(range.start, range.end));
                }
                TagEnd::CodeBlock if list_depth == 0 && quote_depth == 0 && table_depth == 0 => {
                    let mut piece = plain(range.start, range.end);
                    piece.kind = PieceKind::Code;
                    pieces.push(piece);
                }
                TagEnd::List(_) if quote_depth == 0 && table_depth == 0 => {
                    list_depth -= 1;
                    if list_depth == 0 {
                        let start = list_start.take().unwrap_or(range.start);
                        let mut piece = plain(start, range.end);
                        piece.items = std::mem::take(&mut current_items);
                        pieces.push(piece);
                    }
                }
                TagEnd::Item if list_depth == 1 => {
                    if let Some(start) = item_starts.pop() {
                        let body = item_body(text, start, range.end);
                        current_items.push(PieceItem {
                            position: (current_items.len() + 1) as u32,
                            text: body,
                            start,
                            end: range.end,
                        });
                    }
                }
                TagEnd::BlockQuote(_) if list_depth == 0 && table_depth == 0 => {
                    quote_depth -= 1;
                    if quote_depth == 0 {
                        pieces.push(plain(range.start, range.end));
                    }
                }
                TagEnd::Table if list_depth == 0 && quote_depth == 0 => {
                    table_depth -= 1;
                    if table_depth == 0 {
                        pieces.push(plain(range.start, range.end));
                    }
                }
                TagEnd::HtmlBlock if list_depth == 0 && quote_depth == 0 && table_depth == 0 => {
                    pieces.push(plain(range.start, range.end));
                }
                _ => {}
            },
            Event::Rule => {}
            _ => {}
        }
    }
    pieces
}

fn plain(start: usize, end: usize) -> Piece {
    Piece {
        order: 0,
        kind: PieceKind::Statement,
        heading: None,
        items: Vec::new(),
        start,
        end,
    }
}

fn heading_text(source: &str) -> String {
    source
        .lines()
        .next()
        .unwrap_or(source)
        .trim()
        .trim_start_matches('#')
        .trim()
        .trim_end_matches(|c: char| c == '=' || c == '-' || c.is_whitespace())
        .trim()
        .to_string()
}

/// Item body without the list marker. Offsets still cover the whole item.
fn item_body(text: &str, start: usize, end: usize) -> String {
    let raw = &text[start..end];
    let line = raw.lines().next().unwrap_or(raw);
    let stripped = strip_marker(line);
    if raw.lines().nth(1).is_some() {
        let mut out = stripped.to_string();
        let rest_off = line.len();
        let rest = raw.get(rest_off..).unwrap_or("");
        out.push_str(rest);
        out.trim().to_string()
    } else {
        stripped.trim().to_string()
    }
}

fn strip_marker(line: &str) -> &str {
    let trimmed = line.trim_start();
    if let Some(rest) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))
    {
        return rest;
    }
    let bytes = trimmed.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i > 0 && bytes.get(i) == Some(&b'.') {
        let after = trimmed.get(i + 1..).unwrap_or("");
        return after.strip_prefix(' ').unwrap_or(after);
    }
    trimmed
}

fn attach_headings(pieces: &mut Vec<Piece>, text: &str) {
    let mut out = Vec::new();
    let mut i = 0;
    while i < pieces.len() {
        let piece = &pieces[i];
        let is_md_heading = piece.heading.is_some()
            && piece.items.is_empty()
            && piece.kind != PieceKind::Code
            && is_heading_block(text, piece);
        let is_lead_in = is_colon_lead_in(text, piece);
        let next_is_list_or_fence = pieces
            .get(i + 1)
            .is_some_and(|n| !n.items.is_empty() || n.kind == PieceKind::Code);
        let attach = if is_md_heading {
            pieces.get(i + 1).is_some() && !is_heading_block(text, &pieces[i + 1])
        } else {
            is_lead_in && next_is_list_or_fence
        };
        if attach {
            let heading_src = &text[piece.start..piece.end];
            let heading = if is_md_heading {
                piece.heading.clone()
            } else {
                Some(lead_in_label(heading_src))
            };
            let mut next = pieces[i + 1].clone();
            next.heading = heading;
            next.start = piece.start;
            out.push(next);
            i += 2;
        } else if is_md_heading {
            let mut kept = piece.clone();
            kept.heading = None;
            kept.kind = PieceKind::Statement;
            out.push(kept);
            i += 1;
        } else {
            out.push(piece.clone());
            i += 1;
        }
    }
    *pieces = out;
}

fn is_heading_block(text: &str, piece: &Piece) -> bool {
    if piece.heading.is_none() {
        return false;
    }
    let src = text[piece.start..piece.end].trim();
    src.starts_with('#') || src.starts_with("==") || {
        let lines: Vec<&str> = src.lines().collect();
        lines.len() == 2 && lines[1].chars().all(|c| c == '=' || c == '-' || c == ' ')
    }
}

fn is_colon_lead_in(text: &str, piece: &Piece) -> bool {
    if piece.kind == PieceKind::Code || !piece.items.is_empty() || piece.heading.is_some() {
        return false;
    }
    let src = text[piece.start..piece.end].trim();
    if src.contains('\n') || src.chars().count() > MAX_HEADING_CHARS {
        return false;
    }
    let inner = src
        .strip_prefix("**")
        .and_then(|s| s.strip_suffix("**"))
        .unwrap_or(src);
    inner.trim().ends_with(':')
}

fn lead_in_label(source: &str) -> String {
    let trimmed = source.trim();
    let inner = trimmed
        .strip_prefix("**")
        .and_then(|s| s.strip_suffix("**"))
        .unwrap_or(trimmed);
    inner.trim().trim_end_matches(':').trim().to_string()
}

fn split_questions(pieces: Vec<Piece>, text: &str) -> Vec<Piece> {
    let mut out = Vec::new();
    for piece in pieces {
        if piece.kind == PieceKind::Code || !piece.items.is_empty() {
            out.push(piece);
            continue;
        }
        match trailing_question_split(text, piece.start, piece.end) {
            None => out.push(piece),
            Some(split_at) => {
                if split_at > piece.start {
                    let mut head = piece.clone();
                    head.end = split_at;
                    head.heading = piece.heading.clone();
                    out.push(head);
                    let mut tail = piece;
                    tail.heading = None;
                    tail.start = split_at;
                    out.push(tail);
                } else {
                    out.push(piece);
                }
            }
        }
    }
    out
}

/// Byte index where the trailing run of questions begins, if the block ends
/// in a real question mark and a split (or a whole-block question) applies.
/// Returns `Some(start)` when the whole block is the question run, which the
/// caller treats as "no cut".
fn trailing_question_split(text: &str, start: usize, end: usize) -> Option<usize> {
    let src = &text[start..end];
    let masked = mask_protected(src);
    let trimmed_end = src.trim_end();
    if trimmed_end.is_empty() {
        return None;
    }
    let last_rel = trimmed_end.len() - 1;
    if masked.as_bytes().get(last_rel) != Some(&b'?') {
        return None;
    }
    let boundaries = sentence_starts(&masked);
    if boundaries.is_empty() {
        return Some(start);
    }
    let mut run_start = boundaries.len();
    for (i, &b) in boundaries.iter().enumerate().rev() {
        let sentence_end = if i + 1 < boundaries.len() {
            boundaries[i + 1]
        } else {
            trimmed_end.len()
        };
        let sentence = src[b..sentence_end].trim();
        let sentence_mask = &masked[b..sentence_end];
        if sentence_ends_with_question(sentence, sentence_mask) {
            run_start = i;
        } else {
            break;
        }
    }
    if run_start == 0 {
        return Some(start);
    }
    if run_start == boundaries.len() {
        return None;
    }
    Some(start + boundaries[run_start])
}

fn sentence_ends_with_question(sentence: &str, masked: &str) -> bool {
    let t = sentence.trim_end();
    if t.is_empty() {
        return false;
    }
    let rel = t.len() - 1;
    masked
        .as_bytes()
        .get(sentence.len().saturating_sub(sentence.trim_end().len()) + rel)
        == Some(&b'?')
        || t.ends_with('?') && !masked[..t.len()].contains('\u{0}')
}

fn sentence_starts(masked: &str) -> Vec<usize> {
    let bytes = masked.as_bytes();
    let mut starts = vec![leading_ws(masked)];
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'.' || b == b'!' || b == b'?' {
            if b == b'.' && is_abbreviation(masked, i) {
                i += 1;
                continue;
            }
            let mut j = i + 1;
            if j < bytes.len()
                && (bytes[j] == b' ' || bytes[j] == b'\t' || bytes[j] == b'\n' || bytes[j] == b'\r')
            {
                while j < bytes.len()
                    && (bytes[j] == b' '
                        || bytes[j] == b'\t'
                        || bytes[j] == b'\n'
                        || bytes[j] == b'\r')
                {
                    j += 1;
                }
                if j < bytes.len() && j != starts.last().copied().unwrap_or(usize::MAX) {
                    starts.push(j);
                }
            }
        }
        i += 1;
    }
    starts.sort_unstable();
    starts.dedup();
    starts.retain(|s| *s < masked.len());
    if starts.first().is_none_or(|s| *s != 0) && leading_ws(masked) == 0 {
        // keep
    }
    starts
}

fn leading_ws(s: &str) -> usize {
    s.len() - s.trim_start().len()
}

fn is_abbreviation(masked: &str, dot: usize) -> bool {
    const ABBREV: &[&str] = &["e.g.", "i.e.", "etc.", "vs."];
    let head = &masked[..dot + 1];
    ABBREV.iter().any(|a| head.ends_with(a))
}

/// Replace `?` inside inline code and URLs with a placeholder so the splitter
/// ignores them. Other characters stay put, so indexes still match `src`.
fn mask_protected(src: &str) -> String {
    let mut out: Vec<u8> = src.as_bytes().to_vec();
    mask_inline_code(&mut out);
    mask_urls(&mut out);
    String::from_utf8(out).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}

fn mask_inline_code(bytes: &mut [u8]) {
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'`' {
                if bytes[j] == b'?' {
                    bytes[j] = b' ';
                }
                j += 1;
            }
            if j < bytes.len() {
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
}

fn mask_urls(bytes: &mut [u8]) {
    let lower: Vec<u8> = bytes.iter().map(|b| b.to_ascii_lowercase()).collect();
    for key in [b"https://".as_slice(), b"http://".as_slice()] {
        let mut from = 0;
        while let Some(rel) = find_subslice(&lower[from..], key) {
            let start = from + rel;
            let mut end = start + key.len();
            while end < bytes.len() && !bytes[end].is_ascii_whitespace() {
                end += 1;
            }
            for slot in &mut bytes[start..end] {
                if *slot == b'?' {
                    *slot = b' ';
                }
            }
            from = end;
        }
    }
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn label_blocks(pieces: &mut [Piece], text: &str, narration: bool) {
    for piece in pieces.iter_mut() {
        if narration {
            if piece.kind != PieceKind::Code {
                piece.kind = PieceKind::Statement;
            }
            continue;
        }
        if piece.kind == PieceKind::Code {
            continue;
        }
        let body = text[piece.start..piece.end].trim();
        if piece_is_question(text, piece) {
            piece.kind = PieceKind::Question;
        } else if !piece.items.is_empty() && items_are_imperative(&piece.items) {
            piece.kind = PieceKind::Steps;
        } else if starts_caveat(body) {
            piece.kind = PieceKind::Caveat;
        } else {
            piece.kind = PieceKind::Statement;
        }
    }
}

fn piece_is_question(text: &str, piece: &Piece) -> bool {
    trailing_question_split(text, piece.start, piece.end).is_some_and(|at| at <= piece.start) || {
        let src = text[piece.start..piece.end].trim();
        let masked = mask_protected(src);
        let t = src.trim_end();
        !t.is_empty() && masked.trim_end().ends_with('?') && piece.items.is_empty()
    }
}

fn starts_caveat(body: &str) -> bool {
    let t = body.trim_start();
    t.starts_with("This will") || t.starts_with("Note") || t.starts_with("Be careful")
}

fn items_are_imperative(items: &[PieceItem]) -> bool {
    !items.is_empty() && items.iter().all(|item| is_imperative(&item.text))
}

fn is_imperative(text: &str) -> bool {
    let first = text.split_whitespace().next().unwrap_or("");
    let word = first.trim_matches(|c: char| !c.is_ascii_alphabetic() && c != '-');
    let lower = word.to_ascii_lowercase();
    IMPERATIVES.iter().any(|v| *v == lower)
}

const IMPERATIVES: &[&str] = &[
    "add", "apply", "avoid", "call", "change", "check", "convert", "create", "delete", "do",
    "don't", "edit", "fix", "install", "keep", "make", "open", "read", "re-run", "remove", "rerun",
    "return", "run", "set", "skip", "start", "stop", "update", "use", "write",
];

fn ranges_tile(text: &str, pieces: &[Piece]) -> bool {
    if text.is_empty() {
        return pieces.is_empty()
            || (pieces.len() == 1 && pieces[0].start == 0 && pieces[0].end == 0);
    }
    let mut covered = vec![false; text.len()];
    for piece in pieces {
        if piece.start > piece.end || piece.end > text.len() {
            return false;
        }
        if !text.is_char_boundary(piece.start) || !text.is_char_boundary(piece.end) {
            return false;
        }
        for slot in &mut covered[piece.start..piece.end] {
            if *slot {
                return false;
            }
            *slot = true;
        }
    }
    let mut i = 0;
    while i < text.len() {
        if covered[i] {
            i += 1;
            continue;
        }
        let ch = text[i..].chars().next().unwrap();
        let width = ch.len_utf8();
        if ch.is_whitespace() {
            i += width;
            continue;
        }
        let rest = &text[i..];
        if rest.starts_with("---") || rest.starts_with("***") || rest.starts_with("___") {
            let line_end = rest.find('\n').map(|n| i + n + 1).unwrap_or(text.len());
            let line = text[i..line_end].trim();
            if line.chars().all(|c| c == '-' || c == '*' || c == '_') && line.chars().count() >= 3 {
                i = line_end;
                continue;
            }
        }
        return false;
    }
    true
}

/// Cut one HTML index page into pieces. `Err` when the page is not that index.
///
/// A part counts only when an index link, the target id, and a single kind-word
/// class all match. The link is `N.0.0 [kind] Heading` and each `p` or `li`
/// starts `N.0.M`. A `pre` carries that number in `data-n`, not in the code.
/// The link is the heading. A missing or different `h2` is logged and ignored.
/// `start` and `end` are UTF-8 byte offsets of the part element.
#[allow(clippy::result_unit_err)]
pub fn cut_html_index(text: &str) -> Result<Vec<Piece>, ()> {
    let doc = scraper::Html::parse_fragment(text);
    let index = html_by_id(&doc, "index")?;
    let mut pieces = Vec::new();
    for anchor in index.descendants() {
        let Some(el) = scraper::ElementRef::wrap(anchor) else {
            continue;
        };
        if el.value().name() != "a" {
            continue;
        }
        let href = el.value().attr("href").ok_or(())?;
        let id = href
            .strip_prefix('#')
            .filter(|id| !id.is_empty())
            .ok_or(())?;
        let (major, kind, heading) = split_kind_heading(&el.text().collect::<String>()).ok_or(())?;
        let order = u32::try_from(pieces.len() + 1).map_err(|_| ())?;
        if major != order {
            return Err(());
        }
        let part = html_by_id(&doc, id)?;
        if !class_is_kind(part.value().attr("class").unwrap_or(""), kind) {
            return Err(());
        }
        warn_if_h2_disagrees(part, order, heading.as_deref());
        let (start, end) = element_span(text, id).ok_or(())?;
        let items = part_items(part, text, start, order)?;
        pieces.push(Piece {
            order,
            kind,
            heading,
            items,
            start,
            end,
        });
    }
    if pieces.is_empty() {
        return Err(());
    }
    Ok(pieces)
}

fn html_by_id<'a>(doc: &'a scraper::Html, id: &str) -> Result<scraper::ElementRef<'a>, ()> {
    doc.root_element()
        .descendants()
        .filter_map(scraper::ElementRef::wrap)
        .find(|el| el.value().attr("id") == Some(id))
        .ok_or(())
}

fn split_kind_heading(raw: &str) -> Option<(u32, PieceKind, Option<String>)> {
    let text = raw.trim();
    let (num, rest) = text.split_once(char::is_whitespace)?;
    let (major, minor, patch) = parse_outline(num)?;
    if minor != 0 || patch != 0 {
        return None;
    }
    let rest = rest.trim_start().strip_prefix('[')?;
    let (kind_raw, after) = rest.split_once(']')?;
    if kind_raw.contains('[') {
        return None;
    }
    let kind = PieceKind::from_label(kind_raw)?;
    let heading = after.trim();
    let heading = if heading.is_empty() {
        None
    } else {
        Some(heading.to_string())
    };
    Some((major, kind, heading))
}

/// `1.0.2` is major 1, minor 0, patch 2. Anything else is refused.
fn parse_outline(raw: &str) -> Option<(u32, u32, u32)> {
    let mut parts = raw.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    let patch: u32 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

fn expected_h2(order: u32, heading: Option<&str>) -> String {
    match heading.map(str::trim).filter(|h| !h.is_empty()) {
        Some(heading) => format!("{order}.0.0 {heading}"),
        None => format!("{order}.0.0"),
    }
}

/// `true` when the part has no `h2`, or the `h2` text is not `N.0.0 Heading`.
fn h2_disagrees(found: Option<&str>, order: u32, heading: Option<&str>) -> bool {
    let expected = expected_h2(order, heading);
    !matches!(found.map(str::trim), Some(text) if text == expected)
}

fn warn_if_h2_disagrees(part: scraper::ElementRef<'_>, order: u32, heading: Option<&str>) {
    let found = part
        .descendants()
        .filter_map(scraper::ElementRef::wrap)
        .find(|el| el.value().name() == "h2")
        .map(|el| el.text().collect::<String>());
    if h2_disagrees(found.as_deref(), order, heading) {
        tracing::warn!(
            order,
            expected = %expected_h2(order, heading),
            found = found.as_deref().unwrap_or(""),
            "html index h2 does not match the link; the link is the heading"
        );
    }
}

fn class_is_kind(class: &str, kind: PieceKind) -> bool {
    let mut tokens = class.split_whitespace();
    tokens.next() == Some(kind.as_label()) && tokens.next().is_none()
}

fn part_items(
    part: scraper::ElementRef<'_>,
    text: &str,
    part_start: usize,
    order: u32,
) -> Result<Vec<PieceItem>, ()> {
    let mut items = Vec::new();
    for node in part.descendants() {
        let Some(el) = scraper::ElementRef::wrap(node) else {
            continue;
        };
        let name = el.value().name();
        if !is_item_tag(name) || has_item_ancestor(el) || li_inside_index(el) {
            continue;
        }
        let raw: String = el.text().collect();
        if name != "pre" && raw.trim().is_empty() {
            continue;
        }
        let position = u32::try_from(items.len() + 1).map_err(|_| ())?;
        let body = if name == "pre" {
            let expected = format!("{order}.0.{position}");
            if el.value().attr("data-n") != Some(expected.as_str()) {
                return Err(());
            }
            raw.trim().to_string()
        } else {
            let body = strip_line_number(raw.trim(), order, position)?;
            if body.is_empty() {
                return Err(());
            }
            body
        };
        let (start, end) =
            item_span(text, part_start, position as usize).unwrap_or((part_start, part_start));
        items.push(PieceItem {
            position,
            text: body,
            start,
            end,
        });
    }
    Ok(items)
}

/// Drops a leading `N.0.M` and the one space or newline after it.
fn strip_line_number(body: &str, order: u32, position: u32) -> Result<String, ()> {
    let prefix = format!("{order}.0.{position}");
    let rest = body.trim_start().strip_prefix(&prefix).ok_or(())?;
    let rest = rest
        .strip_prefix(' ')
        .or_else(|| rest.strip_prefix('\n'))
        .ok_or(())?;
    Ok(rest.trim().to_string())
}

fn is_item_tag(name: &str) -> bool {
    matches!(name, "p" | "li" | "pre")
}

fn has_item_ancestor(el: scraper::ElementRef<'_>) -> bool {
    el.ancestors().any(|node| {
        scraper::ElementRef::wrap(node).is_some_and(|anc| is_item_tag(anc.value().name()))
    })
}

fn li_inside_index(el: scraper::ElementRef<'_>) -> bool {
    if el.value().name() != "li" {
        return false;
    }
    el.ancestors().any(|node| {
        scraper::ElementRef::wrap(node).and_then(|anc| anc.value().attr("id")) == Some("index")
    })
}

fn element_span(text: &str, id: &str) -> Option<(usize, usize)> {
    let attr_at = [format!("id=\"{id}\""), format!("id='{id}'")]
        .into_iter()
        .filter_map(|needle| text.find(&needle))
        .min()?;
    let start = text[..attr_at].rfind('<')?;
    let tag = tag_name(&text[start..])?;
    let end = start + matching_close(&text[start..], tag)?;
    Some((start, end))
}

fn item_span(text: &str, part_start: usize, nth: usize) -> Option<(usize, usize)> {
    let mut seen = 0usize;
    let mut i = part_start;
    let bytes = text.as_bytes();
    while i < text.len() {
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }
        if text[i..].starts_with("<!--") {
            i = text[i..]
                .find("-->")
                .map(|n| i + n + 3)
                .unwrap_or(text.len());
            continue;
        }
        let is_close = text[i + 1..].starts_with('/');
        if is_close {
            let gt = text[i..].find('>')?;
            i += gt + 1;
            continue;
        }
        let name = tag_name(&text[i..])?;
        let gt = text[i..].find('>')?;
        let self_close = text[i..i + gt].ends_with('/');
        if is_item_tag(name) && !self_close {
            seen += 1;
            let close_rel = matching_close(&text[i..], name)?;
            if seen == nth {
                return Some((i, i + close_rel));
            }
            i += close_rel;
            continue;
        }
        i += gt + 1;
    }
    None
}

fn tag_name(at_tag: &str) -> Option<&str> {
    let rest = at_tag
        .strip_prefix('<')?
        .strip_prefix('/')
        .unwrap_or(at_tag.strip_prefix('<')?);
    let end = rest
        .find(|c: char| c.is_ascii_whitespace() || c == '>' || c == '/')
        .unwrap_or(rest.len());
    let name = &rest[..end];
    if name.is_empty() { None } else { Some(name) }
}

fn matching_close(html: &str, tag: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut i = 0usize;
    while i < html.len() {
        if !html[i..].starts_with('<') {
            i += 1;
            continue;
        }
        if html[i..].starts_with("<!--") {
            i = html[i..]
                .find("-->")
                .map(|n| i + n + 3)
                .unwrap_or(html.len());
            continue;
        }
        let is_close = html[i + 1..].starts_with('/');
        let name = tag_name(&html[i..])?;
        let gt = html[i..].find('>')?;
        if name.eq_ignore_ascii_case(tag) {
            if is_close {
                depth -= 1;
                if depth == 0 {
                    return Some(i + gt + 1);
                }
            } else if html[i..i + gt].ends_with('/') {
                if depth == 0 {
                    return Some(i + gt + 1);
                }
            } else {
                depth += 1;
            }
        }
        i += gt + 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

    fn piece(start: usize, end: usize) -> Piece {
        Piece {
            order: 0,
            kind: PieceKind::Statement,
            heading: None,
            items: Vec::new(),
            start,
            end,
        }
    }

    #[test]
    fn html_index_requires_outline_numbers() {
        let html = r##"<div id="index"><a href="#part1">1.0.0 [statement] Overview</a></div>
<div id="part1" class="statement"><p>1.0.1 One sentence.</p></div>"##;
        let pieces = cut_html_index(html).expect("cut");
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].order, 1);
        assert_eq!(pieces[0].kind, PieceKind::Statement);
        assert_eq!(pieces[0].heading.as_deref(), Some("Overview"));
        assert_eq!(pieces[0].items[0].text, "One sentence.");
        assert_eq!(pieces[0].items[0].position, 1);

        let bare = r##"<div id="index"><a href="#part1">[statement] Overview</a></div>
<div id="part1" class="statement"><p>One sentence.</p></div>"##;
        assert!(cut_html_index(bare).is_err());

        let swapped = r##"<div id="index"><a href="#part1">2.0.0 [statement] Overview</a></div>
<div id="part1" class="statement"><p>2.0.1 One sentence.</p></div>"##;
        assert!(cut_html_index(swapped).is_err());
    }

    #[test]
    fn html_index_cuts_a_linked_part() {
        let html = r##"<div id="index"><a href="#part1">1.0.0 [statement] Overview</a></div>
<div id="part1" class="statement"><p>1.0.1 One sentence.</p></div>"##;
        let pieces = cut_html_index(html).expect("cut");
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].kind, PieceKind::Statement);
        assert_eq!(pieces[0].heading.as_deref(), Some("Overview"));
        assert_eq!(pieces[0].items[0].text, "One sentence.");
        assert_eq!(pieces[0].items[0].position, 1);
    }

    #[test]
    fn html_index_keeps_pre_code_as_one_item() {
        let html = r##"<nav id="index"><ul><li><a href="#c">1.0.0 [code] Sample</a></li></ul></nav>
<section id="c" class="code"><h2>1.0.0 Sample</h2><pre data-n="1.0.1"><code>let x = 1;</code></pre></section>"##;
        let pieces = cut_html_index(html).expect("cut");
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].kind, PieceKind::Code);
        assert_eq!(pieces[0].items.len(), 1);
        assert_eq!(pieces[0].items[0].text, "let x = 1;");
        assert!(!pieces[0].items[0].text.contains("1.0.1"));
        assert_eq!(pieces[0].items[0].position, 1);
    }

    #[test]
    fn html_index_refuses_a_number_inside_code() {
        let html = r##"<nav id="index"><ul><li><a href="#c">1.0.0 [code] Sample</a></li></ul></nav>
<section id="c" class="code"><h2>1.0.0 Sample</h2><pre><code>1.0.1 let x = 1;</code></pre></section>"##;
        assert!(cut_html_index(html).is_err());
    }

    #[test]
    fn html_index_keeps_the_link_when_the_h2_disagrees() {
        let html = r##"<nav id="index"><ul><li><a href="#part1">1.0.0 [statement] Overview</a></li></ul></nav>
<section id="part1" class="statement"><h2>9.9.9 Other</h2><p>1.0.1 One sentence.</p></section>"##;
        let pieces = cut_html_index(html).expect("cut");
        assert_eq!(pieces[0].heading.as_deref(), Some("Overview"));
        assert!(h2_disagrees(Some("9.9.9 Other"), 1, Some("Overview")));
        assert!(!h2_disagrees(Some("1.0.0 Overview"), 1, Some("Overview")));
        assert!(h2_disagrees(None, 1, Some("Overview")));
    }

    #[test]
    fn html_index_cuts_each_kind() {
        let html = r##"<nav id="index"><ul>
<li><a href="#s">1.0.0 [statement] Overview</a></li>
<li><a href="#o">2.0.0 [options] Pick one</a></li>
<li><a href="#t">3.0.0 [steps] Run it</a></li>
<li><a href="#i">4.0.0 [instructions] How</a></li>
<li><a href="#e">5.0.0 [example] Sample</a></li>
<li><a href="#v">6.0.0 [caveat] Limit</a></li>
<li><a href="#c">7.0.0 [code] Patch</a></li>
<li><a href="#q">8.0.0 [question] Ask</a></li>
</ul></nav>
<section id="s" class="statement"><h2>1.0.0 Overview</h2><p>1.0.1 One fact.</p></section>
<section id="o" class="options"><h2>2.0.0 Pick one</h2><ul><li>2.0.1 Use the cache</li><li>2.0.2 Skip the cache</li></ul></section>
<section id="t" class="steps"><h2>3.0.0 Run it</h2><ul><li>3.0.1 Build</li><li>3.0.2 Ship</li></ul></section>
<section id="i" class="instructions"><h2>4.0.0 How</h2><p>4.0.1 Pass the token in the header.</p></section>
<section id="e" class="example"><h2>5.0.0 Sample</h2><figure><p>5.0.1 A request looks like <code>GET /health</code>.</p></figure></section>
<section id="v" class="caveat"><h2>6.0.0 Limit</h2><p>6.0.1 This drops old tokens.</p></section>
<section id="c" class="code"><h2>7.0.0 Patch</h2><pre data-n="7.0.1"><code>let x = 1;</code></pre></section>
<section id="q" class="question"><h2>8.0.0 Ask</h2><p>8.0.1 Should I apply the patch?</p></section>"##;
        let pieces = cut_html_index(html).expect("cut");
        assert_eq!(pieces.len(), 8);
        assert_eq!(pieces[0].items[0].text, "One fact.");
        assert_eq!(pieces[1].kind, PieceKind::Options);
        assert_eq!(pieces[1].items[1].text, "Skip the cache");
        assert_eq!(pieces[2].kind, PieceKind::Steps);
        assert_eq!(pieces[2].items[0].text, "Build");
        assert_eq!(pieces[3].kind, PieceKind::Instructions);
        assert_eq!(pieces[4].kind, PieceKind::Example);
        assert_eq!(pieces[4].items[0].text, "A request looks like GET /health.");
        assert_eq!(pieces[5].kind, PieceKind::Caveat);
        assert_eq!(pieces[5].items[0].text, "This drops old tokens.");
        assert_eq!(pieces[6].kind, PieceKind::Code);
        assert_eq!(pieces[6].items[0].text, "let x = 1;");
        assert_eq!(pieces[7].kind, PieceKind::Question);
        assert_eq!(pieces[7].items[0].text, "Should I apply the patch?");
    }

    #[test]
    fn html_index_rejects_a_dangling_href() {
        let html = r##"<div id="index"><a href="#missing">1.0.0 [statement] Gone</a></div>"##;
        assert!(cut_html_index(html).is_err());
    }

    #[test]
    fn html_index_rejects_a_prefixed_kind_class() {
        let html = r##"<div id="index"><a href="#part1">1.0.0 [statement] Overview</a></div>
<div id="part1" class="kind-statement"><p>1.0.1 One sentence.</p></div>"##;
        assert!(cut_html_index(html).is_err());
    }

    #[test]
    fn html_index_items_are_part_lines_not_index_entries() {
        let html = r##"<div id="index"><ul><li><a href="#part1">1.0.0 [steps] Deploy</a></li></ul></div>
<div id="part1" class="steps"><ul><li>1.0.1 Build</li><li>1.0.2 Ship</li></ul></div>"##;
        let pieces = cut_html_index(html).expect("cut");
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].kind, PieceKind::Steps);
        assert_eq!(pieces[0].heading.as_deref(), Some("Deploy"));
        assert_eq!(pieces[0].items.len(), 2);
        assert_eq!(pieces[0].items[0].text, "Build");
        assert_eq!(pieces[0].items[0].position, 1);
        assert_eq!(pieces[0].items[1].text, "Ship");
        assert_eq!(pieces[0].items[1].position, 2);
    }

    #[test]
    fn html_index_rejects_markdown_without_an_index() {
        let markdown = "# Hello\n\nA paragraph.\n";
        assert!(cut_html_index(markdown).is_err());
    }

    #[test]
    fn cut_html_index_leaves_structure_segment_unchanged() {
        let text = "Spark reads the file, shuffles, then writes.";
        let before = structure_segment(text, false);
        assert!(cut_html_index(text).is_err());
        assert_eq!(structure_segment(text, false), before);
    }

    #[test]
    fn a_whitespace_gap_still_tiles() {
        let text = "Hello.\n\nWorld.";
        let blocks = vec![piece(0, 6), piece(8, text.len())];
        assert!(ranges_tile(text, &blocks));
    }

    #[test]
    fn dropping_one_character_falls_back_to_one_statement() {
        let text = "Hello.\n\nWorld.";
        let blocks = vec![piece(0, 5), piece(8, text.len())];
        let got = commit_blocks(text, blocks);
        assert!(!got.tiled);
        assert_eq!(got.pieces.len(), 1);
        assert_eq!(got.pieces[0].kind, PieceKind::Statement);
        assert_eq!(got.pieces[0].start, 0);
        assert_eq!(got.pieces[0].end, text.len());
        assert!(got.parsed);
    }

    #[test]
    fn shifting_an_offset_by_one_byte_falls_back_to_one_statement() {
        let text = "Hello.\n\nWorld.";
        let blocks = vec![piece(0, 6), piece(9, text.len())];
        let got = commit_blocks(text, blocks);
        assert!(!got.tiled);
        assert_eq!(got.pieces.len(), 1);
        assert_eq!(got.pieces[0].kind, PieceKind::Statement);
        assert_eq!(got.pieces[0].end, text.len());
    }

    #[test]
    fn a_one_byte_shift_inside_a_character_falls_back_to_one_statement() {
        let text = "Ship 🚀 it.";
        let rocket = text.find('🚀').unwrap();
        let blocks = vec![
            piece(0, rocket + 1),
            piece(rocket + '🚀'.len_utf8(), text.len()),
        ];
        let got = commit_blocks(text, blocks);
        assert!(!got.tiled);
        assert_eq!(got.pieces.len(), 1);
        assert_eq!(got.pieces[0].end, text.len());
    }

    fn kinds(text: &str, narration: bool) -> Vec<PieceKind> {
        structure_segment(text, narration)
            .pieces
            .into_iter()
            .map(|p| p.kind)
            .collect()
    }

    #[test]
    fn over_cap_is_one_statement_and_is_not_parsed() {
        let text = "a".repeat(MAX_SEGMENT_CHARS + 1);
        let got = structure_segment(&text, false);
        assert!(!got.parsed);
        assert!(got.tiled);
        assert_eq!(got.pieces.len(), 1);
        assert_eq!(got.pieces[0].kind, PieceKind::Statement);
        assert_eq!(got.pieces[0].end, text.len());
    }

    #[test]
    fn trailing_question_splits_off_the_statement() {
        let text = "Fixed the expiry check. Want me to push it?";
        let got = structure_segment(text, false);
        assert!(got.tiled, "untiled: {text:?}");
        assert_eq!(got.pieces.len(), 2);
        assert_eq!(got.pieces[0].kind, PieceKind::Statement);
        assert!(got.pieces[0].items.is_empty());
        assert!(text[got.pieces[0].start..got.pieces[0].end].contains("Fixed the expiry check."));
        assert_eq!(got.pieces[1].kind, PieceKind::Question);
        assert_eq!(got.pieces[1].order, 2);
        assert!(text[got.pieces[1].start..got.pieces[1].end].contains("Want me to push it?"));
    }

    #[test]
    fn trailing_question_run_stays_one_piece() {
        let text = "Want me to push it? Or open a PR?";
        let got = structure_segment(text, false);
        assert!(got.tiled);
        assert_eq!(got.pieces.len(), 1);
        assert_eq!(got.pieces[0].kind, PieceKind::Question);
    }

    #[test]
    fn question_mark_inside_inline_code_does_not_split() {
        let text = "See `foo?` in the docs. Then restart.";
        let got = structure_segment(text, false);
        assert!(got.tiled);
        assert_eq!(kinds(text, false), vec![PieceKind::Statement]);
    }

    #[test]
    fn abbreviation_does_not_end_the_sentence() {
        let text = "See e.g. the tests. Want me to run them?";
        let got = structure_segment(text, false);
        assert!(got.tiled);
        assert_eq!(got.pieces.len(), 2);
        assert!(text[got.pieces[0].start..got.pieces[0].end].contains("e.g. the tests."));
        assert_eq!(got.pieces[1].kind, PieceKind::Question);
    }

    #[test]
    fn file_extension_and_version_are_not_sentence_ends() {
        let text = "The bug is in auth.rs at v0.3. Want me to patch it?";
        let got = structure_segment(text, false);
        assert!(got.tiled);
        assert_eq!(got.pieces.len(), 2);
        assert!(text[got.pieces[0].start..got.pieces[0].end].contains("auth.rs"));
        assert!(text[got.pieces[0].start..got.pieces[0].end].contains("v0.3."));
    }

    #[test]
    fn question_on_its_own_line_splits() {
        let text = "Fixed the expiry check.\nWant me to push it?";
        let got = structure_segment(text, false);
        assert!(got.tiled);
        assert_eq!(got.pieces.len(), 2);
        assert_eq!(got.pieces[1].kind, PieceKind::Question);
    }

    #[test]
    fn colon_lead_in_becomes_the_list_heading() {
        let text = "To fix it:\n1. Convert `exp` to milliseconds.\n2. Re-run the auth tests.\n";
        let got = structure_segment(text, false);
        assert!(got.tiled, "{text:?} pieces={:?}", got.pieces);
        assert_eq!(got.pieces.len(), 1);
        assert_eq!(got.pieces[0].kind, PieceKind::Steps);
        assert_eq!(got.pieces[0].heading.as_deref(), Some("To fix it"));
        assert_eq!(got.pieces[0].items.len(), 2);
        assert_eq!(got.pieces[0].items[0].position, 1);
        assert!(got.pieces[0].items[0].text.contains("Convert"));
        assert_eq!(got.pieces[0].items[1].position, 2);
        assert!(got.pieces[0].items[1].text.contains("Re-run"));
    }

    #[test]
    fn markdown_heading_attaches_to_the_following_paragraph() {
        let text = "# Why\n\nThe comparison uses seconds.\n";
        let got = structure_segment(text, false);
        assert!(got.tiled, "{:?}", got.pieces);
        assert_eq!(got.pieces.len(), 1);
        assert_eq!(got.pieces[0].heading.as_deref(), Some("Why"));
        assert_eq!(got.pieces[0].kind, PieceKind::Statement);
        assert!(text[got.pieces[0].start..got.pieces[0].end].contains("# Why"));
    }

    #[test]
    fn fence_is_code_and_a_following_warning_is_a_caveat() {
        let text = "```rust\nif claims.exp * 1000 < now_ms() {\n```\n\nThis will invalidate tokens issued under the old bug.\n";
        let got = structure_segment(text, false);
        assert!(got.tiled, "{:?}", got.pieces);
        assert_eq!(
            got.pieces.iter().map(|p| p.kind).collect::<Vec<_>>(),
            vec![PieceKind::Code, PieceKind::Caveat]
        );
    }

    #[test]
    fn baseline_marks_this_will_print_as_a_caveat() {
        let text = "This will print `hi`.";
        assert_eq!(kinds(text, false), vec![PieceKind::Caveat]);
    }

    #[test]
    fn baseline_marks_imperative_options_as_steps() {
        let text = "- Use X\n- Use Y\n";
        let got = structure_segment(text, false);
        assert!(got.tiled);
        assert_eq!(got.pieces.len(), 1);
        assert_eq!(got.pieces[0].kind, PieceKind::Steps);
        assert_eq!(got.pieces[0].items.len(), 2);
        assert_eq!(got.pieces[0].order, 1);
    }

    #[test]
    fn three_paragraphs_are_three_pieces() {
        let text = "One.\n\nTwo.\n\nThree.\n";
        let got = structure_segment(text, false);
        assert!(got.tiled);
        assert_eq!(got.pieces.len(), 3);
        assert!(got.pieces.iter().all(|p| p.kind == PieceKind::Statement));
        assert_eq!(
            got.pieces.iter().map(|p| p.order).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn narration_parses_a_plan_but_labels_only_statement_and_code() {
        let text =
            "Let me check.\n\n1. Read auth.rs\n2. Run the tests\n\n```rust\nlet x = 1;\n```\n";
        let got = structure_segment(text, true);
        assert!(got.tiled, "{:?}", got.pieces);
        assert!(got.pieces.iter().any(|p| !p.items.is_empty()));
        assert!(
            got.pieces
                .iter()
                .all(|p| { p.kind == PieceKind::Statement || p.kind == PieceKind::Code })
        );
        assert!(got.pieces.iter().any(|p| p.kind == PieceKind::Code));
    }

    #[test]
    fn bullet_items_are_numbered_by_position() {
        let text = "- Bananas\n- Apples\n";
        let got = structure_segment(text, false);
        assert!(got.tiled);
        assert_eq!(got.pieces[0].kind, PieceKind::Statement);
        assert_eq!(
            got.pieces[0]
                .items
                .iter()
                .map(|i| i.position)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!(!got.pieces[0].items[0].text.starts_with("1."));
    }

    #[test]
    fn horizontal_rule_is_not_a_piece() {
        let text = "Before.\n\n---\n\nAfter.\n";
        let got = structure_segment(text, false);
        assert!(got.tiled, "{:?}", got.pieces);
        assert_eq!(got.pieces.len(), 2);
        assert!(
            got.pieces
                .iter()
                .all(|p| !text[p.start..p.end].contains("---"))
        );
    }

    #[test]
    fn auth_reply_tiles_into_the_agreed_shapes() {
        let text = "\
The failure is in `verify_token`: it compares expiry in seconds against milliseconds.

To fix it:
1. Convert `exp` to milliseconds before comparing.
2. Re-run the auth tests.

```rust
if claims.exp * 1000 < now_ms() {
```

This will invalidate tokens issued under the old bug.

Should I apply the patch?
";
        let got = structure_segment(text, false);
        assert!(got.tiled, "{:#?}", got.pieces);
        assert_eq!(
            got.pieces.iter().map(|p| p.kind).collect::<Vec<_>>(),
            vec![
                PieceKind::Statement,
                PieceKind::Steps,
                PieceKind::Code,
                PieceKind::Caveat,
                PieceKind::Question,
            ]
        );
        assert_eq!(got.pieces[1].heading.as_deref(), Some("To fix it"));
        assert_eq!(got.pieces[1].items.len(), 2);
        assert_eq!(got.pieces[1].items[0].position, 1);
    }

    #[test]
    fn question_mark_inside_a_url_does_not_split() {
        let text = "See https://example.com/a?b=1 for the note. Then restart.";
        assert_eq!(kinds(text, false), vec![PieceKind::Statement]);
    }

    #[test]
    fn bold_note_lead_in_heads_the_list() {
        let text = "**Note:**\n- Use the old token\n- Use the new token\n";
        let got = structure_segment(text, false);
        assert!(got.tiled, "{:?}", got.pieces);
        assert_eq!(
            got.pieces.len(),
            1,
            "{:?}",
            got.pieces
                .iter()
                .map(|p| &text[p.start..p.end])
                .collect::<Vec<_>>()
        );
        assert_eq!(got.pieces[0].heading.as_deref(), Some("Note"));
    }

    #[test]
    fn blockquote_containing_a_table_stays_one_piece() {
        let text = "> ## Problem\n>\n> | File | Risk |\n> | --- | --- |\n> | a | b |\n";
        let got = structure_segment(text, false);
        assert!(got.tiled, "{:#?}", got.pieces);
        assert_eq!(got.pieces.len(), 1);
    }

    #[test]
    fn blockquote_containing_a_heading_does_not_overlap() {
        let text = "> ## Problem\n>\n> Two functions.\n";
        let got = structure_segment(text, false);
        assert!(
            got.tiled,
            "{:#?}",
            got.pieces
                .iter()
                .map(|p| (p.kind, p.start, p.end, &text[p.start..p.end]))
                .collect::<Vec<_>>()
        );
    }

    /// Live Cursor transcripts under `~/.cursor/projects`. They are not in the repo.
    ///
    /// `cargo test` skips this. A machine without that directory stays green, including
    /// `cargo test -- --ignored`. Run it locally with:
    ///
    /// ```bash
    /// cargo test -p graphirm-agent --lib cursor_transcripts_tile_when_present -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "reads ~/.cursor/projects; local only, run with --ignored"]
    fn cursor_transcripts_tile_when_present() {
        let Some(root) = cursor_projects_dir() else {
            skip_no_transcripts();
            return;
        };
        let mut files = Vec::new();
        let Ok(entries) = std::fs::read_dir(&root) else {
            skip_no_transcripts();
            return;
        };
        for project in entries.flatten() {
            let transcripts = project.path().join("agent-transcripts");
            let Ok(sessions) = std::fs::read_dir(&transcripts) else {
                continue;
            };
            for session in sessions.flatten() {
                let id = session.file_name();
                let path = session.path().join(&id).with_extension("jsonl");
                if path.is_file() {
                    files.push(path);
                }
            }
        }
        if files.is_empty() {
            skip_no_transcripts();
            return;
        }
        let mut checked = 0usize;
        let mut over_cap = 0usize;
        let mut failures: Vec<String> = Vec::new();
        for path in &files {
            let Ok(raw) = std::fs::read_to_string(path) else {
                continue;
            };
            for (line_no, line) in raw.lines().enumerate() {
                let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                    continue;
                };
                let Some(text) = assistant_text(&value) else {
                    continue;
                };
                if text.trim().is_empty() {
                    continue;
                }
                checked += 1;
                if text.chars().count() > MAX_SEGMENT_CHARS {
                    over_cap += 1;
                    continue;
                }
                let mut blocks = parse_blocks(&text);
                attach_headings(&mut blocks, &text);
                let mut blocks = split_questions(blocks, &text);
                label_blocks(&mut blocks, &text, false);
                if !ranges_tile(&text, &blocks) {
                    failures.push(format!(
                        "{}:{} {}",
                        path.display(),
                        line_no + 1,
                        gap_preview(&text, &blocks)
                    ));
                    if failures.len() >= 8 {
                        break;
                    }
                }
            }
            if failures.len() >= 8 {
                break;
            }
        }
        if checked == 0 {
            skip_no_transcripts();
            return;
        }
        eprintln!("cursor assistant texts {checked}, over the 16000-character cap {over_cap}");
        assert!(
            failures.is_empty(),
            "untiled segments ({} checked): {}",
            checked,
            failures.join(", ")
        );
    }

    fn skip_no_transcripts() {
        eprintln!("skipped: no transcripts found");
    }

    fn cursor_projects_dir() -> Option<std::path::PathBuf> {
        let home = std::env::var_os("HOME")?;
        let root = std::path::PathBuf::from(home).join(".cursor/projects");
        root.is_dir().then_some(root)
    }

    fn gap_preview(text: &str, pieces: &[Piece]) -> String {
        for piece in pieces {
            if piece.start > piece.end || piece.end > text.len() {
                return format!(
                    "bad-range {}..{} len {}",
                    piece.start,
                    piece.end,
                    text.len()
                );
            }
            if !text.is_char_boundary(piece.start) || !text.is_char_boundary(piece.end) {
                return format!("bad-boundary {}..{}", piece.start, piece.end);
            }
        }
        let mut covered = vec![0u8; text.len()];
        for (n, piece) in pieces.iter().enumerate() {
            for slot in &mut covered[piece.start..piece.end] {
                if *slot != 0 {
                    let prev =
                        text[piece.start..piece.end.min(piece.start + 40)].replace('\n', "\\n");
                    return format!(
                        "overlap piece {} at {} prev-mark {slot} {:?}",
                        n + 1,
                        piece.start,
                        prev
                    );
                }
                *slot = 1;
            }
        }
        let mut i = 0;
        while i < text.len() {
            if covered[i] == 1 {
                i += 1;
                continue;
            }
            let ch = text[i..].chars().next().unwrap();
            if ch.is_whitespace() {
                i += ch.len_utf8();
                continue;
            }
            let end = (i + 48).min(text.len());
            let end = (end..=text.len())
                .find(|n| text.is_char_boundary(*n))
                .unwrap_or(text.len());
            let snippet: String = text[i..end].chars().take(24).collect();
            return format!("byte {i} {:?}", snippet);
        }
        format!("no-gap pieces={}", pieces.len())
    }

    fn assistant_text(value: &serde_json::Value) -> Option<String> {
        let message = value.get("message").unwrap_or(value);
        let role = value
            .get("role")
            .or_else(|| message.get("role"))
            .and_then(|r| r.as_str())?;
        if role != "assistant" {
            return None;
        }
        let content = message.get("content")?;
        let mut parts = Vec::new();
        match content {
            serde_json::Value::String(s) => parts.push(s.clone()),
            serde_json::Value::Array(items) => {
                for item in items {
                    if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                        if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                            parts.push(text.to_string());
                        }
                    }
                }
            }
            _ => return None,
        }
        let text = parts.join("\n");
        if text.is_empty() { None } else { Some(text) }
    }

    #[test]
    fn cursor_subset_snapshot_locks_parser_and_baseline_versions() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pieces");
        let raw = std::fs::read_to_string(dir.join("cursor-subset.jsonl")).expect("cursor subset");
        let snapshot: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("cursor-subset.snapshot.json")).expect("snapshot"),
        )
        .expect("snapshot json");
        let mut boundaries = String::new();
        let mut kinds = String::new();
        let mut count = 0usize;
        for line in raw.lines() {
            if line.is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(line).expect("subset line");
            let text = value
                .get("text")
                .and_then(|text| text.as_str())
                .expect("subset text");
            let got = structure_segment(text, false);
            count += 1;
            boundaries.push_str(&format!("parsed={} tiled={}\n", got.parsed, got.tiled));
            for piece in &got.pieces {
                let heading = piece.heading.as_deref().unwrap_or("");
                boundaries.push_str(&format!(
                    "{}\t{}\t{}\t{}\n",
                    piece.start,
                    piece.end,
                    heading.len(),
                    heading
                ));
                kinds.push_str(piece.kind.as_label());
                kinds.push('\n');
            }
            boundaries.push('\n');
            kinds.push('\n');
        }
        let expected_texts = snapshot["texts"].as_u64().expect("texts") as usize;
        assert_eq!(
            count, expected_texts,
            "cursor subset has {count} texts and the snapshot lists {expected_texts}"
        );
        let boundary_hash = hex_sha256(&boundaries);
        let kind_hash = hex_sha256(&kinds);
        let parser_version = snapshot["parser_version"].as_str().expect("parser_version");
        let baseline_version = snapshot["baseline_version"]
            .as_str()
            .expect("baseline_version");
        assert_eq!(
            parser_version, PIECE_PARSER_VERSION,
            "snapshot parser_version is {parser_version} and PIECE_PARSER_VERSION is {PIECE_PARSER_VERSION}. Set them to the same value."
        );
        assert_eq!(
            baseline_version, PIECE_BASELINE_VERSION,
            "snapshot baseline_version is {baseline_version} and PIECE_BASELINE_VERSION is {PIECE_BASELINE_VERSION}. Set them to the same value."
        );
        let mut problems = Vec::new();
        if snapshot["boundaries_sha256"].as_str() != Some(boundary_hash.as_str()) {
            problems.push(format!(
                "block boundaries or headings no longer match the snapshot. Set boundaries_sha256 to {boundary_hash} in tests/fixtures/pieces/cursor-subset.snapshot.json. Bump PIECE_PARSER_VERSION (now {PIECE_PARSER_VERSION}) and the snapshot parser_version if the parser rules changed."
            ));
        }
        if snapshot["kinds_sha256"].as_str() != Some(kind_hash.as_str()) {
            problems.push(format!(
                "baseline kinds no longer match the snapshot. Set kinds_sha256 to {kind_hash} in tests/fixtures/pieces/cursor-subset.snapshot.json. Bump PIECE_BASELINE_VERSION (now {PIECE_BASELINE_VERSION}) and the snapshot baseline_version if the kind rules changed."
            ));
        }
        assert!(
            problems.is_empty(),
            "cursor subset snapshot is stale:\n{}",
            problems.join("\n")
        );
    }

    fn hex_sha256(text: &str) -> String {
        let digest = sha2::Sha256::digest(text.as_bytes());
        let mut hex = String::with_capacity(digest.len() * 2);
        for byte in digest {
            hex.push(char::from(b"0123456789abcdef"[usize::from(byte >> 4)]));
            hex.push(char::from(b"0123456789abcdef"[usize::from(byte & 0xf)]));
        }
        hex
    }

    #[test]
    fn emoji_offsets_are_char_boundaries() {
        let text = "Ship it 🚀. Want me to push?";
        let got = structure_segment(text, false);
        assert!(got.tiled);
        for piece in &got.pieces {
            assert!(text.is_char_boundary(piece.start));
            assert!(text.is_char_boundary(piece.end));
            let _ = &text[piece.start..piece.end];
        }
    }
}
