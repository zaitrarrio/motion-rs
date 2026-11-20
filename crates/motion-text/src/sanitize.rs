//! `kimodo/sanitize.py`: prompts are normalised before they are encoded or looked up.

/// `sanitize_text(text, paragraph=True)`.
pub fn sanitize_text(text: &str) -> String {
    let t = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.is_empty() {
        return t;
    }
    let chars: Vec<char> = t.chars().collect();
    let start = chars.iter().position(|c| c.is_alphanumeric()).unwrap_or(chars.len() - 1);
    let mut s: String = chars[start..].iter().collect();
    s = capitalize(&s);
    let finals = ".!?\"])'";
    let cs: Vec<char> = s.chars().collect();
    let mut end = cs.len() - 1;
    for i in (0..cs.len()).rev() {
        if !cs[i].is_alphanumeric() && !finals.contains(cs[i]) {
            continue;
        }
        end = i;
        break;
    }
    s = cs[..=end].iter().collect();
    if !s.ends_with(['.', '!', '?']) {
        s.push('.');
    }
    for brk in ['.', '!', '?'] {
        let parts: Vec<String> = s
            .split(brk)
            .map(|x| {
                let y = x.trim();
                let mut c = y.chars();
                match c.next() {
                    Some(f) => f.to_uppercase().chain(c).collect(),
                    None => String::new(),
                }
            })
            .collect();
        s = parts.join(&format!("{brk} ")).trim().to_string();
    }
    s
}

/// Python's `str.capitalize`: first character upper, the rest lower.
fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c.flat_map(|x| x.to_lowercase())).collect(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::sanitize_text;

    /// The examples of `sanitize.py`'s `__main__`, with the outputs the Python prints.
    #[test]
    fn matches_upstream_examples() {
        let cases = [
            (" A person is    walking.", "A person is walking."),
            ("someone go forward", "Someone go forward."),
            ("jump", "Jump."),
            ("jumping!", "Jumping!"),
            ("jumping)", "Jumping)."),
            ("-go", "Go."),
            ("blocasdji  -----", "Blocasdji."),
            ("", ""),
            ("walk. then RUN", "Walk. Then run."),
        ];
        for (i, o) in cases {
            assert_eq!(sanitize_text(i), o, "{i:?}");
        }
    }
}
