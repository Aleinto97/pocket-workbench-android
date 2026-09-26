#[derive(Clone, Debug)]
pub enum ClassItem {
    Ch(char),
    Range(char, char),
    Letter,
    Number,
    Space,
    NotR,
    NotN,
}

#[derive(Clone, Debug)]
pub enum Node {
    Char(char),
    Class { neg: bool, items: Vec<ClassItem> },
    Any,
    Group(Vec<Vec<Node>>),
    Repeat { node: Box<Node>, min: usize, max: Option<usize> },
    Look { neg: bool, group: Vec<Vec<Node>> },
}

pub struct Regex {
    pub alts: Vec<Vec<Node>>,
}

struct Parser {
    c: Vec<char>,
    o: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.c.get(self.o).copied()
    }
    fn next(&mut self) -> Option<char> {
        let v = self.peek();
        if v.is_some() {
            self.o += 1;
        }
        v
    }
    fn parse_alts(&mut self, in_group: bool) -> Vec<Vec<Node>> {
        let mut alts = Vec::new();
        let mut seq = Vec::new();
        loop {
            match self.peek() {
                None => break,
                Some(')') if in_group => break,
                Some('|') => {
                    self.next();
                    alts.push(seq);
                    seq = Vec::new();
                }
                _ => {
                    if let Some(n) = self.parse_atom() {
                        seq.push(n);
                    }
                }
            }
        }
        alts.push(seq);
        alts
    }
    fn parse_atom(&mut self) -> Option<Node> {
        let ch = self.next()?;
        let base = match ch {
            '(' => {
                if self.peek() == Some('?') {
                    self.next();
                    match self.peek() {
                        Some(':') => {
                            self.next();
                            let alts = self.parse_alts(true);
                            self.expect(')');
                            Node::Group(alts)
                        }
                        Some('!') => {
                            self.next();
                            let alts = self.parse_alts(true);
                            self.expect(')');
                            Node::Look { neg: true, group: alts }
                        }
                        Some('=') => {
                            self.next();
                            let alts = self.parse_alts(true);
                            self.expect(')');
                            Node::Look { neg: false, group: alts }
                        }
                        _ => {
                            let alts = self.parse_alts(true);
                            self.expect(')');
                            Node::Group(alts)
                        }
                    }
                } else {
                    let alts = self.parse_alts(true);
                    self.expect(')');
                    Node::Group(alts)
                }
            }
            '[' => self.parse_class(),
            '.' => Node::Any,
            '\\' => {
                let e = self.next()?;
                match e {
                    'p' => {
                        if self.peek() == Some('{') {
                            self.next();
                            let mut name = String::new();
                            while let Some(c) = self.next() {
                                if c == '}' {
                                    break;
                                }
                                name.push(c);
                            }
                            match name.as_str() {
                                "L" | "Lu" | "Ll" | "Lt" | "Lm" | "Lo" => Node::Class {
                                    neg: false,
                                    items: vec![ClassItem::Letter],
                                },
                                "N" => Node::Class { neg: false, items: vec![ClassItem::Number] },
                                "P" | "S" | "M" => Node::Class { neg: false, items: vec![] },
                                _ => Node::Class { neg: false, items: vec![] },
                            }
                        } else {
                            Node::Char('p')
                        }
                    }
                    's' => Node::Class { neg: false, items: vec![ClassItem::Space] },
                    'S' => Node::Class { neg: true, items: vec![ClassItem::Space] },
                    'd' => Node::Class {
                        neg: false,
                        items: vec![ClassItem::Range('0', '9')],
                    },
                    'w' => Node::Class {
                        neg: false,
                        items: vec![
                            ClassItem::Range('a', 'z'),
                            ClassItem::Range('A', 'Z'),
                            ClassItem::Range('0', '9'),
                            ClassItem::Ch('_'),
                        ],
                    },
                    'r' => Node::Char('\r'),
                    'n' => Node::Char('\n'),
                    't' => Node::Char('\t'),
                    other => Node::Char(other),
                }
            }
            c => Node::Char(c),
        };
        Some(self.parse_quant(base))
    }
    fn parse_quant(&mut self, base: Node) -> Node {
        let (min, max) = match self.peek() {
            Some('*') => {
                self.next();
                (0, None)
            }
            Some('+') => {
                self.next();
                (1, None)
            }
            Some('?') => {
                self.next();
                (0, Some(1))
            }
            Some('{') => {
                let save = self.o;
                self.next();
                let mut a = String::new();
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        a.push(c);
                        self.next();
                    } else {
                        break;
                    }
                }
                let mut b = String::new();
                let mut comma = false;
                if self.peek() == Some(',') {
                    comma = true;
                    self.next();
                    while let Some(c) = self.peek() {
                        if c.is_ascii_digit() {
                            b.push(c);
                            self.next();
                        } else {
                            break;
                        }
                    }
                }
                if self.peek() == Some('}') && (!a.is_empty() || comma) {
                    self.next();
                    let mn = a.parse().unwrap_or(0);
                    let mx = if b.is_empty() {
                        if comma {
                            None
                        } else {
                            Some(mn)
                        }
                    } else {
                        Some(b.parse().unwrap())
                    };
                    (mn, mx)
                } else {
                    self.o = save;
                    return base;
                }
            }
            _ => return base,
        };
        if self.peek() == Some('?') {
            self.next();
        }
        Node::Repeat { node: Box::new(base), min, max }
    }
    fn parse_class(&mut self) -> Node {
        let neg = if self.peek() == Some('^') {
            self.next();
            true
        } else {
            false
        };
        let mut items = Vec::new();
        loop {
            match self.next() {
                None => break,
                Some(']') => break,
                Some('\\') => {
                    let e = self.next().unwrap_or('\\');
                    match e {
                        's' => items.push(ClassItem::Space),
                        'S' => items.push(ClassItem::NotR),
                        'd' => items.push(ClassItem::Range('0', '9')),
                        'r' => items.push(ClassItem::Ch('\r')),
                        'n' => items.push(ClassItem::Ch('\n')),
                        't' => items.push(ClassItem::Ch('\t')),
                        'p' => {
                            if self.peek() == Some('{') {
                                self.next();
                                let mut name = String::new();
                                while let Some(c) = self.next() {
                                    if c == '}' {
                                        break;
                                    }
                                    name.push(c);
                                }
                                match name.as_str() {
                                    "L" | "Lu" | "Ll" | "Lt" | "Lm" | "Lo" => {
                                        items.push(ClassItem::Letter)
                                    }
                                    "N" => items.push(ClassItem::Number),
                                    _ => {}
                                }
                            }
                        }
                        other => items.push(ClassItem::Ch(other)),
                    }
                }
                Some(c) => {
                    if self.peek() == Some('-') && self.c.get(self.o + 1).copied() != Some(']') {
                        self.next();
                        let hi = match self.next() {
                            Some('\\') => match self.next() {
                                Some('r') => '\r',
                                Some('n') => '\n',
                                Some('t') => '\t',
                                Some(o) => o,
                                None => '-',
                            },
                            Some(h) => h,
                            None => '-',
                        };
                        items.push(ClassItem::Range(c, hi));
                    } else {
                        items.push(ClassItem::Ch(c));
                    }
                }
            }
        }
        Node::Class { neg, items }
    }
    fn expect(&mut self, ch: char) {
        if self.peek() == Some(ch) {
            self.next();
        }
    }
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

fn is_number(c: char) -> bool {
    c.is_numeric()
}

fn class_match(neg: bool, items: &[ClassItem], c: char) -> bool {
    let mut hit = false;
    for it in items {
        let m = match it {
            ClassItem::Ch(x) => c == *x,
            ClassItem::Range(a, b) => c >= *a && c <= *b,
            ClassItem::Letter => is_letter(c),
            ClassItem::Number => is_number(c),
            ClassItem::Space => c.is_whitespace(),
            ClassItem::NotR => c != '\r',
            ClassItem::NotN => c != '\n',
        };
        if m {
            hit = true;
            break;
        }
    }
    hit != neg
}

fn match_seq(nodes: &[Node], t: &[char], pos: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
    match nodes.split_first() {
        None => k(pos),
        Some((node, rest)) => match node {
            Node::Char(c) => {
                if pos < t.len() && t[pos] == *c {
                    match_seq(rest, t, pos + 1, k)
                } else {
                    false
                }
            }
            Node::Any => {
                if pos < t.len() && t[pos] != '\n' && t[pos] != '\r' {
                    match_seq(rest, t, pos + 1, k)
                } else {
                    false
                }
            }
            Node::Class { neg, items } => {
                if pos < t.len() && class_match(*neg, items, t[pos]) {
                    match_seq(rest, t, pos + 1, k)
                } else {
                    false
                }
            }
            Node::Group(alts) => {
                for alt in alts {
                    let ok = match_seq(alt, t, pos, &mut |p2| match_seq(rest, t, p2, k));
                    if ok {
                        return true;
                    }
                }
                false
            }
            Node::Look { neg, group } => {
                let mut matched = false;
                for alt in group {
                    if match_seq(alt, t, pos, &mut |_| true) {
                        matched = true;
                        break;
                    }
                }
                if matched == *neg {
                    false
                } else {
                    match_seq(rest, t, pos, k)
                }
            }
            Node::Repeat { node, min, max } => {
                fn rep(
                    node: &Node,
                    rest: &[Node],
                    t: &[char],
                    pos: usize,
                    count: usize,
                    min: usize,
                    max: Option<usize>,
                    k: &mut dyn FnMut(usize) -> bool,
                ) -> bool {
                    let can_more = max.map(|m| count < m).unwrap_or(true);
                    if can_more {
                        let adv = match node {
                            Node::Repeat { .. } | Node::Look { .. } => false,
                            _ => true,
                        };
                        if adv {
                            let ok = match_seq(
                                core::slice::from_ref(node),
                                t,
                                pos,
                                &mut |p2| {
                                    if p2 == pos {
                                        return false;
                                    }
                                    rep(node, rest, t, p2, count + 1, min, max, k)
                                },
                            );
                            if ok {
                                return true;
                            }
                        }
                    }
                    if count >= min {
                        match_seq(rest, t, pos, k)
                    } else {
                        false
                    }
                }
                rep(node, rest, t, pos, 0, *min, *max, k)
            }
        },
    }
}

impl Regex {
    pub fn new(pattern: &str) -> Regex {
        let mut p = Parser { c: pattern.chars().collect(), o: 0 };
        Regex { alts: p.parse_alts(false) }
    }

    pub fn is_match_at(&self, t: &[char], pos: usize) -> Option<usize> {
        for alt in &self.alts {
            let mut end = None;
            let ok = match_seq(alt, t, pos, &mut |p| {
                end = Some(p);
                true
            });
            if ok {
                return end;
            }
        }
        None
    }

    pub fn find(&self, t: &[char]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < t.len() {
            let mut found = None;
            let mut j = i;
            while j < t.len() {
                if let Some(e) = self.is_match_at(t, j) {
                    if e > j {
                        found = Some((j, e));
                    }
                    break;
                }
                j += 1;
            }
            match found {
                Some((s, e)) => {
                    out.push((s, e));
                    i = e;
                }
                None => break,
            }
        }
        out
    }
}

pub fn split(pattern: &str, text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let re = Regex::new(pattern);
    let matches = re.find(&chars);
    let mut out = Vec::new();
    let mut last = 0;
    for (s, e) in matches {
        if s > last {
            out.push(chars[last..s].iter().collect());
        }
        out.push(chars[s..e].iter().collect());
        last = e;
    }
    if last < chars.len() {
        out.push(chars[last..].iter().collect());
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}
