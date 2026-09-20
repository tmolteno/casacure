//! A TaQL (Table Query Language) subset implementation.
//!
//! Covers the dialect dask-ms generates and a DDL subset for fixtures:
//!
//! - `SELECT [UNIQUE] <expr> [AS name] [, ...] FROM $N | 'path'
//!   [WHERE <expr>] [ORDERBY <expr> [DESC], ...] [GROUPBY <expr>, ...]
//!   [LIMIT n]`
//! - `SELECT *`
//! - `ORDERBY` on multiple keys, `ASC`/`DESC` (descending reverses tie order,
//!   matching casacore).
//! - Row context functions: `ROWID()`, plus a small set of scalar math
//!   functions used in `WHERE` clauses.
//! - Group context: `GROUPBY` with `GROWID()`, `GROWID()[i]`, `GCOUNT()`,
//!   `GAGGR(col)`.
//! - `SELECT UNIQUE`
//! - Scalar/array subqueries `[SELECT ... FROM ...]` indexed by `[expr]`.
//! - DDL: `CREATE TABLE path [(colspec, ...)] LIMIT n`.
//!
//! Expression evaluation follows casacore `TaQL` semantics: `&&`, `||`, `!`,
//! arithmetic, comparisons, mixed int/float arithmetic, and array indexing
//! `expr[i]`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;

use thiserror::Error;

use crate::record::{ArrayData, ArrayValue, DataType, RecordValue};
use crate::table::Table;
use crate::tabledesc::{ColumnDesc, ColumnKind, TableDesc};
use crate::WritableTable;

/// Result of a TaQL `SELECT` query: one homogeneous typed column per select
/// expression, `nrows` rows.
#[derive(Debug, Clone)]
pub struct TaqlTable {
    pub colnames: Vec<String>,
    /// Column values in row-major order.
    pub columns: Vec<Vec<RecordValue>>,
}

impl TaqlTable {
    pub fn nrows(&self) -> usize {
        self.columns.first().map_or(0, |c| c.len())
    }

    /// The index of a column, or `None`.
    pub fn col_index(&self, name: &str) -> Option<usize> {
        self.colnames.iter().position(|c| c == name)
    }

    /// All values of a column (`taql_result.getcol(name)`).
    pub fn getcol(&self, name: &str) -> Option<&Vec<RecordValue>> {
        let idx = self.col_index(name)?;
        self.columns.get(idx)
    }

    /// One cell (`taql_result.getcell(col, row)`).
    pub fn getcell(&self, col: usize, row: usize) -> Option<&RecordValue> {
        self.columns.get(col)?.get(row)
    }
}

/// Errors from parsing or executing TaQL.
#[derive(Debug, Error)]
pub enum TaqlError {
    #[error("TaQL parse error: {0}")]
    Parse(String),
    #[error("TaQL error: {0}")]
    Eval(String),
    #[error("table $N references an out-of-range table (got {0})")]
    NoSuchTable(usize),
    #[error("unknown column or function: {0}")]
    Unknown(String),
    #[error("attribute `{field}` of {what} does not exist")]
    NoSuchAttribute { what: String, field: String },
    #[error(transparent)]
    Read(#[from] crate::table::TableReadError),
    #[error(transparent)]
    Create(#[from] crate::WriteTableError),
}

/// The type of a TaQL `execute`: a query result or a created table.
#[derive(Debug)]
pub enum TaqlResult {
    Query(TaqlTable),
    Created(std::path::PathBuf),
}

type TResult<T> = Result<T, TaqlError>;

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// A runtime TaQL value.
#[derive(Debug, Clone)]
enum TqValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<TqValue>),
    /// Result of a `[SELECT ...]` subquery.
    Subtable(std::rc::Rc<TaqlTable>),
    /// A compiled pattern from `regex`/`pattern`/`sqlpattern`.
    Regex(RegexKind, String),
}

/// The pattern flavour produced by `regex` / `pattern` / `sqlpattern`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RegexKind {
    /// POSIX-ish regex (`.` `*` `+` `?` `[...]` `^` `$` `|` `()`).
    Regex,
    /// C-style pattern: `*` matches any run, `?` one char.
    Pattern,
    /// SQL pattern: `%` any run, `_` one char.
    SqlPattern,
}

impl PartialEq for TqValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TqValue::Bool(a), TqValue::Bool(b)) => a == b,
            (TqValue::Int(a), TqValue::Int(b)) => a == b,
            (TqValue::Float(a), TqValue::Float(b)) => a.to_bits() == b.to_bits(),
            (TqValue::Str(a), TqValue::Str(b)) => a == b,
            (TqValue::Arr(a), TqValue::Arr(b)) => a == b,
            (TqValue::Regex(a, ap), TqValue::Regex(b, bp)) => a == b && ap == bp,
            _ => false,
        }
    }
}

impl Eq for TqValue {}

impl std::hash::Hash for TqValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            TqValue::Bool(b) => (0u8, b).hash(state),
            TqValue::Int(i) => (1u8, i).hash(state),
            TqValue::Float(f) => (2u8, f.to_bits()).hash(state),
            TqValue::Str(s) => (3u8, s).hash(state),
            TqValue::Arr(items) => {
                4u8.hash(state);
                items.hash(state);
            }
            TqValue::Subtable(_) => 5u8.hash(state),
            TqValue::Regex(k, s) => {
                6u8.hash(state);
                k.hash(state);
                s.hash(state);
            }
        }
    }
}

impl TqValue {
    fn truthy(&self) -> bool {
        match self {
            TqValue::Bool(b) => *b,
            TqValue::Int(i) => *i != 0,
            TqValue::Float(f) => *f != 0.0,
            _ => true,
        }
    }

    fn to_record(&self) -> RecordValue {
        match self {
            TqValue::Bool(b) => RecordValue::Bool(*b),
            TqValue::Int(i) => RecordValue::Int64(*i),
            TqValue::Float(f) => RecordValue::Double(*f),
            TqValue::Str(s) => RecordValue::String(s.clone()),
            TqValue::Arr(items) => {
                let elems: Vec<_> = items.iter().map(TqValue::to_record).collect();
                RecordValue::Array(ArrayValue {
                    shape: vec![elems.len() as u32],
                    data: elems_array_data(&elems),
                })
            }
            TqValue::Subtable(_) => RecordValue::Int64(0),
            TqValue::Regex(_, _) => RecordValue::Int64(0),
        }
    }
}

fn elems_array_data(elems: &[RecordValue]) -> ArrayData {
    if elems.iter().all(|e| matches!(e, RecordValue::Int64(_))) {
        ArrayData::Int64(
            elems
                .iter()
                .map(|e| match e {
                    RecordValue::Int64(i) => *i,
                    _ => unreachable!(),
                })
                .collect(),
        )
    } else if elems.iter().all(|e| matches!(e, RecordValue::Double(_))) {
        ArrayData::Double(
            elems
                .iter()
                .map(|e| match e {
                    RecordValue::Double(d) => *d,
                    _ => unreachable!(),
                })
                .collect(),
        )
    } else if elems.iter().all(|e| matches!(e, RecordValue::String(_))) {
        ArrayData::String(
            elems
                .iter()
                .map(|e| match e {
                    RecordValue::String(s) => s.clone(),
                    _ => unreachable!(),
                })
                .collect(),
        )
    } else if elems.iter().all(|e| matches!(e, RecordValue::Bool(_))) {
        ArrayData::Bool(
            elems
                .iter()
                .map(|e| match e {
                    RecordValue::Bool(b) => *b,
                    _ => unreachable!(),
                })
                .collect(),
        )
    } else if elems
        .iter()
        .all(|e| matches!(e, RecordValue::Complex(_, _)))
    {
        ArrayData::Complex(
            elems
                .iter()
                .map(|e| match e {
                    RecordValue::Complex(re, im) => (*re, *im),
                    _ => unreachable!(),
                })
                .collect(),
        )
    } else if elems
        .iter()
        .all(|e| matches!(e, RecordValue::DComplex(_, _)))
    {
        ArrayData::DComplex(
            elems
                .iter()
                .map(|e| match e {
                    RecordValue::DComplex(re, im) => (*re, *im),
                    _ => unreachable!(),
                })
                .collect(),
        )
    } else {
        // Mixed numeric types fall back to Double.
        ArrayData::Double(
            elems
                .iter()
                .map(|e| match e {
                    RecordValue::Int64(i) => *i as f64,
                    RecordValue::Double(d) => *d,
                    RecordValue::Int(i) => f64::from(*i),
                    RecordValue::Float(f) => f64::from(*f),
                    RecordValue::Bool(b) => f64::from(*b),
                    _ => 0.0,
                })
                .collect(),
        )
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),
    Op(String),
}

fn tokenize(input: &str) -> Result<Vec<Tok>, TaqlError> {
    let mut toks = Vec::new();
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '\'' => {
                let mut s = String::new();
                i += 1;
                loop {
                    if i >= bytes.len() {
                        return Err(TaqlError::Parse("unterminated string".into()));
                    }
                    let c = bytes[i] as char;
                    if c == '\'' {
                        if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                            s.push('\'');
                            i += 2;
                        } else {
                            i += 1;
                            break;
                        }
                    } else {
                        s.push(c);
                        i += 1;
                    }
                }
                toks.push(Tok::Str(s));
            }
            c if c.is_ascii_digit() => {
                let start = i;
                while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                    i += 1;
                }
                let mut is_float = false;
                if i < bytes.len() && bytes[i] == b'.' {
                    is_float = true;
                    i += 1;
                    while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < bytes.len() && matches!(bytes[i], b'e' | b'E') {
                    let j = i + 1;
                    if j < bytes.len()
                        && ((bytes[j] as char).is_ascii_digit()
                            || ((bytes[j] == b'+' || bytes[j] == b'-')
                                && j + 1 < bytes.len()
                                && (bytes[j + 1] as char).is_ascii_digit()))
                    {
                        is_float = true;
                        i = j + 1;
                        while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
                let text = &input[start..i];
                if is_float {
                    toks.push(Tok::Float(text.parse().map_err(|_| {
                        TaqlError::Parse(format!("invalid number {text}"))
                    })?));
                } else {
                    toks.push(Tok::Int(text.parse().map_err(|_| {
                        TaqlError::Parse(format!("invalid number {text}"))
                    })?));
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' || c == '$' => {
                let start = i;
                while i < bytes.len()
                    && ((bytes[i] as char).is_ascii_alphanumeric()
                        || bytes[i] == b'_'
                        || bytes[i] == b'$'
                        || bytes[i] == b'.')
                {
                    i += 1;
                }
                toks.push(Tok::Ident(input[start..i].to_string()));
            }
            c if c.is_ascii_punctuation() => {
                let rest = &input[i..];
                let three = &rest[..rest.len().min(3)];
                let op = if ["<=>", "!<", "!>"].contains(&three) {
                    i += 3;
                    three.to_string()
                } else if ["==", "!=", "<=", ">=", "<>", "&&", "||", "!~"].contains(&two(rest)) {
                    i += 2;
                    two(rest).to_string()
                } else {
                    i += 1;
                    c.to_string()
                };
                toks.push(Tok::Op(op));
            }
            other => {
                return Err(TaqlError::Parse(format!(
                    "unexpected character {other:?} at position {i}"
                )));
            }
        }
    }
    Ok(toks)
}

fn two(s: &str) -> &str {
    &s[..s.len().min(2)]
}

/// Words that end an unquoted table path.
fn is_clause_word(w: &str) -> bool {
    matches!(
        w.to_ascii_lowercase().as_str(),
        "where"
            | "orderby"
            | "groupby"
            | "having"
            | "limit"
            | "offset"
            | "as"
            | "unique"
            | "desc"
            | "asc"
            | "from"
            | "and"
            | "or"
            | "not"
            | "into"
            | "set"
            | "add"
            | "drop"
            | "rename"
            | "remove"
            | "insert"
            | "update"
            | "delete"
            | "select"
            | "count"
            | "calc"
            | "show"
            | "help"
            | "alter"
            | "create"
            | "table"
            | "droptable"
            | "values"
            | "column"
            | "keyword"
            | "keywords"
            | "comment"
            | "ndim"
            | "shape"
            | "option"
            | "maxlen"
    )
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum BinOp {
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    /// `LIKE`/`ILIKE` pattern match; `case_insensitive` = ILIKE,
    /// `negated` = `NOT LIKE`.
    Like {
        case_insensitive: bool,
        negated: bool,
    },
    /// `IN` set membership; `negated` = `NOT IN`.
    In {
        negated: bool,
    },
    /// `~` / `!~` pattern match against a `regex`/`pattern`/`sqlpattern`
    /// value (casacore's REGEX operator).
    Regex {
        negated: bool,
    },
}

/// The comparison-level keyword operators parsed by `parse_cmp`.
#[derive(Debug, Clone, Copy)]
enum CmpKind {
    /// `LIKE` / `ILIKE` (case-insensitive).
    Like(bool),
    /// `IN`.
    In,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone)]
enum Expr {
    Int(i64),
    Float(f64),
    Str(String),
    /// Bare identifier: column reference.
    Name(String),
    Call(String, Vec<Expr>),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// `expr[expr]` (array element or subtable row).
    Index(Box<Expr>, Box<Expr>),
    /// `[SELECT ...]` subquery.
    Subquery(Box<Select>),
    /// `(a, b, c)` / `[a, b, c]` literal set (right operand of `IN`).
    Set(Vec<Expr>),
}

#[derive(Debug, Clone)]
struct Select {
    unique: bool,
    columns: Option<Vec<(Expr, Option<String>)>>,
    table: TableRef,
    where_: Option<Expr>,
    orderby: Vec<OrderKey>,
    groupby: Vec<Expr>,
    having: Option<Expr>,
    limit: Option<i64>,
    offset: Option<i64>,
    /// `SELECT ... INTO <table>`: the result is written to a new table.
    into: Option<String>,
}

#[derive(Debug, Clone)]
enum TableRef {
    /// `$N` — the N-th supplied table (1-based).
    Table(usize),
    /// `'path'` — a table on disk.
    Path(String),
}

#[derive(Debug, Clone)]
struct OrderKey {
    expr: Expr,
    desc: bool,
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect_op(&mut self, op: &str) -> Result<(), TaqlError> {
        match self.next() {
            Some(Tok::Op(o)) if o == op => Ok(()),
            other => Err(TaqlError::Parse(format!(
                "expected `{op}`, found {other:?}"
            ))),
        }
    }

    fn expect_ident(&mut self) -> Result<String, TaqlError> {
        match self.next() {
            Some(Tok::Ident(s)) => Ok(s),
            other => Err(TaqlError::Parse(format!(
                "expected identifier, found {other:?}"
            ))),
        }
    }

    fn parse_select(&mut self) -> Result<Select, TaqlError> {
        let kw = self.expect_ident()?;
        if !kw.eq_ignore_ascii_case("select") {
            return Err(TaqlError::Parse(format!("expected SELECT, found {kw}")));
        }
        let mut unique = false;
        if let Some(Tok::Ident(w)) = self.peek() {
            if w.eq_ignore_ascii_case("unique") {
                unique = true;
                self.next();
            }
        }
        // Columns: either `*` or a comma list of `expr [AS name]`.
        let columns = if matches!(self.peek(), Some(Tok::Op(o)) if o == "*") {
            self.next();
            None
        } else {
            Some(self.parse_column_list()?)
        };
        // Optional `INTO <table>` names the target table of the result.
        let mut into = None;
        if matches!(self.peek(), Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("into")) {
            self.next();
            into = Some(parse_table_path(self)?);
        }
        let from = self.expect_ident()?;
        if !from.eq_ignore_ascii_case("from") {
            return Err(TaqlError::Parse(format!("expected FROM, found {from}")));
        }
        // A table reference is uniform across statements: `$N`, a quoted
        // `'path'`, or an unquoted path (relative name or one starting with
        // `/`, `-`, `.`, `:`), exactly as `parse_table_ref` handles it.
        let table = parse_table_ref(self)?;

        let mut where_ = None;
        let mut orderby = Vec::new();
        let mut groupby = Vec::new();
        let mut having = None;
        let mut limit = None;
        let mut offset = None;
        loop {
            match self.peek() {
                Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("where") => {
                    self.next();
                    where_ = Some(self.parse_expr()?);
                }
                Some(Tok::Ident(w))
                    if w.eq_ignore_ascii_case("order") || w.eq_ignore_ascii_case("orderby") =>
                {
                    let spaced = w.eq_ignore_ascii_case("order");
                    self.next();
                    // `ORDER BY <expr>` (casacore's spaced form) vs `ORDERBY`.
                    if spaced {
                        if let Some(Tok::Ident(b)) = self.peek() {
                            if b.eq_ignore_ascii_case("by") {
                                self.next();
                            }
                        }
                    }
                    loop {
                        let expr = self.parse_expr()?;
                        let mut desc = false;
                        if let Some(Tok::Ident(w)) = self.peek() {
                            if w.eq_ignore_ascii_case("desc") {
                                desc = true;
                                self.next();
                            } else if w.eq_ignore_ascii_case("asc") {
                                self.next();
                            }
                        }
                        orderby.push(OrderKey { expr, desc });
                        match self.peek() {
                            Some(Tok::Op(o)) if o == "," => {
                                self.next();
                            }
                            _ => break,
                        }
                    }
                }
                Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("groupby") => {
                    self.next();
                    loop {
                        groupby.push(self.parse_expr()?);
                        match self.peek() {
                            Some(Tok::Op(o)) if o == "," => {
                                self.next();
                            }
                            _ => break,
                        }
                    }
                }
                Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("limit") => {
                    self.next();
                    match self.next() {
                        Some(Tok::Int(n)) => limit = Some(n),
                        other => {
                            return Err(TaqlError::Parse(format!(
                                "expected integer LIMIT, found {other:?}"
                            )));
                        }
                    }
                }
                Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("offset") => {
                    self.next();
                    match self.next() {
                        Some(Tok::Int(n)) => offset = Some(n),
                        other => {
                            return Err(TaqlError::Parse(format!(
                                "expected integer OFFSET, found {other:?}"
                            )));
                        }
                    }
                }
                Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("having") => {
                    self.next();
                    having = Some(self.parse_expr()?);
                }
                _ => break,
            }
        }
        Ok(Select {
            unique,
            columns,
            table,
            where_,
            orderby,
            groupby,
            having,
            limit,
            offset,
            into,
        })
    }

    fn parse_column_list(&mut self) -> Result<Vec<(Expr, Option<String>)>, TaqlError> {
        let mut cols = Vec::new();
        loop {
            let expr = self.parse_expr()?;
            let mut alias = None;
            if matches!(self.peek(), Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("as")) {
                self.next();
                alias = Some(self.expect_ident()?);
            }
            cols.push((expr, alias));
            match self.peek() {
                Some(Tok::Op(o)) if o == "," => {
                    self.next();
                }
                _ => break,
            }
        }
        Ok(cols)
    }

    fn parse_expr(&mut self) -> Result<Expr, TaqlError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, TaqlError> {
        let mut left = self.parse_and()?;
        loop {
            let is_or = matches!(self.peek(), Some(Tok::Op(o)) if o == "||" || o == "|")
                || matches!(self.peek(), Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("or"));
            if is_or {
                self.next();
                let right = self.parse_and()?;
                left = Expr::Binary(BinOp::Or, Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, TaqlError> {
        let mut left = self.parse_cmp()?;
        loop {
            let is_and = matches!(self.peek(), Some(Tok::Op(o)) if o == "&&" || o == "&")
                || matches!(self.peek(), Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("and"));
            if is_and {
                self.next();
                let right = self.parse_cmp()?;
                left = Expr::Binary(BinOp::And, Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_cmp(&mut self) -> Result<Expr, TaqlError> {
        let mut left = self.parse_add()?;

        // `[NOT] LIKE / ILIKE / IN` at comparison precedence; restore the
        // token position if the `NOT` is not followed by one of these.
        let mut negated = false;
        let mut kind: Option<CmpKind> = None;
        let saved = self.pos;
        if let Some(Tok::Ident(w)) = self.peek() {
            let up = || w.to_ascii_uppercase();
            if up() == "NOT" {
                self.next();
                match self.peek() {
                    Some(Tok::Ident(w2)) => match w2.to_ascii_uppercase().as_str() {
                        "LIKE" => {
                            self.next();
                            kind = Some(CmpKind::Like(false));
                        }
                        "ILIKE" => {
                            self.next();
                            kind = Some(CmpKind::Like(true));
                        }
                        "IN" => {
                            self.next();
                            kind = Some(CmpKind::In);
                        }
                        _ => self.pos = saved,
                    },
                    _ => self.pos = saved,
                }
                negated = kind.is_some();
            } else if up() == "LIKE" {
                self.next();
                kind = Some(CmpKind::Like(false));
            } else if up() == "ILIKE" {
                self.next();
                kind = Some(CmpKind::Like(true));
            } else if up() == "IN" {
                self.next();
                kind = Some(CmpKind::In);
            }
        }

        match kind {
            Some(CmpKind::Like(ci)) => {
                let right = self.parse_unary()?;
                left = Expr::Binary(
                    BinOp::Like {
                        case_insensitive: ci,
                        negated,
                    },
                    Box::new(left),
                    Box::new(right),
                );
                return Ok(left);
            }
            Some(CmpKind::In) => {
                let open = match self.next() {
                    Some(Tok::Op(o)) if o == "(" || o == "[" => o,
                    other => {
                        return Err(TaqlError::Parse(format!(
                            "IN expects a set, found {other:?}"
                        )));
                    }
                };
                let close = if open == "(" { ")" } else { "]" };
                let mut items = Vec::new();
                if !matches!(self.peek(), Some(Tok::Op(o)) if o == close) {
                    items.push(self.parse_expr()?);
                    while matches!(self.peek(), Some(Tok::Op(o)) if o == ",") {
                        self.next();
                        items.push(self.parse_expr()?);
                    }
                }
                self.expect_op(close)?;
                left = Expr::Binary(
                    BinOp::In { negated },
                    Box::new(left),
                    Box::new(Expr::Set(items)),
                );
                return Ok(left);
            }
            None => {}
        }

        let op = match self.peek() {
            Some(Tok::Op(o)) if o == "==" || o == "=" => Some(BinOp::Eq),
            Some(Tok::Op(o)) if o == "!=" || o == "<>" => Some(BinOp::Ne),
            Some(Tok::Op(o)) if o == "<" => Some(BinOp::Lt),
            Some(Tok::Op(o)) if o == "<=" => Some(BinOp::Le),
            Some(Tok::Op(o)) if o == ">" => Some(BinOp::Gt),
            Some(Tok::Op(o)) if o == ">=" => Some(BinOp::Ge),
            Some(Tok::Op(o)) if o == "~" => Some(BinOp::Regex { negated: false }),
            Some(Tok::Op(o)) if o == "!~" => Some(BinOp::Regex { negated: true }),
            _ => None,
        };
        if let Some(op) = op {
            self.next();
            let right = self.parse_add()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_add(&mut self) -> Result<Expr, TaqlError> {
        let mut left = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op(o)) if o == "+" => Some(BinOp::Add),
                Some(Tok::Op(o)) if o == "-" => Some(BinOp::Sub),
                _ => None,
            };
            let Some(op) = op else { break };
            self.next();
            let right = self.parse_mul()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<Expr, TaqlError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op(o)) if o == "*" => Some(BinOp::Mul),
                Some(Tok::Op(o)) if o == "/" => Some(BinOp::Div),
                Some(Tok::Op(o)) if o == "%" => Some(BinOp::Mod),
                _ => None,
            };
            let Some(op) = op else { break };
            self.next();
            let right = self.parse_unary()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, TaqlError> {
        match self.peek() {
            Some(Tok::Op(o)) if o == "-" => {
                self.next();
                Ok(Expr::Unary(UnOp::Neg, Box::new(self.parse_unary()?)))
            }
            Some(Tok::Op(o)) if o == "!" || o == "~" => {
                self.next();
                Ok(Expr::Unary(UnOp::Not, Box::new(self.parse_unary()?)))
            }
            Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("not") => {
                self.next();
                Ok(Expr::Unary(UnOp::Not, Box::new(self.parse_unary()?)))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, TaqlError> {
        let mut expr = self.parse_primary()?;
        loop {
            if matches!(self.peek(), Some(Tok::Op(o)) if o == "[") {
                self.next();
                let idx = self.parse_expr()?;
                self.expect_op("]")?;
                expr = Expr::Index(Box::new(expr), Box::new(idx));
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, TaqlError> {
        match self.next() {
            Some(Tok::Int(n)) => Ok(Expr::Int(n)),
            Some(Tok::Float(f)) => Ok(Expr::Float(f)),
            Some(Tok::Str(s)) => Ok(Expr::Str(s)),
            Some(Tok::Ident(name)) => {
                if matches!(self.peek(), Some(Tok::Op(o)) if o == "(") {
                    self.next();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Tok::Op(o)) if o == ")") {
                        args.push(self.parse_expr()?);
                        while matches!(self.peek(), Some(Tok::Op(o)) if o == ",") {
                            self.next();
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect_op(")")?;
                    return Ok(Expr::Call(name, args));
                }
                let upper = name.to_ascii_uppercase();
                if upper == "TRUE" {
                    return Ok(Expr::Int(1));
                }
                if upper == "FALSE" {
                    return Ok(Expr::Int(0));
                }
                Ok(Expr::Name(name))
            }
            Some(Tok::Op(o)) if o == "(" => {
                let e = self.parse_expr()?;
                self.expect_op(")")?;
                Ok(e)
            }
            Some(Tok::Op(o)) if o == "[" => {
                // Subquery: [SELECT ...]
                let sel = self.parse_select()?;
                self.expect_op("]")?;
                Ok(Expr::Subquery(Box::new(sel)))
            }
            other => Err(TaqlError::Parse(format!(
                "unexpected token in expression: {other:?}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Execute a TaQL command. `tables` are the positional `$N` tables.
pub fn execute(query: &str, tables: &[&Table]) -> TResult<TaqlResult> {
    execute_into(query, tables, &mut Vec::new())
}

/// Execute a TaQL command, recording every on-disk table directory that a
/// mutating statement writes to (UPDATE/DELETE/INSERT/ALTER/DROPTABLE,
/// CREATE, SELECT INTO) in `touched`, so an embedding can invalidate caches
/// keyed by directory (the pyo3 layer drops stale writable state for these
/// paths after the call).
pub fn execute_into(
    query: &str,
    tables: &[&Table],
    touched: &mut Vec<std::path::PathBuf>,
) -> TResult<TaqlResult> {
    let toks = tokenize(query)?;
    let mut p = Parser { toks, pos: 0 };
    let kw = match p.peek() {
        Some(Tok::Ident(w)) => w.clone(),
        _ => return Err(TaqlError::Parse("empty query".into())),
    };
    if kw.eq_ignore_ascii_case("create") {
        let created = create_table(query)?;
        touched.push(created.clone());
        return Ok(TaqlResult::Created(created));
    }
    if kw.eq_ignore_ascii_case("count") {
        let out = run_count(query, tables)?;
        return Ok(TaqlResult::Query(out));
    }
    match kw.to_ascii_uppercase().as_str() {
        "UPDATE" => return Ok(TaqlResult::Query(run_update(query, tables, touched)?)),
        "DELETE" => return Ok(TaqlResult::Query(run_delete(query, tables, touched)?)),
        "INSERT" => return Ok(TaqlResult::Query(run_insert(query, tables, touched)?)),
        "DROPTABLE" => return Ok(TaqlResult::Query(run_droptable(query, tables, touched)?)),
        "ALTER" => return Ok(TaqlResult::Query(run_alter(query, tables, touched)?)),
        "SHOW" | "HELP" => return Ok(TaqlResult::Query(run_show(query, tables)?)),
        "CALC" => return Ok(TaqlResult::Query(run_calc(query, tables)?)),
        _ => {}
    }
    let sel = p.parse_select()?;
    if let Some(into_path) = &sel.into {
        let created = run_select_into(into_path, &sel, tables)?;
        touched.push(created.clone());
        return Ok(TaqlResult::Created(created));
    }
    if matches!(p.peek(), Some(Tok::Ident(w)) if w == ";") {
        p.next();
    }
    if p.pos < p.toks.len() {
        return Err(TaqlError::Parse(format!(
            "trailing tokens after query: {:?}",
            &p.toks[p.pos..]
        )));
    }
    let out = run_select(&sel, tables)?;
    Ok(TaqlResult::Query(out))
}

/// The `COUNT [col|*] FROM table [WHERE expr]` command: the number of rows
/// matching the optional WHERE clause, returned as a single-row table
/// (casacore `TableGram.yy` `countcomm`).
fn run_count(query: &str, tables: &[&Table]) -> TResult<TaqlTable> {
    let toks = tokenize(query)?;
    let mut p = Parser { toks, pos: 0 };
    let kw = p.expect_ident()?;
    if !kw.eq_ignore_ascii_case("count") {
        return Err(TaqlError::Parse("expected COUNT".into()));
    }
    // The column list is ignored (`COUNT *` or `COUNT col1, col2`).
    match p.peek() {
        Some(Tok::Op(o)) if o == "*" => {
            p.next();
        }
        Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("from") => {}
        Some(_) => {
            let _ = p.parse_column_list()?;
        }
        None => {
            return Err(TaqlError::Parse("expected FROM after COUNT".into()));
        }
    }
    let from = p.expect_ident()?;
    if !from.eq_ignore_ascii_case("from") {
        return Err(TaqlError::Parse(format!("expected FROM, found {from}")));
    }
    let table = parse_table_ref(&mut p)?;
    let mut where_ = None;
    if let Some(Tok::Ident(w)) = p.peek() {
        if w.eq_ignore_ascii_case("where") {
            p.next();
            where_ = Some(p.parse_expr()?);
        }
    }

    let owned: Option<Table>;
    let t: &Table = match &table {
        TableRef::Table(n) => tables
            .get(n.saturating_sub(1))
            .copied()
            .ok_or(TaqlError::NoSuchTable(*n))?,
        TableRef::Path(path) => {
            owned = Some(
                Table::open(path, false)
                    .map_err(|e| TaqlError::Eval(format!("cannot open table {path}: {e}")))?,
            );
            owned.as_ref().unwrap()
        }
    };
    let colidx: HashMap<String, usize> = t
        .colnames()
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();
    let ctx = EvalCtx {
        table: t,
        colidx,
        columns: RefCell::new(HashMap::new()),
        tables,
    };
    let count = match &where_ {
        None => t.nrows() as i64,
        Some(w) => (0..t.nrows() as i64)
            .filter(|&r| ctx.eval_row(w, r).map(|v| v.truthy()).unwrap_or(false))
            .count() as i64,
    };
    Ok(TaqlTable {
        colnames: vec!["count".to_string()],
        columns: vec![vec![RecordValue::Int64(count)]],
    })
}

// ---------------------------------------------------------------------------
// Write statements (Tier B): UPDATE / DELETE / INSERT / DROPTABLE / ALTER,
// SHOW / HELP / CALC, and SELECT ... INTO. In-place edits materialise the
// table into a `WritableTable`, mutate the cell store, and flush (which
// regenerates `table.dat` + every data file), matching how the pyo3 layer
// already re-opens tables for writing.
// ---------------------------------------------------------------------------

fn empty_table() -> TaqlTable {
    TaqlTable {
        colnames: Vec::new(),
        columns: Vec::new(),
    }
}

/// Parse an unquoted (or quoted) table path: `'path'`, `relative`, or an
/// absolute path beginning with `/` (also `-`, `.`, `:` fragments). Numeric
/// path segments (e.g. `casacure-test-1221-0`) are consumed too.
fn parse_table_path(p: &mut Parser) -> TResult<String> {
    fn rest(p: &mut Parser, mut name: String) -> TResult<String> {
        loop {
            match p.peek() {
                Some(Tok::Op(o)) if matches!(o.as_str(), "/" | "-" | "." | ":") => {
                    name.push_str(o);
                    p.next();
                }
                Some(Tok::Ident(w)) if !is_clause_word(w) => {
                    name.push_str(w);
                    p.next();
                }
                Some(Tok::Int(n)) => {
                    name.push_str(&n.to_string());
                    p.next();
                }
                _ => break,
            }
        }
        while name.ends_with('/') {
            name.pop();
        }
        Ok(name)
    }
    match p.next() {
        Some(Tok::Str(s)) => Ok(s),
        Some(Tok::Op(o)) if matches!(o.as_str(), "/" | "-" | "." | ":") => rest(p, o),
        Some(Tok::Ident(s)) => rest(p, s),
        other => Err(TaqlError::Parse(format!(
            "expected table path, found {other:?}"
        ))),
    }
}

fn parse_table_ref(p: &mut Parser) -> TResult<TableRef> {
    if matches!(p.peek(), Some(Tok::Ident(s)) if s.starts_with('$')) {
        let s = p.expect_ident()?;
        let n: usize = s[1..]
            .parse()
            .map_err(|_| TaqlError::Parse(format!("invalid table reference {s}")))?;
        if n == 0 {
            return Err(TaqlError::Parse("table references are 1-based".into()));
        }
        return Ok(TableRef::Table(n));
    }
    Ok(TableRef::Path(parse_table_path(p)?))
}

/// Resolve a table reference; `owned` holds the opened table alive for the
/// caller's scope.
fn resolve_table<'a>(
    tr: &TableRef,
    tables: &'a [&'a Table],
    owned: &'a mut Option<Table>,
) -> TResult<&'a Table> {
    Ok(match tr {
        TableRef::Table(n) => tables
            .get(n.saturating_sub(1))
            .copied()
            .ok_or(TaqlError::NoSuchTable(*n))?,
        TableRef::Path(path) => {
            *owned = Some(
                Table::open(path, false)
                    .map_err(|e| TaqlError::Eval(format!("cannot open table {path}: {e}")))?,
            );
            owned.as_ref().unwrap()
        }
    })
}

/// Optional `WHERE` / `ORDERBY` / `LIMIT` / `OFFSET` clause tail shared by
/// the write statements and `COUNT`.
struct Tail {
    where_: Option<Expr>,
    orderby: Vec<OrderKey>,
    limit: Option<i64>,
    offset: Option<i64>,
}

fn parse_tail(p: &mut Parser) -> TResult<Tail> {
    let mut tail = Tail {
        where_: None,
        orderby: Vec::new(),
        limit: None,
        offset: None,
    };
    loop {
        match p.peek() {
            Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("where") => {
                p.next();
                tail.where_ = Some(p.parse_expr()?);
            }
            Some(Tok::Ident(w))
                if w.eq_ignore_ascii_case("order") || w.eq_ignore_ascii_case("orderby") =>
            {
                let spaced = w.eq_ignore_ascii_case("order");
                p.next();
                if spaced {
                    if let Some(Tok::Ident(b)) = p.peek() {
                        if b.eq_ignore_ascii_case("by") {
                            p.next();
                        }
                    }
                }
                loop {
                    let expr = p.parse_expr()?;
                    let mut desc = false;
                    if let Some(Tok::Ident(k)) = p.peek() {
                        if k.eq_ignore_ascii_case("desc") {
                            desc = true;
                            p.next();
                        } else if k.eq_ignore_ascii_case("asc") {
                            p.next();
                        }
                    }
                    tail.orderby.push(OrderKey { expr, desc });
                    if matches!(p.peek(), Some(Tok::Op(o)) if o == ",") {
                        p.next();
                    } else {
                        break;
                    }
                }
            }
            Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("limit") => {
                p.next();
                match p.next() {
                    Some(Tok::Int(n)) => tail.limit = Some(n),
                    other => {
                        return Err(TaqlError::Parse(format!(
                            "expected integer LIMIT, found {other:?}"
                        )));
                    }
                }
            }
            Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("offset") => {
                p.next();
                match p.next() {
                    Some(Tok::Int(n)) => tail.offset = Some(n),
                    other => {
                        return Err(TaqlError::Parse(format!(
                            "expected integer OFFSET, found {other:?}"
                        )));
                    }
                }
            }
            _ => break,
        }
    }
    Ok(tail)
}

/// The rows a write statement acts on: filtered, sorted, offset, limited.
fn selected_rows(ctx: &EvalCtx<'_>, tail: &Tail, n: u64) -> TResult<Vec<i64>> {
    let mut rows: Vec<i64> = (0..n as i64).collect();
    if let Some(w) = &tail.where_ {
        rows.retain(|&r| ctx.eval_row(w, r).map(|v| v.truthy()).unwrap_or(false));
    }
    if !tail.orderby.is_empty() {
        sort_rows(ctx, &mut rows, &tail.orderby)?;
    }
    if let Some(off) = tail.offset {
        rows.drain(..(off.min(rows.len() as i64).max(0)) as usize);
    }
    if let Some(lim) = tail.limit {
        rows.truncate(lim.max(0) as usize);
    }
    Ok(rows)
}

fn row_ctx<'a>(t: &'a Table, tables: &'a [&'a Table]) -> EvalCtx<'a> {
    let colidx = t
        .colnames()
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();
    EvalCtx {
        table: t,
        colidx,
        columns: RefCell::new(HashMap::new()),
        tables,
    }
}

fn writable_edit(t: &Table) -> TResult<WritableTable> {
    WritableTable::from_table(std::path::PathBuf::from(t.name()), t)
        .map_err(|e| TaqlError::Eval(format!("cannot edit table {}: {e}", t.name())))
}

/// `UPDATE table SET col = expr [, ...] [WHERE ...] [ORDERBY ...] [LIMIT] [OFFSET]`.
fn run_update(
    query: &str,
    tables: &[&Table],
    touched: &mut Vec<std::path::PathBuf>,
) -> TResult<TaqlTable> {
    let toks = tokenize(query)?;
    let mut p = Parser { toks, pos: 0 };
    if !p.expect_ident()?.eq_ignore_ascii_case("update") {
        return Err(TaqlError::Parse("expected UPDATE".into()));
    }
    let tr = parse_table_ref(&mut p)?;
    if !p.expect_ident()?.eq_ignore_ascii_case("set") {
        return Err(TaqlError::Parse("expected SET".into()));
    }
    let mut assigns = Vec::new();
    loop {
        let col = p.expect_ident()?;
        p.expect_op("=")?;
        let expr = p.parse_expr()?;
        assigns.push((col, expr));
        if matches!(p.peek(), Some(Tok::Op(o)) if o == ",") {
            p.next();
        } else {
            break;
        }
    }
    let tail = parse_tail(&mut p)?;

    let mut owned: Option<Table> = None;
    let t = resolve_table(&tr, tables, &mut owned)?;
    touched.push(t.name().into());
    let ctx = row_ctx(t, tables);
    let rows = selected_rows(&ctx, &tail, t.nrows())?;
    let mut wt = writable_edit(t)?;
    for (name, expr) in &assigns {
        let ci = wt
            .desc()
            .columns
            .iter()
            .position(|c| c.name == *name)
            .ok_or_else(|| TaqlError::Eval(format!("UPDATE: no column {name}")))?;
        let dt = wt.desc().columns[ci].data_type;
        for &r in &rows {
            let v = ctx.eval_row(expr, r)?;
            wt.putcell(ci, r as u64, coerce_to(dt, v.to_record()))?;
        }
    }
    wt.flush()?;
    Ok(empty_table())
}

/// `DELETE FROM table [WHERE ...] [ORDERBY ...] [LIMIT] [OFFSET]`.
fn run_delete(
    query: &str,
    tables: &[&Table],
    touched: &mut Vec<std::path::PathBuf>,
) -> TResult<TaqlTable> {
    let toks = tokenize(query)?;
    let mut p = Parser { toks, pos: 0 };
    if !p.expect_ident()?.eq_ignore_ascii_case("delete") {
        return Err(TaqlError::Parse("expected DELETE".into()));
    }
    if !p.expect_ident()?.eq_ignore_ascii_case("from") {
        return Err(TaqlError::Parse("expected FROM after DELETE".into()));
    }
    let tr = parse_table_ref(&mut p)?;
    let tail = parse_tail(&mut p)?;

    let mut owned: Option<Table> = None;
    let t = resolve_table(&tr, tables, &mut owned)?;
    touched.push(t.name().into());
    let ctx = row_ctx(t, tables);
    let rows = selected_rows(&ctx, &tail, t.nrows())?;
    let mut wt = writable_edit(t)?;
    let drop: Vec<u64> = rows.iter().map(|&r| r as u64).collect();
    wt.drop_rows(&drop);
    wt.flush()?;
    Ok(empty_table())
}

/// `INSERT INTO table [(col, ...)] <SELECT ...>`.
fn run_insert(
    query: &str,
    tables: &[&Table],
    touched: &mut Vec<std::path::PathBuf>,
) -> TResult<TaqlTable> {
    let toks = tokenize(query)?;
    let mut p = Parser { toks, pos: 0 };
    if !p.expect_ident()?.eq_ignore_ascii_case("insert") {
        return Err(TaqlError::Parse("expected INSERT".into()));
    }
    if !p.expect_ident()?.eq_ignore_ascii_case("into") {
        return Err(TaqlError::Parse("expected INTO after INSERT".into()));
    }
    let tr = parse_table_ref(&mut p)?;
    let mut names: Option<Vec<String>> = None;
    if matches!(p.peek(), Some(Tok::Op(o)) if o == "(") {
        p.next();
        let mut list = Vec::new();
        list.push(p.expect_ident()?);
        while matches!(p.peek(), Some(Tok::Op(o)) if o == ",") {
            p.next();
            list.push(p.expect_ident()?);
        }
        p.expect_op(")")?;
        names = Some(list);
    }
    let sel = p.parse_select()?;
    let result = run_select(&sel, tables)?;

    let mut owned: Option<Table> = None;
    let t = resolve_table(&tr, tables, &mut owned)?;
    touched.push(t.name().into());
    let mut wt = writable_edit(t)?;
    let start = wt.nrows();
    if result.columns.is_empty() {
        return Ok(empty_table());
    }
    let ncols = wt.desc().columns.len();
    wt.addrows(result.nrows() as u64);
    for (ci, col) in result.columns.iter().enumerate() {
        let target = match &names {
            Some(list) => {
                // Result column i maps to the named target column.
                let name = list.get(ci).ok_or_else(|| {
                    TaqlError::Eval("INSERT: result has more columns than names".into())
                })?;
                wt.desc()
                    .columns
                    .iter()
                    .position(|c| c.name == *name)
                    .ok_or_else(|| TaqlError::Eval(format!("INSERT: no column {name}")))?
            }
            None => {
                if ci >= ncols {
                    return Err(TaqlError::Eval(
                        "INSERT: result has more columns than the table".into(),
                    ));
                }
                ci
            }
        };
        // Expression results are coerced to the target column type
        // (e.g. an `Int64` INTO an `Int` column), like UPDATE.
        let dt = wt.desc().columns[target].data_type;
        let coerced: Vec<RecordValue> = col.iter().map(|v| coerce_to(dt, v.clone())).collect();
        wt.putcol(target, start as u64, &coerced)?;
    }
    wt.flush()?;
    Ok(empty_table())
}

/// `DROPTABLE table [, table ...]` — delete the table directory.
fn run_droptable(
    query: &str,
    _tables: &[&Table],
    touched: &mut Vec<std::path::PathBuf>,
) -> TResult<TaqlTable> {
    let toks = tokenize(query)?;
    let mut p = Parser { toks, pos: 0 };
    if !p.expect_ident()?.eq_ignore_ascii_case("droptable") {
        return Err(TaqlError::Parse("expected DROPTABLE".into()));
    }
    loop {
        let tr = parse_table_ref(&mut p)?;
        let path = match tr {
            TableRef::Path(path) => path,
            _ => {
                return Err(TaqlError::Eval(
                    "DROPTABLE requires an on-disk table path".into(),
                ));
            }
        };
        touched.push(path.clone().into());
        std::fs::remove_dir_all(&path)
            .map_err(|e| TaqlError::Eval(format!("DROPTABLE {path}: {e}")))?;
        if matches!(p.peek(), Some(Tok::Op(o)) if o == ",") {
            p.next();
        } else {
            break;
        }
    }
    Ok(empty_table())
}

/// `ALTER TABLE table (ADD [COLUMN] colspec | DROP [COLUMN] col |
/// RENAME COLUMN a TO b | SET keyword = value | REMOVE keyword)`.
fn run_alter(
    query: &str,
    tables: &[&Table],
    touched: &mut Vec<std::path::PathBuf>,
) -> TResult<TaqlTable> {
    let toks = tokenize(query)?;
    let mut p = Parser { toks, pos: 0 };
    if !p.expect_ident()?.eq_ignore_ascii_case("alter") {
        return Err(TaqlError::Parse("expected ALTER".into()));
    }
    if !p.expect_ident()?.eq_ignore_ascii_case("table") {
        return Err(TaqlError::Parse("expected TABLE after ALTER".into()));
    }
    let tr = parse_table_ref(&mut p)?;

    enum Op {
        Add(crate::tabledesc::ColumnDesc),
        Drop(String),
        Rename(String, String),
        SetKeyword(String, Expr),
        RemoveKeyword(String),
    }
    let mut ops = Vec::new();
    while let Some(Tok::Ident(w)) = p.peek() {
        let kw = w.to_ascii_uppercase();
        match kw.as_str() {
            "ADD" => {
                p.next();
                if let Some(Tok::Ident(w)) = p.peek() {
                    if w.eq_ignore_ascii_case("column") {
                        p.next();
                    }
                }
                ops.push(Op::Add(parse_column_spec(&mut p)?));
            }
            "DROP" => {
                p.next();
                if let Some(Tok::Ident(w)) = p.peek() {
                    if w.eq_ignore_ascii_case("column") {
                        p.next();
                    }
                }
                ops.push(Op::Drop(p.expect_ident()?));
            }
            "RENAME" => {
                p.next();
                if let Some(Tok::Ident(w)) = p.peek() {
                    if w.eq_ignore_ascii_case("column") {
                        p.next();
                    }
                }
                let from = p.expect_ident()?;
                let to = match p.peek() {
                    Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("to") => {
                        p.next();
                        p.expect_ident()?
                    }
                    _ => {
                        p.expect_op("=")?;
                        p.expect_ident()?
                    }
                };
                ops.push(Op::Rename(from, to));
            }
            "SET" | "CREATE" => {
                p.next();
                let name = p.expect_ident()?;
                p.expect_op("=")?;
                ops.push(Op::SetKeyword(name, p.parse_expr()?));
            }
            "REMOVE" | "DELETE" => {
                p.next();
                if let Some(Tok::Ident(w)) = p.peek() {
                    if w.eq_ignore_ascii_case("keyword") {
                        p.next();
                    }
                }
                ops.push(Op::RemoveKeyword(p.expect_ident()?));
            }
            _ => break,
        }
    }
    if ops.is_empty() {
        return Err(TaqlError::Parse(
            "ALTER TABLE requires ADD/DROP/RENAME/SET/REMOVE sub-commands".into(),
        ));
    }

    for op in ops {
        let mut owned: Option<Table> = None;
        let t = resolve_table(&tr, tables, &mut owned)?;
        touched.push(t.name().into());
        let mut wt = writable_edit(t)?;
        match op {
            Op::Add(cd) => {
                if wt.desc().columns.iter().any(|c| c.name == cd.name) {
                    return Err(TaqlError::Eval(format!(
                        "ALTER ADD: column {} already exists",
                        cd.name
                    )));
                }
                wt.addcol(cd);
            }
            Op::Drop(name) => {
                let idx = wt
                    .desc()
                    .columns
                    .iter()
                    .position(|c| c.name == name)
                    .ok_or_else(|| TaqlError::Eval(format!("ALTER DROP: no column {name}")))?;
                wt.removecol(idx);
            }
            Op::Rename(from, to) => wt.renamecol(&from, &to)?,
            Op::SetKeyword(name, expr) => {
                // The keyword value is a literal in the statement.
                wt.putkeyword(&name, literal_value(&expr)?);
            }
            Op::RemoveKeyword(name) => wt.removekeyword(&name),
        }
        wt.flush()?;
    }
    Ok(empty_table())
}

/// A statement-level literal (`SET keyword = <value>`).
fn literal_value(e: &Expr) -> TResult<RecordValue> {
    match e {
        Expr::Int(n) => Ok(RecordValue::Int64(*n)),
        Expr::Float(f) => Ok(RecordValue::Double(*f)),
        Expr::Str(s) => Ok(RecordValue::String(s.clone())),
        _ => Err(TaqlError::Eval(
            "expected a literal value in SET keyword".into(),
        )),
    }
}

fn num_as_i64(v: &RecordValue) -> i64 {
    match v {
        RecordValue::Bool(b) => i64::from(*b),
        RecordValue::UChar(u) => i64::from(*u),
        RecordValue::Short(i) => i64::from(*i),
        RecordValue::UShort(u) => i64::from(*u),
        RecordValue::Int(i) => i64::from(*i),
        RecordValue::UInt(u) => i64::from(*u),
        RecordValue::Int64(i) => *i,
        RecordValue::Float(f) => *f as i64,
        RecordValue::Double(d) => *d as i64,
        _ => 0,
    }
}

fn num_as_f64(v: &RecordValue) -> f64 {
    match v {
        RecordValue::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        RecordValue::UChar(u) => f64::from(*u),
        RecordValue::Short(i) => f64::from(*i),
        RecordValue::UShort(u) => f64::from(*u),
        RecordValue::Int(i) => f64::from(*i),
        RecordValue::UInt(u) => f64::from(*u),
        RecordValue::Int64(i) => *i as f64,
        RecordValue::Float(f) => f64::from(*f),
        RecordValue::Double(d) => *d,
        _ => 0.0,
    }
}

/// Coerce an expression result to a column's declared type (casacure `putcol`
/// casts to the column type). Arithmetic in TaQL yields `Int64`/`Double`
/// even for `Int`/`Float` columns, so without this the values would be
/// written as zeros by the column-type encoder.
fn coerce_to(dt: DataType, v: RecordValue) -> RecordValue {
    use RecordValue as RV;
    match dt {
        DataType::Bool => RV::Bool(match v {
            RV::Bool(b) => b,
            other => num_as_i64(&other) != 0,
        }),
        DataType::UChar => RV::UChar(num_as_i64(&v) as u8),
        DataType::Short => RV::Short(num_as_i64(&v) as i16),
        DataType::UShort => RV::UShort(num_as_i64(&v) as u16),
        DataType::Int => RV::Int(num_as_i64(&v) as i32),
        DataType::UInt => RV::UInt(num_as_i64(&v) as u32),
        DataType::Int64 => RV::Int64(num_as_i64(&v)),
        DataType::Float => RV::Float(num_as_f64(&v) as f32),
        DataType::Double => RV::Double(num_as_f64(&v)),
        DataType::Complex => match v {
            RV::Complex(re, im) => RV::Complex(re, im),
            RV::DComplex(re, im) => RV::Complex(re as f32, im as f32),
            other => {
                let n = num_as_f64(&other) as f32;
                RV::Complex(n, 0.0)
            }
        },
        DataType::DComplex => match v {
            RV::Complex(re, im) => RV::DComplex(f64::from(re), f64::from(im)),
            RV::DComplex(re, im) => RV::DComplex(re, im),
            other => {
                let n = num_as_f64(&other);
                RV::DComplex(n, 0.0)
            }
        },
        DataType::String => match v {
            RV::String(s) => RV::String(s),
            other => RV::String(stringify(&cell_value(&other))),
        },
        _ => v,
    }
}

/// `SHOW TABLE <path>` — one row per column; `SHOW`/`HELP` — a summary.
fn run_show(query: &str, tables: &[&Table]) -> TResult<TaqlTable> {
    let toks = tokenize(query)?;
    let mut p = Parser { toks, pos: 0 };
    p.expect_ident()?; // SHOW or HELP
    if matches!(p.peek(), Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("table")) {
        p.next();
        let tr = parse_table_ref(&mut p)?;
        let mut owned: Option<Table> = None;
        let t = resolve_table(&tr, tables, &mut owned)?;
        let mut out = TaqlTable {
            colnames: vec![
                "name".into(),
                "datatype".into(),
                "ndim".into(),
                "shape".into(),
                "comment".into(),
            ],
            columns: vec![Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        };
        let colnames = t.colnames();
        for (ci, cd) in t.dat.desc.columns.iter().enumerate() {
            let _ = ci;
            let shape = match &cd.shape {
                Some(s) => format!("{s:?}"),
                None => "scalar".into(),
            };
            out.columns[0].push(RecordValue::String(cd.name.clone()));
            out.columns[1].push(RecordValue::String(
                crate::casa_value_type(cd.data_type).into(),
            ));
            out.columns[2].push(RecordValue::Int64(i64::from(cd.ndim)));
            out.columns[3].push(RecordValue::String(shape));
            out.columns[4].push(RecordValue::String(cd.comment.clone()));
        }
        let _ = colnames;
        return Ok(out);
    }
    // SHOW/HELP: summary of the statement keywords.
    Ok(TaqlTable {
        colnames: vec!["help".into()],
        columns: vec![vec![RecordValue::String(
            "TaQL statements: SELECT [INTO], CREATE TABLE, COUNT, UPDATE, DELETE, \
             INSERT INTO, DROPTABLE, ALTER TABLE (ADD/DROP/RENAME COLUMN, SET/REMOVE \
             keyword), SHOW TABLE, CALC"
                .into(),
        )]],
    })
}

/// `CALC <expr> [, <expr>...] [FROM table]` — evaluate at row 0 of a table.
fn run_calc(query: &str, tables: &[&Table]) -> TResult<TaqlTable> {
    let toks = tokenize(query)?;
    let mut p = Parser { toks, pos: 0 };
    if !p.expect_ident()?.eq_ignore_ascii_case("calc") {
        return Err(TaqlError::Parse("expected CALC".into()));
    }
    let mut exprs = Vec::new();
    loop {
        exprs.push(p.parse_expr()?);
        if matches!(p.peek(), Some(Tok::Op(o)) if o == ",") {
            p.next();
        } else {
            break;
        }
    }
    let mut owned: Option<Table> = None;
    let t: &Table = match p.peek() {
        Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("from") => {
            p.next();
            let tr = parse_table_ref(&mut p)?;
            resolve_table(&tr, tables, &mut owned)?
        }
        _ => {
            return Err(TaqlError::Eval(
                "CALC requires FROM <table> to evaluate in a row context".into(),
            ));
        }
    };
    let ctx = row_ctx(t, tables);
    let mut out = TaqlTable {
        colnames: (0..exprs.len()).map(|i| format!("col{i}")).collect(),
        columns: vec![Vec::new(); exprs.len()],
    };
    for (i, e) in exprs.iter().enumerate() {
        let v = ctx.eval_row(e, 0)?;
        out.columns[i].push(v.to_record());
    }
    Ok(out)
}

/// `SELECT ... INTO <path> FROM ...`: run the query and persist the result
/// to a new table; the column types are inferred from the result values.
fn run_select_into(path: &str, sel: &Select, tables: &[&Table]) -> TResult<std::path::PathBuf> {
    let result = run_select(sel, tables)?;
    let desc = desc_of_result(&result.colnames, &result.columns);
    let values: Vec<Vec<RecordValue>> = result.columns.clone();
    crate::create_table(std::path::Path::new(path), &desc, &values)
        .map_err(|e| TaqlError::Eval(format!("SELECT INTO {path}: {e}")))?;
    Ok(path.into())
}

/// Infer a writable `TableDesc` from a query-result column set: one scalar
/// column per value type, array columns when the result holds arrays.
fn desc_of_result(
    colnames: &[String],
    columns: &[Vec<RecordValue>],
) -> crate::tabledesc::TableDesc {
    fn scalar(name: &str, dt: DataType, def: RecordValue) -> crate::tabledesc::ColumnDesc {
        crate::tabledesc::ColumnDesc {
            name: name.into(),
            comment: String::new(),
            data_type: dt,
            data_manager_type: "StandardStMan".into(),
            data_manager_group: "StandardStMan".into(),
            options: 0,
            ndim: -1,
            shape: None,
            max_length: 0,
            keywords: crate::record::TableRecord {
                desc: Default::default(),
                record_type: 0,
                values: Vec::new(),
            },
            kind: crate::tabledesc::ColumnKind::Scalar(def),
        }
    }
    let mut desc = crate::tabledesc::TableDesc {
        name: String::new(),
        version: String::new(),
        comment: String::new(),
        keywords: crate::record::TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        },
        private_keywords: crate::record::TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        },
        columns: Vec::new(),
    };
    for (ci, name) in colnames.iter().enumerate() {
        let sample = columns
            .get(ci)
            .and_then(|c| c.first())
            .cloned()
            .unwrap_or(RecordValue::Int64(0));
        let cd = match &sample {
            RecordValue::Bool(_) => scalar(name, DataType::Bool, RecordValue::Bool(false)),
            RecordValue::Int(_) => scalar(name, DataType::Int, RecordValue::Int(0)),
            RecordValue::Int64(_) => scalar(name, DataType::Int64, RecordValue::Int64(0)),
            RecordValue::Float(_) => scalar(name, DataType::Float, RecordValue::Float(0.0)),
            RecordValue::Double(_) => scalar(name, DataType::Double, RecordValue::Double(0.0)),
            RecordValue::String(_) => {
                scalar(name, DataType::String, RecordValue::String(String::new()))
            }
            RecordValue::Array(a) => {
                let (elt, def) = match &a.data {
                    ArrayData::Bool(_) => (DataType::Bool, ArrayData::Bool(vec![])),
                    ArrayData::UChar(_) => (DataType::UChar, ArrayData::UChar(vec![])),
                    ArrayData::UShort(_) => (DataType::UShort, ArrayData::UShort(vec![])),
                    ArrayData::Short(_) => (DataType::Short, ArrayData::Short(vec![])),
                    ArrayData::Int(_) => (DataType::Int, ArrayData::Int(vec![])),
                    ArrayData::UInt(_) => (DataType::UInt, ArrayData::UInt(vec![])),
                    ArrayData::Int64(_) => (DataType::Int64, ArrayData::Int64(vec![])),
                    ArrayData::Float(_) => (DataType::Float, ArrayData::Float(vec![])),
                    ArrayData::Double(_) => (DataType::Double, ArrayData::Double(vec![])),
                    ArrayData::Complex(_) => (DataType::Complex, ArrayData::Complex(vec![])),
                    ArrayData::DComplex(_) => (DataType::DComplex, ArrayData::DComplex(vec![])),
                    ArrayData::String(_) => (DataType::String, ArrayData::String(vec![])),
                };
                let _ = def;
                crate::tabledesc::ColumnDesc {
                    name: name.into(),
                    comment: String::new(),
                    data_type: elt,
                    data_manager_type: "StandardStMan".into(),
                    data_manager_group: "StandardStMan".into(),
                    options: 0,
                    ndim: a.shape.len() as i32,
                    shape: None,
                    max_length: 0,
                    keywords: crate::record::TableRecord {
                        desc: Default::default(),
                        record_type: 0,
                        values: Vec::new(),
                    },
                    kind: crate::tabledesc::ColumnKind::Array,
                }
            }
            other => {
                let _ = other;
                scalar(name, DataType::String, RecordValue::String(String::new()))
            }
        };
        desc.columns.push(cd);
    }
    desc
}

fn run_select(sel: &Select, tables: &[&Table]) -> TResult<TaqlTable> {
    // A `'path'` FROM clause opens the table for the duration of the query.
    let owned: Option<Table>;
    let table: &Table = match &sel.table {
        TableRef::Table(n) => tables
            .get(n.saturating_sub(1))
            .copied()
            .ok_or(TaqlError::NoSuchTable(*n))?,
        TableRef::Path(path) => {
            owned = Some(
                Table::open(path, false)
                    .map_err(|e| TaqlError::Eval(format!("cannot open table {path}: {e}")))?,
            );
            owned.as_ref().unwrap()
        }
    };
    let colidx: HashMap<String, usize> = table
        .colnames()
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();

    let columns = RefCell::new(HashMap::<usize, Vec<TqValue>>::new());
    let ctx = EvalCtx {
        table,
        colidx,
        columns,
        tables,
    };

    if !sel.groupby.is_empty() {
        return run_group(&ctx, sel);
    }

    // Row-mode: filter, sort, project.
    let mut rows: Vec<i64> = (0..table.nrows() as i64).collect();
    if let Some(where_) = &sel.where_ {
        rows.retain(|&r| ctx.eval_row(where_, r).map(|v| v.truthy()).unwrap_or(false));
    }
    if !sel.orderby.is_empty() {
        sort_rows(&ctx, &mut rows, &sel.orderby)?;
    }
    if let Some(off) = sel.offset {
        rows.drain(..(off.min(rows.len() as i64).max(0)) as usize);
    }
    if let Some(limit) = sel.limit {
        rows.truncate(limit.max(0) as usize);
    }
    project_rows(&ctx, sel, &rows)
}

/// The group pipeline: rows grouped by the group-by expressions, in
/// first-appearance order.
fn run_group(ctx: &EvalCtx<'_>, sel: &Select) -> TResult<TaqlTable> {
    let table = ctx.table;
    // Group membership: key = the evaluated group-by values at a row.
    let mut rows: Vec<i64> = (0..table.nrows() as i64).collect();
    if let Some(where_) = &sel.where_ {
        rows.retain(|&r| ctx.eval_row(where_, r).map(|v| v.truthy()).unwrap_or(false));
    }
    let mut key_to_index: HashMap<Vec<TqValue>, usize> = HashMap::new();
    let mut groups: Vec<Vec<i64>> = Vec::new();
    for r in rows {
        let mut key = Vec::with_capacity(sel.groupby.len());
        for g in &sel.groupby {
            key.push(ctx.eval_row(g, r)?);
        }
        let gi = match key_to_index.get(&key) {
            Some(&i) => i,
            None => {
                key_to_index.insert(key, groups.len());
                groups.push(Vec::new());
                groups.len() - 1
            }
        };
        groups[gi].push(r);
    }
    if let Some(having) = &sel.having {
        groups.retain(|g| {
            ctx.eval_group(having, g)
                .map(|v| v.truthy())
                .unwrap_or(false)
        });
    }
    if !sel.orderby.is_empty() {
        sort_groups(ctx, &mut groups, &sel.orderby)?;
    }
    if let Some(off) = sel.offset {
        groups.drain(..(off.min(groups.len() as i64).max(0)) as usize);
    }
    if let Some(limit) = sel.limit {
        groups.truncate(limit.max(0) as usize);
    }
    if sel.unique {
        // A duplicate group key yields one output row already; nothing extra.
    }

    // Resolve and evaluate the select columns in group context.
    let cols: Vec<(Expr, String)> = match &sel.columns {
        None => table
            .colnames()
            .into_iter()
            .map(|n| (Expr::Name(n.clone()), n))
            .collect(),
        Some(list) => list
            .iter()
            .map(|(e, a)| {
                let name = a.clone().unwrap_or_else(|| expr_name(e, table));
                (e.clone(), name)
            })
            .collect(),
    };
    let mut out_cols: Vec<Vec<RecordValue>> = Vec::with_capacity(cols.len());
    for (e, _name) in &cols {
        let mut vals = Vec::with_capacity(groups.len());
        for group in &groups {
            let v = match e {
                // Bare columns in a grouped query: the group's members share
                // the value only when singleton; keep the first cell verbatim.
                Expr::Name(n) if !n.eq_ignore_ascii_case("rowid") && group.len() == 1 => {
                    ctx.cell(n, group[0])?
                }
                _ => ctx.eval_group(e, group)?.to_record(),
            };
            vals.push(v);
        }
        out_cols.push(vals);
    }
    Ok(TaqlTable {
        colnames: cols.into_iter().map(|(_, n)| n).collect(),
        columns: out_cols,
    })
}

/// Default (unaliased) output name of a select expression: the column name
/// for a bare column reference, otherwise an empty computed-column name.
fn expr_name(e: &Expr, _table: &Table) -> String {
    match e {
        Expr::Name(n) => n.clone(),
        _ => "col0".to_string(),
    }
}

fn project_rows(ctx: &EvalCtx<'_>, sel: &Select, rows: &[i64]) -> TResult<TaqlTable> {
    let table = ctx.table;
    let cols: Vec<(Expr, String)> = match &sel.columns {
        None => table
            .colnames()
            .into_iter()
            .map(|n| (Expr::Name(n.clone()), n))
            .collect(),
        Some(list) => {
            let mut unnamed = 0usize;
            list.iter()
                .map(|(e, a)| {
                    let name = match a {
                        Some(a) => a.clone(),
                        None => match e {
                            Expr::Name(n) => n.clone(),
                            _ => {
                                let n = format!("col{unnamed}");
                                unnamed += 1;
                                n
                            }
                        },
                    };
                    (e.clone(), name)
                })
                .collect()
        }
    };
    let mut out_cols: Vec<Vec<RecordValue>> = Vec::with_capacity(cols.len());
    for (e, _name) in &cols {
        let mut vals = Vec::with_capacity(rows.len());
        for &r in rows {
            let v = match e {
                Expr::Name(n) if !n.eq_ignore_ascii_case("rowid") => ctx.cell(n, r)?,
                _ => ctx.eval_row(e, r)?.to_record(),
            };
            vals.push(v);
        }
        out_cols.push(vals);
    }
    let mut out = TaqlTable {
        colnames: cols.into_iter().map(|(_, n)| n).collect(),
        columns: out_cols,
    };
    if sel.unique {
        out = unique_rows(out);
    }
    Ok(out)
}

/// `SELECT UNIQUE`: drop duplicate output rows, keeping first occurrence.
fn unique_rows(t: TaqlTable) -> TaqlTable {
    let mut seen = std::collections::HashSet::new();
    let mut keep = Vec::new();
    for r in 0..t.nrows() {
        let mut key = Vec::new();
        for c in &t.columns {
            key.push(c[r].to_json_string());
        }
        if seen.insert(key) {
            keep.push(r);
        }
    }
    let mut out = TaqlTable {
        colnames: t.colnames,
        columns: Vec::with_capacity(t.columns.len()),
    };
    for col in t.columns {
        let mut c = Vec::with_capacity(keep.len());
        for &r in &keep {
            c.push(col[r].clone());
        }
        out.columns.push(c);
    }
    out
}

// ---------------------------------------------------------------------------
// Evaluation contexts
// ---------------------------------------------------------------------------

struct EvalCtx<'a> {
    table: &'a Table,
    colidx: HashMap<String, usize>,
    columns: RefCell<HashMap<usize, Vec<TqValue>>>,
    tables: &'a [&'a Table],
}

impl<'a> EvalCtx<'a> {
    fn column(&self, name: &str) -> TResult<Vec<TqValue>> {
        let idx = *self
            .colidx
            .get(name)
            .ok_or_else(|| TaqlError::NoSuchAttribute {
                what: "table".to_string(),
                field: name.to_string(),
            })?;
        if let Some(cached) = self.columns.borrow().get(&idx) {
            return Ok(cached.clone());
        }
        let cells = self.table.getcol(idx, 0, self.table.nrows())?;
        let vals: Vec<TqValue> = cells.iter().map(cell_value).collect();
        self.columns.borrow_mut().insert(idx, vals.clone());
        Ok(vals)
    }

    fn column_value(&self, name: &str, row: i64) -> TResult<TqValue> {
        Ok(self
            .column(name)?
            .get(row as usize)
            .cloned()
            .unwrap_or(TqValue::Int(0)))
    }

    /// The raw cell for a bare column reference, cloned verbatim from the
    /// source table. Array / record cells (complexes, multidim shapes) do not
    /// survive the `TqValue` round-trip, so projections keep them intact.
    fn cell(&self, name: &str, row: i64) -> TResult<RecordValue> {
        let idx = *self
            .colidx
            .get(name)
            .ok_or_else(|| TaqlError::NoSuchAttribute {
                what: "table".to_string(),
                field: name.to_string(),
            })?;
        self.table
            .getcell(idx, row as u64)
            .map_err(|e| TaqlError::Eval(format!("getcell {name}[{row}]: {e}")))
    }

    /// Evaluate in a row context.
    fn eval_row(&self, e: &Expr, row: i64) -> TResult<TqValue> {
        match e {
            Expr::Int(n) => Ok(TqValue::Int(*n)),
            Expr::Float(f) => Ok(TqValue::Float(*f)),
            Expr::Str(s) => Ok(TqValue::Str(s.clone())),
            Expr::Name(name) => {
                if name.eq_ignore_ascii_case("rowid") {
                    return Ok(TqValue::Int(row));
                }
                self.column_value(name, row)
            }
            Expr::Call(name, args) => self.call_row(name, args, row),
            Expr::Unary(op, inner) => {
                let v = self.eval_row(inner, row)?;
                Ok(match op {
                    UnOp::Neg => match v {
                        TqValue::Int(i) => TqValue::Int(-i),
                        TqValue::Float(f) => TqValue::Float(-f),
                        other => {
                            return Err(TaqlError::Eval(format!("cannot negate {other:?}")));
                        }
                    },
                    UnOp::Not => TqValue::Bool(!v.truthy()),
                })
            }
            Expr::Binary(op, l, r) => {
                if matches!(
                    op,
                    BinOp::Like { .. } | BinOp::In { .. } | BinOp::Regex { .. }
                ) {
                    self.like_in(op, l, r, &[row], false)
                } else {
                    self.eval_binary(op, l, r, &[row], false)
                }
            }
            Expr::Index(base, idx) => {
                let base_v = self.eval_row(base, row)?;
                let idx_v = self.eval_row(idx, row)?;
                index_value(base_v, idx_v)
            }
            Expr::Subquery(sel) => {
                let sub = run_select(sel, self.tables)?;
                Ok(TqValue::Subtable(std::rc::Rc::new(sub)))
            }
            Expr::Set(items) => Ok(TqValue::Arr(
                items
                    .iter()
                    .map(|e| self.eval_row(e, row))
                    .collect::<TResult<_>>()?,
            )),
        }
    }

    /// Evaluate in a group context: `group` is the group's row members.
    fn eval_group(&self, e: &Expr, group: &[i64]) -> TResult<TqValue> {
        match e {
            Expr::Int(n) => Ok(TqValue::Int(*n)),
            Expr::Float(f) => Ok(TqValue::Float(*f)),
            Expr::Str(s) => Ok(TqValue::Str(s.clone())),
            Expr::Name(name) => self.group_column_value(name, group),
            Expr::Call(name, args) => self.call_group(name, args, group),
            Expr::Unary(op, inner) => {
                let v = self.eval_group(inner, group)?;
                Ok(match op {
                    UnOp::Neg => match v {
                        TqValue::Int(i) => TqValue::Int(-i),
                        TqValue::Float(f) => TqValue::Float(-f),
                        other => {
                            return Err(TaqlError::Eval(format!("cannot negate {other:?}")));
                        }
                    },
                    UnOp::Not => TqValue::Bool(!v.truthy()),
                })
            }
            Expr::Binary(op, l, r) => {
                if matches!(
                    op,
                    BinOp::Like { .. } | BinOp::In { .. } | BinOp::Regex { .. }
                ) {
                    self.like_in(op, l, r, group, true)
                } else {
                    self.eval_binary(op, l, r, group, true)
                }
            }
            Expr::Index(base, idx) => {
                let base_v = self.eval_group(base, group)?;
                let idx_v = self.eval_group(idx, group)?;
                index_value(base_v, idx_v)
            }
            Expr::Subquery(sel) => {
                let sub = run_select(sel, self.tables)?;
                Ok(TqValue::Subtable(std::rc::Rc::new(sub)))
            }
            Expr::Set(items) => Ok(TqValue::Arr(
                items
                    .iter()
                    .map(|e| self.eval_group(e, group))
                    .collect::<TResult<_>>()?,
            )),
        }
    }

    /// A plain column in a group context: constant value of the group-by
    /// columns (at the first row of the group).
    fn group_column_value(&self, name: &str, group: &[i64]) -> TResult<TqValue> {
        if name.eq_ignore_ascii_case("rowid") || name.eq_ignore_ascii_case("growid") {
            return Err(TaqlError::Eval(format!(
                "{name} is only available as {name}()"
            )));
        }
        let col = self.column(name)?;
        let rep = group.first().copied().unwrap_or(0);
        Ok(col.get(rep as usize).cloned().unwrap_or(TqValue::Int(0)))
    }

    fn call_row(&self, name: &str, args: &[Expr], row: i64) -> TResult<TqValue> {
        let upper = name.to_ascii_uppercase();
        match upper.as_str() {
            // Row-number functions (no arguments).
            "ROWID" | "ROWNUMBER" | "ROWNR" => return Ok(TqValue::Int(row)),
            // Presence probes that need the table context.
            "ISCOLUMN" | "ISKEYWORD" => {
                let key = match args.first() {
                    Some(Expr::Str(s)) => s.clone(),
                    Some(Expr::Name(s)) => s.clone(),
                    _ => {
                        return Err(TaqlError::Eval(format!("{name} expects a name argument")));
                    }
                };
                let present = if upper == "ISCOLUMN" {
                    self.colidx.contains_key(key.as_str())
                } else {
                    let kws = crate::record::parse_json_record(&self.table.getkeywords())
                        .map_err(|e| TaqlError::Eval(format!("iskeyword: {e}")))?;
                    kws.get(&key).is_some()
                };
                return Ok(TqValue::Bool(present));
            }
            _ => {}
        }
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            vals.push(self.eval_row(a, row)?);
        }
        scalar_func(&upper, &vals)
    }

    fn call_group(&self, name: &str, args: &[Expr], group: &[i64]) -> TResult<TqValue> {
        let upper = name.to_ascii_uppercase();
        match upper.as_str() {
            "GROWID" => Ok(TqValue::Arr(
                group.iter().map(|&r| TqValue::Int(r)).collect(),
            )),
            "GCOUNT" => Ok(TqValue::Int(group.len() as i64)),
            "GAGGR" => {
                let colname = match &args[0] {
                    Expr::Name(n) => n.clone(),
                    other => {
                        return Err(TaqlError::Eval(format!(
                            "GAGGR expects a column name, got {other:?}"
                        )));
                    }
                };
                let col = self.column(&colname)?;
                let items: Vec<TqValue> = group
                    .iter()
                    .map(|&r| col.get(r as usize).cloned().unwrap_or(TqValue::Int(0)))
                    .collect();
                Ok(TqValue::Arr(items))
            }
            // Typed group aggregates over a column. GFRACTILE takes a second
            // (fraction) argument evaluated in the group's row context.
            fname @ ("GMIN" | "GMAX" | "GSUM" | "GSUMSQR" | "GPRODUCT" | "GMEAN" | "GAVG"
            | "GVARIANCE" | "GSTDDEV" | "GRMS" | "GMEDIAN" | "GFRACTILE" | "GFIRST"
            | "GLAST" | "GANY" | "GALL" | "GNTRUE" | "GNFALSE") => {
                let colname = match &args[0] {
                    Expr::Name(n) => n.clone(),
                    other => {
                        return Err(TaqlError::Eval(format!(
                            "{fname} expects a column name, got {other:?}"
                        )));
                    }
                };
                let col = self.column(&colname)?;
                let items: Vec<TqValue> = group
                    .iter()
                    .map(|&r| col.get(r as usize).cloned().unwrap_or(TqValue::Int(0)))
                    .collect();
                let frac = if upper == "GFRACTILE" {
                    Some(if args.len() >= 2 {
                        self.eval_row(&args[1], group.first().copied().unwrap_or(0))?
                    } else {
                        TqValue::Float(0.5)
                    })
                } else {
                    None
                };
                group_agg(fname, &items, frac.as_ref())
            }
            _ => {
                // Row-context-only functions keep their row behaviour (the
                // group's first member row).
                match upper.as_str() {
                    "ROWID" | "ROWNUMBER" | "ROWNR" | "ISCOLUMN" | "ISKEYWORD" => {
                        return self.call_row(name, args, group.first().copied().unwrap_or(0));
                    }
                    _ => {}
                }
                // Evaluate the arguments in group context so nested group
                // functions (e.g. `nelements(gaggr(WHAT))`) work.
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    vals.push(self.eval_group(a, group)?);
                }
                scalar_func(&upper, &vals)
            }
        }
    }

    fn eval_binary(
        &self,
        op: &BinOp,
        l: &Expr,
        r: &Expr,
        rows: &[i64],
        group: bool,
    ) -> TResult<TqValue> {
        let lv = if group {
            self.eval_group(l, rows)?
        } else {
            self.eval_row(l, rows[0])?
        };
        let rv = if group {
            self.eval_group(r, rows)?
        } else {
            self.eval_row(r, rows[0])?
        };
        binary_value(op, &lv, &rv)
    }

    /// `LIKE` / `ILIKE` / `IN` evaluation (handled before `binary_value`
    /// because the `IN` right operand is an unevaluated `Expr::Set`).
    fn like_in(
        &self,
        op: &BinOp,
        l: &Expr,
        r: &Expr,
        rows: &[i64],
        group: bool,
    ) -> TResult<TqValue> {
        let ev = |e: &Expr| -> TResult<TqValue> {
            if group {
                self.eval_group(e, rows)
            } else {
                self.eval_row(e, rows[0])
            }
        };
        match op {
            BinOp::Like {
                case_insensitive,
                negated,
            } => {
                let text = stringify(&ev(l)?);
                let pat = match ev(r)? {
                    TqValue::Str(s) => s,
                    other => {
                        return Err(TaqlError::Eval(format!(
                            "LIKE pattern must be a string, got {other:?}"
                        )));
                    }
                };
                let matched = sql_like(&text, &pat, *case_insensitive);
                Ok(TqValue::Bool(matched != *negated))
            }
            BinOp::In { negated } => {
                let lhs = ev(l)?;
                let items: Vec<TqValue> = match r {
                    Expr::Set(items) => items.iter().map(ev).collect::<TResult<_>>()?,
                    other => {
                        return Err(TaqlError::Eval(format!(
                            "IN needs a set of values, got {other:?}"
                        )));
                    }
                };
                let found = items
                    .iter()
                    .any(|x| compare(x, &lhs) == std::cmp::Ordering::Equal);
                Ok(TqValue::Bool(found != *negated))
            }
            BinOp::Regex { negated } => {
                // `str ~ regex('pat')` (a plain string RHS is a bare regex).
                let text = stringify(&ev(l)?);
                let pat = match ev(r)? {
                    TqValue::Regex(kind, s) => (kind, s),
                    TqValue::Str(s) => (RegexKind::Regex, s),
                    other => {
                        return Err(TaqlError::Eval(format!(
                            "~ needs a regex/pattern/sqlpattern on the right, got {other:?}"
                        )));
                    }
                };
                let matched = regex_matches(&text, pat.0, &pat.1);
                Ok(TqValue::Bool(matched != *negated))
            }
            _ => unreachable!(),
        }
    }
}

/// casacore `LIKE` pattern match: `%` matches any run, `_` matches one
/// character, `\` escapes the next character.
fn sql_like(text: &str, pat: &str, case_insensitive: bool) -> bool {
    enum Tok {
        Any,
        One,
        Lit(char),
    }
    let (text, pat) = if case_insensitive {
        (text.to_lowercase(), pat.to_lowercase())
    } else {
        (text.to_string(), pat.to_string())
    };
    let t: Vec<char> = text.chars().collect();
    let p: Vec<char> = pat.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < p.len() {
        match p[i] {
            '%' => {
                toks.push(Tok::Any);
                i += 1;
            }
            '_' => {
                toks.push(Tok::One);
                i += 1;
            }
            '\\' if i + 1 < p.len() => {
                toks.push(Tok::Lit(p[i + 1]));
                i += 2;
            }
            '\\' => {
                toks.push(Tok::Lit('\\'));
                i += 1;
            }
            c => {
                toks.push(Tok::Lit(c));
                i += 1;
            }
        }
    }
    let (n, m) = (t.len(), toks.len());
    let mut dp = vec![vec![false; m + 1]; n + 1];
    dp[0][0] = true;
    for j in 1..=m {
        if matches!(toks[j - 1], Tok::Any) {
            dp[0][j] = dp[0][j - 1];
        }
    }
    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = match toks[j - 1] {
                Tok::Any => dp[i][j - 1] || dp[i - 1][j],
                Tok::One => dp[i - 1][j - 1],
                Tok::Lit(c) => t[i - 1] == c && dp[i - 1][j - 1],
            };
        }
    }
    dp[n][m]
}

fn cell_value(c: &RecordValue) -> TqValue {
    match c {
        RecordValue::Bool(b) => TqValue::Bool(*b),
        RecordValue::UChar(u) => TqValue::Int(i64::from(*u)),
        RecordValue::Short(i) => TqValue::Int(i64::from(*i)),
        RecordValue::UShort(u) => TqValue::Int(i64::from(*u)),
        RecordValue::Int(i) => TqValue::Int(i64::from(*i)),
        RecordValue::UInt(u) => TqValue::Int(i64::from(*u)),
        RecordValue::Int64(i) => TqValue::Int(*i),
        RecordValue::Float(f) => TqValue::Float(f64::from(*f)),
        RecordValue::Double(d) => TqValue::Float(*d),
        RecordValue::Complex(re, im) => TqValue::Arr(vec![
            TqValue::Float(f64::from(*re)),
            TqValue::Float(f64::from(*im)),
        ]),
        RecordValue::DComplex(re, im) => {
            TqValue::Arr(vec![TqValue::Float(*re), TqValue::Float(*im)])
        }
        RecordValue::String(s) => TqValue::Str(s.clone()),
        RecordValue::Table(s) => TqValue::Str(s.clone()),
        RecordValue::Record(r) => TqValue::Str(r.to_json_string()),
        RecordValue::Array(a) => TqValue::Arr(a.elements().iter().map(cell_value).collect()),
    }
}

/// Exact integer view of a value (`None` for floats / strings / arrays).
fn as_i(v: &TqValue) -> Option<i64> {
    match v {
        TqValue::Int(i) => Some(*i),
        TqValue::Bool(b) => Some(i64::from(*b)),
        _ => None,
    }
}

/// Numeric view of a value (`None` for strings / arrays).
fn as_f(v: &TqValue) -> Option<f64> {
    match v {
        TqValue::Int(i) => Some(*i as f64),
        TqValue::Float(f) => Some(*f),
        TqValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// A value as its flat element list (scalars are one-element lists).
fn flat_elements(v: &TqValue) -> Vec<&TqValue> {
    match v {
        TqValue::Arr(items) => items.iter().collect(),
        other => vec![other],
    }
}

/// A complex value as (re, im): our complex scalars are 2-element arrays.
fn complex_parts(v: &TqValue) -> (f64, f64) {
    match v {
        TqValue::Arr(items) if items.len() == 2 => {
            let re = as_f(&items[0]).unwrap_or(0.0);
            let im = as_f(&items[1]).unwrap_or(0.0);
            (re, im)
        }
        other => (as_f(other).unwrap_or(0.0), 0.0),
    }
}

/// Number to string, integer-valued floats rendered without a decimal point.
fn fmt_num(f: f64) -> String {
    if f.is_finite() && f.fract() == 0.0 {
        format!("{}", f as i64)
    } else {
        format!("{f}")
    }
}

fn stringify(v: &TqValue) -> String {
    match v {
        TqValue::Bool(b) => b.to_string(),
        TqValue::Int(i) => i.to_string(),
        TqValue::Float(f) => fmt_num(*f),
        TqValue::Str(s) => s.clone(),
        TqValue::Arr(_) => "[array]".to_string(),
        TqValue::Subtable(_) => "[query]".to_string(),
        TqValue::Regex(_, _) => "[regex]".to_string(),
    }
}

/// Trim `side`: 1 = both, 2 = left, 3 = right. `cutset` empty = whitespace.
fn str_trim(s: &str, side: i32, cutset: &str) -> String {
    let is_cut = |c: char| {
        if cutset.is_empty() {
            c.is_whitespace()
        } else {
            cutset.contains(c)
        }
    };
    let chars: Vec<char> = s.chars().collect();
    let mut start = 0;
    let mut end = chars.len();
    if side != 3 {
        while start < end && is_cut(chars[start]) {
            start += 1;
        }
    }
    if side != 2 {
        while end > start && is_cut(chars[end - 1]) {
            end -= 1;
        }
    }
    chars[start..end].iter().collect()
}

/// `substr(str, start[, end])`: 0-based, `end` inclusive; negative indices
/// count from the end of the string.
fn substr_of(s: &str, start: i64, end: Option<i64>) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len() as i64;
    if n == 0 {
        return String::new();
    }
    let norm = |i: i64| if i < 0 { n + i } else { i };
    let a = norm(start).clamp(0, n - 1) as usize;
    let b = match end {
        Some(e) => norm(e).clamp(0, n - 1) as usize,
        None => (n - 1) as usize,
    };
    if a > b {
        return String::new();
    }
    chars[a..=b].iter().collect()
}

/// Numeric elements of the arguments (scalars join as-is, a single array
/// argument contributes its elements). Returns the values and whether every
/// element is an exact integer (so aggregates can stay `Int`).
fn stat_input(fname: &str, args: &[TqValue]) -> TResult<(Vec<f64>, bool)> {
    let mut out = Vec::new();
    let mut all_i = true;
    for a in args {
        for x in flat_elements(a) {
            if let Some(i) = as_i(x) {
                out.push(i as f64);
            } else if let Some(f) = as_f(x) {
                out.push(f);
                all_i = false;
            } else {
                return Err(TaqlError::Eval(format!("{fname} on non-number {x:?}")));
            }
        }
    }
    Ok((out, all_i))
}

fn sorted(v: Vec<f64>) -> Vec<f64> {
    let mut v = v;
    v.sort_by(f64::total_cmp);
    v
}

fn median_of(v: &[f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn mean_of(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// The `sums`/`mins`/`means`/... plural variants: the stem aggregate applied
/// element-wise across equal-length arrays (scalar arguments broadcast).
fn stats_multi(fname: &str, args: &[TqValue]) -> TResult<TqValue> {
    let stem = &fname[..fname.len() - 1];
    let cols: Vec<Vec<TqValue>> = args
        .iter()
        .map(|a| match a {
            TqValue::Arr(items) => items.clone(),
            other => vec![other.clone()],
        })
        .collect();
    let len = cols.iter().map(|c| c.len()).max().unwrap_or(0);
    for (ci, c) in cols.iter().enumerate() {
        if c.len() != 1 && c.len() != len {
            return Err(TaqlError::Eval(format!(
                "{fname}: argument {} has {} elements, expected {len}",
                ci + 1,
                c.len()
            )));
        }
    }
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let vals: Vec<TqValue> = cols
            .iter()
            .map(|c| {
                if c.len() == 1 {
                    c[0].clone()
                } else {
                    c[i].clone()
                }
            })
            .collect();
        out.push(stats(stem, &vals)?);
    }
    Ok(TqValue::Arr(out))
}

/// Statistical/boolean aggregates over scalar or array arguments. Typing: an
/// all-integer input yields `Int` for the count/sum-family results; the
/// variance/median/rms family is always `Float` (matching casacore).
fn stats(fname: &str, args: &[TqValue]) -> TResult<TqValue> {
    // Boolean family over the raw values (strings/numbers all count as
    // truthy/non-zero).
    let flatten: Vec<&TqValue> = args.iter().flat_map(flat_elements).collect();
    match fname {
        "ANY" => return Ok(TqValue::Bool(flatten.iter().any(|v| v.truthy()))),
        "ALL" => {
            return Ok(TqValue::Bool(
                !flatten.is_empty() && flatten.iter().all(|v| v.truthy()),
            ))
        }
        "NTRUE" => {
            return Ok(TqValue::Int(
                flatten.iter().filter(|v| v.truthy()).count() as i64
            ));
        }
        "NFALSE" => {
            return Ok(TqValue::Int(
                flatten.iter().filter(|v| !v.truthy()).count() as i64
            ));
        }
        _ => {}
    }

    // FRACTILE takes a fraction argument after the data.
    if fname == "FRACTILE" {
        if args.len() != 2 {
            return Err(TaqlError::Eval("fractile expects (array, fraction)".into()));
        }
        let (data, _) = stat_input(fname, std::slice::from_ref(&args[0]))?;
        let frac = as_f(&args[1])
            .ok_or_else(|| TaqlError::Eval("fractile fraction must be numeric".into()))?;
        let s = sorted(data);
        if s.is_empty() {
            return Ok(TqValue::Float(0.0));
        }
        let pos = frac * (s.len() as f64 - 1.0);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let v = if lo == hi {
            s[lo]
        } else {
            s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
        };
        return Ok(TqValue::Float(v));
    }

    let (x, all_i) = stat_input(fname, args)?;
    let x = sorted(x);
    let empty = x.is_empty();
    let fi = |v: f64| TqValue::Float(v);
    let ii = |v: i64| TqValue::Int(v);
    Ok(match fname {
        "MIN" => {
            if empty {
                ii(0)
            } else if all_i {
                ii(x[0] as i64)
            } else {
                fi(x[0])
            }
        }
        "MAX" => {
            if empty {
                ii(0)
            } else if all_i {
                ii(x[x.len() - 1] as i64)
            } else {
                fi(x[x.len() - 1])
            }
        }
        "SUM" => {
            let s: f64 = x.iter().sum();
            if all_i {
                ii(s as i64)
            } else {
                fi(s)
            }
        }
        "PRODUCT" => {
            let p: f64 = x.iter().product();
            if all_i {
                ii(p as i64)
            } else {
                fi(p)
            }
        }
        "SUMSQR" => {
            let s: f64 = x.iter().map(|v| v * v).sum();
            if all_i {
                ii(s as i64)
            } else {
                fi(s)
            }
        }
        "MEAN" | "AVG" => fi(mean_of(&x)),
        "VARIANCE" => {
            let m = mean_of(&x);
            fi(if empty {
                0.0
            } else {
                x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64
            })
        }
        "SAMPLEVARIANCE" => {
            let m = mean_of(&x);
            fi(if x.len() > 1 {
                x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (x.len() - 1) as f64
            } else {
                0.0
            })
        }
        "STDDEV" => {
            let m = mean_of(&x);
            fi(if empty {
                0.0
            } else {
                (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64).sqrt()
            })
        }
        "SAMPLESTDDEV" => {
            let m = mean_of(&x);
            fi(if x.len() > 1 {
                (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (x.len() - 1) as f64).sqrt()
            } else {
                0.0
            })
        }
        "AVDEV" => {
            let m = mean_of(&x);
            fi(if empty {
                0.0
            } else {
                x.iter().map(|v| (v - m).abs()).sum::<f64>() / x.len() as f64
            })
        }
        "RMS" => fi(if empty {
            0.0
        } else {
            (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
        }),
        "MEDIAN" => fi(median_of(&x)),
        _ => unreachable!(),
    })
}

/// Group aggregates (`GMIN`, `GSUM`, ...) over a group's column values.
fn group_agg(fname: &str, items: &[TqValue], frac: Option<&TqValue>) -> TResult<TqValue> {
    match fname {
        "GFIRST" => Ok(items.first().cloned().unwrap_or(TqValue::Int(0))),
        "GLAST" => Ok(items.last().cloned().unwrap_or(TqValue::Int(0))),
        "GANY" => Ok(TqValue::Bool(items.iter().any(|v| v.truthy()))),
        "GALL" => Ok(TqValue::Bool(
            !items.is_empty() && items.iter().all(|v| v.truthy()),
        )),
        "GNTRUE" => Ok(TqValue::Int(
            items.iter().filter(|v| v.truthy()).count() as i64
        )),
        "GNFALSE" => Ok(TqValue::Int(
            items.iter().filter(|v| !v.truthy()).count() as i64
        )),
        "GMIN" | "GMAX" | "GSUM" | "GSUMSQR" | "GPRODUCT" | "GMEAN" | "GAVG" | "GVARIANCE"
        | "GSTDDEV" | "GRMS" | "GMEDIAN" | "GFRACTILE" => {
            let (x, all_i) = stat_input(fname, items)?;
            let x = sorted(x);
            let empty = x.is_empty();
            let fi = |v: f64| TqValue::Float(v);
            let ii = |v: i64| TqValue::Int(v);
            let mean = |s: &[f64]| -> f64 {
                if s.is_empty() {
                    0.0
                } else {
                    s.iter().sum::<f64>() / s.len() as f64
                }
            };
            Ok(match fname {
                "GMIN" => {
                    if empty {
                        ii(0)
                    } else if all_i {
                        ii(x[0] as i64)
                    } else {
                        fi(x[0])
                    }
                }
                "GMAX" => {
                    if empty {
                        ii(0)
                    } else if all_i {
                        ii(x[x.len() - 1] as i64)
                    } else {
                        fi(x[x.len() - 1])
                    }
                }
                "GSUM" => {
                    let s: f64 = x.iter().sum();
                    if all_i {
                        ii(s as i64)
                    } else {
                        fi(s)
                    }
                }
                "GSUMSQR" => {
                    let s: f64 = x.iter().map(|v| v * v).sum();
                    if all_i {
                        ii(s as i64)
                    } else {
                        fi(s)
                    }
                }
                "GPRODUCT" => {
                    let p: f64 = x.iter().product();
                    if all_i {
                        ii(p as i64)
                    } else {
                        fi(p)
                    }
                }
                "GMEAN" | "GAVG" => fi(mean(&x)),
                "GVARIANCE" => {
                    let m = mean(&x);
                    fi(if empty {
                        0.0
                    } else {
                        x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64
                    })
                }
                "GSTDDEV" => {
                    let m = mean(&x);
                    fi(if empty {
                        0.0
                    } else {
                        (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64).sqrt()
                    })
                }
                "GRMS" => fi(if empty {
                    0.0
                } else {
                    (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
                }),
                "GMEDIAN" => fi(median_of(&x)),
                "GFRACTILE" => {
                    let frac = frac.and_then(as_f).unwrap_or(0.5);
                    if empty {
                        TqValue::Float(0.0)
                    } else {
                        let pos = frac * (x.len() as f64 - 1.0);
                        let lo = pos.floor() as usize;
                        let hi = pos.ceil() as usize;
                        fi(if lo == hi {
                            x[lo]
                        } else {
                            x[lo] + (x[hi] - x[lo]) * (pos - lo as f64)
                        })
                    }
                }
                _ => unreachable!(),
            })
        }
        _ => unreachable!(),
    }
}

const MONTH_ABBREV: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
const DAY_ABBREV: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

fn mjd_floor(mjd: f64) -> i64 {
    mjd.floor() as i64
}

/// casacore `MVTime::ymd` (Gregorian calendar): (year, month, day) of an
/// MJD (days since 1858-11-17). Verbatim translation of `MVTime.cc`.
fn mjd_ymd(mjd: f64) -> (i64, i64, i64) {
    let z = mjd_floor(mjd) + 2_400_001;
    let mut dd = z;
    if z >= 2_299_161 {
        let al = (((z as f64 - 1_867_216.25) / 36_524.25).floor()) as i64;
        dd = z + 1 + al - al / 4;
    }
    dd += 1524;
    let yyyy = ((dd as f64 - 122.1) / 365.25).floor() as i64;
    let d0 = (365.25 * yyyy as f64).floor() as i64;
    let tmp = ((dd as f64 - d0 as f64) / 30.6001).floor() as i64;
    let day = dd - d0 - (30.6001 * tmp as f64).floor() as i64;
    let mm = if tmp < 14 { tmp - 1 } else { tmp - 13 };
    let yyyy = if mm > 2 { yyyy - 4715 - 1 } else { yyyy - 4715 };
    (yyyy, mm, day)
}

/// casacore `MVTime::weekday`: 1=Monday .. 7=Sunday.
fn mjd_weekday(mjd: f64) -> i64 {
    (((mjd_floor(mjd) + 2) % 7 + 7) % 7) + 1
}

/// casacore `MVTime::yearday` (1..366).
fn mjd_yearday(mjd: f64) -> i64 {
    let (y, m, d) = mjd_ymd(mjd);
    let c = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
        (m + 9) / 12
    } else {
        2 * ((m + 9) / 12)
    };
    (275 * m) / 9 - c + d - 30
}

/// casacore `MVTime::yearweek` (ISO-style week of year; can be 0).
fn mjd_yearweek(mjd: f64) -> i64 {
    let mut yd = mjd_yearday(mjd) - 4;
    let yw = (yd + 7) / 7;
    yd %= 7;
    if yd >= 0 {
        if yd >= mjd_weekday(mjd) {
            yw + 1
        } else {
            yw
        }
    } else if yd + 7 >= mjd_weekday(mjd) {
        yw + 1
    } else {
        yw
    }
}

/// Days since 1970-01-01 of a (proleptic Gregorian) calendar date.
fn civil_days(year: i64, month: i64, day: i64) -> i64 {
    let mut y = year;
    if month <= 2 {
        y -= 1;
    }
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Parse an ISO `"YYYY-MM-DD[ HH:MM:SS[.fff]]"` (or `"/"` separators) string
/// to an MJD.
fn parse_datetime(s: &str) -> TResult<f64> {
    let mut toks = s.trim().split([' ', 'T']).filter(|p| !p.is_empty());
    let date = match toks.next() {
        Some(d) => d,
        None => return Err(TaqlError::Eval(format!("invalid date string {s:?}"))),
    };
    let time = toks.next();
    let ds: Vec<&str> = date.split(['-', '/']).collect();
    if ds.len() != 3 {
        return Err(TaqlError::Eval(format!("invalid date string {s:?}")));
    }
    let num = |x: &str, what: &str| -> TResult<i64> {
        x.parse()
            .map_err(|_| TaqlError::Eval(format!("invalid date string {s:?}: {what}")))
    };
    let (y, m, d) = (
        num(ds[0], "year")?,
        num(ds[1], "month")?,
        num(ds[2], "day")?,
    );
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return Err(TaqlError::Eval(format!("invalid date string {s:?}")));
    }
    let mut frac = 0.0;
    if let Some(time) = time {
        let t: Vec<&str> = time.split(':').collect();
        if t.len() > 3 {
            return Err(TaqlError::Eval(format!("invalid date string {s:?}")));
        }
        let h = num(t[0], "hour")?;
        let mi = if t.len() > 1 { num(t[1], "minute")? } else { 0 };
        let sec: f64 = if t.len() > 2 {
            t[2].parse()
                .map_err(|_| TaqlError::Eval(format!("invalid date string {s:?}")))?
        } else {
            0.0
        };
        if !(0..=23).contains(&h) || !(0..=59).contains(&mi) || !(0.0..60.0).contains(&sec) {
            return Err(TaqlError::Eval(format!("invalid date string {s:?}")));
        }
        frac = (h as f64 * 3600.0 + mi as f64 * 60.0 + sec) / 86_400.0;
    }
    Ok(civil_days(y, m, d) as f64 + 40_587.0 + frac)
}

/// A date/time argument: an MJD number or an ISO date string.
fn date_arg(v: &TqValue) -> TResult<f64> {
    match v {
        TqValue::Str(s) => parse_datetime(s),
        TqValue::Int(i) => Ok(*i as f64),
        TqValue::Float(f) => Ok(*f),
        other => Err(TaqlError::Eval(format!("date/time on non-date {other:?}"))),
    }
}

/// (hours, minutes, seconds) of the fractional day.
fn mjd_hms(mjd: f64) -> (i64, i64, f64) {
    let total = mjd.fract().rem_euclid(1.0) * 86_400.0;
    let h = total.div_euclid(3600.0) as i64;
    let mi = ((total - h as f64 * 3600.0) / 60.0).floor() as i64;
    let sec = total - h as f64 * 3600.0 - mi as f64 * 60.0;
    (h, mi, sec)
}

/// `hms(radians)`: time-of-day angle as `HHhMMmSS.sss` (casacore stringHMS).
fn hms_str(rad: f64) -> String {
    let hours = (rad / std::f64::consts::TAU - (rad / std::f64::consts::TAU).floor()) * 24.0;
    sexa_str(hours, "h", "m", 2, 2, false, false)
}

/// `dms(radians)`: angle as `+DDDdMMmSS.sss` (casacore stringDMS).
fn dms_str(rad: f64) -> String {
    let deg = rad * 180.0 / std::f64::consts::PI;
    sexa_str(deg.abs(), "d", "m", 3, 2, true, deg < 0.0)
}

fn sexa_str(v: f64, s1: &str, s2: &str, w1: usize, w2: usize, signed: bool, neg: bool) -> String {
    let d = v.floor() as i64;
    let m = ((v - d as f64) * 60.0).floor() as i64;
    let sec = (v - d as f64) * 3600.0 - m as f64 * 60.0;
    if signed {
        let sign = if neg { "-" } else { "+" };
        format!("{sign}{d:0w1$}{s1}{m:0w2$}{s2}{sec:06.3}")
    } else {
        format!("{d:0w1$}{s1}{m:0w2$}{s2}{sec:06.3}")
    }
}

/// The `running*`/`boxed*` cumulative statistic families: the stem aggregate
/// applied to every prefix of the array (running sum, running min, ...),
/// returning an array. `boxed*` is the same for the flat 1-D value model.
fn stats_cumulative(fname_stem: &str, args: &[TqValue]) -> TResult<TqValue> {
    let mut vals: Vec<TqValue> = Vec::new();
    for a in args {
        for x in flat_elements(a) {
            if as_f(x).is_none() {
                return Err(TaqlError::Eval(format!("{fname_stem} on non-number {x:?}")));
            }
            vals.push(x.clone());
        }
    }
    let mut out = Vec::with_capacity(vals.len());
    for k in 1..=vals.len() {
        out.push(stats(fname_stem, &vals[..k])?);
    }
    Ok(TqValue::Arr(out))
}

// ---------------------------------------------------------------------------
// A minimal regex engine for the `~` (REGEX) operator: literals, `.`,
// `^`/`$` anchors, `*`/`+`/`?`, `[...]`/`[^...]`, `(...)`, `a|b`, and `\`
// escapes. Matches a substring unless `^`/`$` anchors it (like casacore
// `Regex::match`).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum RNode {
    Lit(char),
    Any,
    AnchorStart,
    AnchorEnd,
    Class {
        negated: bool,
        ranges: Vec<(char, char)>,
    },
    Star(Box<RNode>),
    Plus(Box<RNode>),
    Opt(Box<RNode>),
    Group(Vec<RNode>),
    Alt(Vec<RNode>, Vec<RNode>),
}

struct RParse<'a> {
    cs: &'a [char],
    pos: usize,
}

impl RParse<'_> {
    fn peek(&self) -> Option<char> {
        self.cs.get(self.pos).copied()
    }
    fn next(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }
    fn parse_alt(&mut self) -> Result<Vec<RNode>, TaqlError> {
        let mut seq = self.parse_seq()?;
        while self.peek() == Some('|') {
            self.next();
            let other = self.parse_seq()?;
            seq = vec![RNode::Alt(seq, other)];
        }
        Ok(seq)
    }
    fn parse_seq(&mut self) -> Result<Vec<RNode>, TaqlError> {
        let mut out = Vec::new();
        while let Some(c) = self.peek() {
            if c == ')' || c == '|' {
                break;
            }
            out.push(self.parse_atom()?);
        }
        Ok(out)
    }
    fn parse_atom(&mut self) -> Result<RNode, TaqlError> {
        let c = match self.next() {
            Some(c) => c,
            None => return Err(TaqlError::Eval("regex: unexpected end of pattern".into())),
        };
        let atom = match c {
            '^' => RNode::AnchorStart,
            '$' => RNode::AnchorEnd,
            '.' => RNode::Any,
            '[' => {
                let (negated, ranges) = self.parse_class()?;
                RNode::Class { negated, ranges }
            }
            '(' => {
                let sub = self.parse_alt()?;
                if self.next() != Some(')') {
                    return Err(TaqlError::Eval("regex: missing ')'".into()));
                }
                RNode::Group(sub)
            }
            '\\' => {
                let esc = self
                    .next()
                    .ok_or_else(|| TaqlError::Eval("regex: trailing \\".into()))?;
                RNode::Lit(esc)
            }
            '*' | '+' | '?' => RNode::Lit(c),
            other => RNode::Lit(other),
        };
        Ok(match self.peek() {
            Some('*') => {
                self.next();
                RNode::Star(Box::new(atom))
            }
            Some('+') => {
                self.next();
                RNode::Plus(Box::new(atom))
            }
            Some('?') => {
                self.next();
                RNode::Opt(Box::new(atom))
            }
            _ => atom,
        })
    }
    fn parse_class(&mut self) -> Result<(bool, Vec<(char, char)>), TaqlError> {
        let negated = if self.peek() == Some('^') {
            self.next();
            true
        } else {
            false
        };
        let mut ranges: Vec<(char, char)> = Vec::new();
        loop {
            let c = match self.next() {
                Some(']') => break,
                Some(c) => c,
                None => return Err(TaqlError::Eval("regex: unterminated [...]".into())),
            };
            let lo = if c == '\\' {
                self.next()
                    .ok_or_else(|| TaqlError::Eval("regex: trailing \\".into()))?
            } else {
                c
            };
            if self.peek() == Some('-') && self.cs.get(self.pos + 1).is_some_and(|&h| h != ']') {
                self.next(); // '-'
                let hi = self
                    .next()
                    .ok_or_else(|| TaqlError::Eval("regex: bad range".into()))?;
                ranges.push((lo, hi));
            } else {
                ranges.push((lo, lo));
            }
        }
        Ok((negated, ranges))
    }
}

fn class_matches(c: char, negated: bool, ranges: &[(char, char)]) -> bool {
    let hit = ranges.iter().any(|&(a, b)| c >= a && c <= b);
    hit != negated
}

/// Match the head of the text at `s` against one node.
fn rmatch_node(n: &RNode, cs: &[char], s: usize) -> Option<usize> {
    match n {
        RNode::Lit(c) if s < cs.len() && cs[s] == *c => Some(s + 1),
        RNode::Any if s < cs.len() => Some(s + 1),
        RNode::Class { negated, ranges }
            if s < cs.len() && class_matches(cs[s], *negated, ranges) =>
        {
            Some(s + 1)
        }
        RNode::AnchorStart if s == 0 => Some(s),
        RNode::AnchorEnd if s == cs.len() => Some(s),
        RNode::Group(g) => rmatch_seq(g, cs, s),
        RNode::Alt(a, b) => rmatch_seq(a, cs, s).or_else(|| rmatch_seq(b, cs, s)),
        RNode::Opt(inner) => rmatch_node(inner, cs, s).or(Some(s)),
        // A quantified node nested in another quantifier: consume greedily.
        RNode::Star(inner) | RNode::Plus(inner) => {
            let mut k = s;
            let mut last = Some(s);
            while let Some(nk) = rmatch_node(inner, cs, k) {
                last = Some(nk);
                k = nk;
            }
            last
        }
        _ => None,
    }
}

/// Match `nodes` against the text at `s`, returning the end position.
fn rmatch_seq(nodes: &[RNode], cs: &[char], s: usize) -> Option<usize> {
    if nodes.is_empty() {
        return Some(s);
    }
    let head = &nodes[0];
    let rest = &nodes[1..];
    match head {
        RNode::AnchorStart => {
            if s == 0 {
                rmatch_seq(rest, cs, s)
            } else {
                None
            }
        }
        RNode::AnchorEnd => {
            if s == cs.len() {
                rmatch_seq(rest, cs, s)
            } else {
                None
            }
        }
        RNode::Lit(c) => {
            if s < cs.len() && cs[s] == *c {
                rmatch_seq(rest, cs, s + 1)
            } else {
                None
            }
        }
        RNode::Any => {
            if s < cs.len() {
                rmatch_seq(rest, cs, s + 1)
            } else {
                None
            }
        }
        RNode::Class { negated, ranges } => {
            if s < cs.len() && class_matches(cs[s], *negated, ranges) {
                rmatch_seq(rest, cs, s + 1)
            } else {
                None
            }
        }
        RNode::Group(g) => match rmatch_seq(g, cs, s) {
            Some(e) => rmatch_seq(rest, cs, e),
            None => None,
        },
        RNode::Alt(a, b) => rmatch_seq(a, cs, s)
            .or_else(|| rmatch_seq(b, cs, s))
            .and_then(|e| rmatch_seq(rest, cs, e)),
        RNode::Star(inner) => {
            // Greedy with backtracking: try every possible inner length.
            let mut k = s;
            let mut ends = Vec::new();
            while let Some(nk) = rmatch_node(inner, cs, k) {
                ends.push(nk);
                k = nk;
            }
            for e in ends.iter().rev().copied() {
                if let Some(r) = rmatch_seq(rest, cs, e) {
                    return Some(r);
                }
            }
            rmatch_seq(rest, cs, s)
        }
        RNode::Plus(inner) => {
            let first = rmatch_node(inner, cs, s)?;
            let mut k = first;
            let mut ends = vec![first];
            while let Some(nk) = rmatch_node(inner, cs, k) {
                ends.push(nk);
                k = nk;
            }
            for e in ends.iter().rev().copied() {
                if let Some(r) = rmatch_seq(rest, cs, e) {
                    return Some(r);
                }
            }
            None
        }
        RNode::Opt(inner) => {
            if let Some(nk) = rmatch_node(inner, cs, s) {
                if let Some(e) = rmatch_seq(rest, cs, nk) {
                    return Some(e);
                }
            }
            rmatch_seq(rest, cs, s)
        }
    }
}

/// Substring regex search (`^` anchors to the text start).
fn regex_search(text: &str, pat: &str) -> bool {
    let nodes = match parse_regex(pat) {
        Ok(n) => n,
        Err(_) => return false,
    };
    let cs: Vec<char> = text.chars().collect();
    let starts: Vec<usize> = if matches!(nodes.first(), Some(RNode::AnchorStart)) {
        vec![0]
    } else {
        (0..=cs.len()).collect()
    };
    starts
        .into_iter()
        .any(|s| rmatch_seq(&nodes, &cs, s).is_some())
}

fn parse_regex(pat: &str) -> Result<Vec<RNode>, TaqlError> {
    let cs: Vec<char> = pat.chars().collect();
    let mut p = RParse { cs: &cs, pos: 0 };
    let nodes = p.parse_alt()?;
    if p.pos != p.cs.len() {
        return Err(TaqlError::Eval(format!("invalid regex pattern {pat:?}")));
    }
    Ok(nodes)
}

/// The `~` operator match for the three pattern flavours.
fn regex_matches(text: &str, kind: RegexKind, pat: &str) -> bool {
    match kind {
        RegexKind::Regex => regex_search(text, pat),
        RegexKind::SqlPattern => sql_like(text, pat, false),
        RegexKind::Pattern => pattern_star(text, pat),
    }
}

/// C-style pattern: `*` matches any run, `?` matches one char.
fn pattern_star(text: &str, pat: &str) -> bool {
    let t: Vec<char> = text.chars().collect();
    let p: Vec<char> = pat.chars().collect();
    // Same DP as sql_like with `*`/`?` wildcards.
    let (n, m) = (t.len(), p.len());
    let mut dp = vec![vec![false; m + 1]; n + 1];
    dp[0][0] = true;
    for j in 1..=m {
        if p[j - 1] == '*' {
            dp[0][j] = dp[0][j - 1];
        }
    }
    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = match p[j - 1] {
                '*' => dp[i][j - 1] || dp[i - 1][j],
                '?' => dp[i - 1][j - 1],
                c => t[i - 1] == c && dp[i - 1][j - 1],
            };
        }
    }
    dp[n][m]
}

/// The scalar/array function library (casacore TaQL function registry:
/// `TableParseFunc.cc`). `fname` is upper-cased; the selected families are
/// numeric/trig, integer casts, complex parts, strings, presence probes,
/// array shape, statistics, and date/time. The statistic families remain the
/// plain, plural, and running/boxed forms.
fn scalar_func(fname: &str, args: &[TqValue]) -> TResult<TqValue> {
    macro_rules! need {
        ($n:expr) => {{
            if args.len() != $n {
                return Err(TaqlError::Eval(format!(
                    "{fname} expects {} argument(s), got {}",
                    $n,
                    args.len()
                )));
            }
        }};
    }
    macro_rules! onef {
        () => {{
            need!(1);
            as_f(&args[0])
                .ok_or_else(|| TaqlError::Eval(format!("{fname} on non-number {:?}", args[0])))?
        }};
    }
    macro_rules! strarg {
        ($i:expr) => {
            match &args[$i] {
                TqValue::Str(s) => s.clone(),
                other => {
                    return Err(TaqlError::Eval(format!("{fname} on non-string {other:?}")));
                }
            }
        };
    }

    // Plural (`sums`, `mins`, `means`, ...) variants: the aggregate applied
    // element-wise across equal-length arrays.
    if let Some(stem) = fname.strip_suffix('S') {
        if matches!(
            stem,
            "SUM"
                | "PRODUCT"
                | "SUMSQR"
                | "MIN"
                | "MAX"
                | "MEAN"
                | "AVG"
                | "VARIANCE"
                | "SAMPLEVARIANCE"
                | "STDDEV"
                | "SAMPLESTDDEV"
                | "AVDEV"
                | "RMS"
                | "MEDIAN"
                | "FRACTILE"
                | "ANY"
                | "ALL"
                | "NTRUE"
                | "NFALSE"
        ) {
            return stats_multi(fname, args);
        }
    }

    // `running*` / `boxed*` cumulative variants over an array.
    if let Some(stem) = fname
        .strip_prefix("RUNNING")
        .or_else(|| fname.strip_prefix("BOXED"))
    {
        if matches!(
            stem,
            "SUM"
                | "PRODUCT"
                | "SUMSQR"
                | "MIN"
                | "MAX"
                | "MEAN"
                | "AVG"
                | "VARIANCE"
                | "SAMPLEVARIANCE"
                | "STDDEV"
                | "SAMPLESTDDEV"
                | "AVDEV"
                | "RMS"
                | "MEDIAN"
                | "ANY"
                | "ALL"
                | "NTRUE"
                | "NFALSE"
        ) {
            return stats_cumulative(stem, args);
        }
    }

    // Statistics over scalar/array arguments.
    if matches!(
        fname,
        "MIN"
            | "MAX"
            | "SUM"
            | "PRODUCT"
            | "SUMSQR"
            | "MEAN"
            | "AVG"
            | "VARIANCE"
            | "SAMPLEVARIANCE"
            | "STDDEV"
            | "SAMPLESTDDEV"
            | "AVDEV"
            | "RMS"
            | "MEDIAN"
            | "FRACTILE"
            | "ANY"
            | "ALL"
            | "NTRUE"
            | "NFALSE"
    ) {
        return stats(fname, args);
    }

    Ok(match fname {
        "PI" => TqValue::Float(std::f64::consts::PI),
        "E" => TqValue::Float(std::f64::consts::E),
        "C" => TqValue::Float(299_792_458.0), // speed of light [m/s]

        // ---- integer-preserving unary ----
        "ABS" => match &args[0] {
            TqValue::Int(i) if args.len() == 1 => TqValue::Int(i.abs()),
            TqValue::Float(f) if args.len() == 1 => TqValue::Float(f.abs()),
            other => {
                need!(1);
                return Err(TaqlError::Eval(format!("abs({other:?})")));
            }
        },
        "NORM" => {
            need!(1);
            match &args[0] {
                TqValue::Int(i) => TqValue::Int(i.abs()),
                other => {
                    let (re, im) = complex_parts(other);
                    TqValue::Float(f64::hypot(re, im))
                }
            }
        }
        "SIGN" => {
            need!(1);
            match &args[0] {
                TqValue::Int(i) => TqValue::Int(i.signum()),
                other => {
                    let f = as_f(other).unwrap_or(0.0);
                    TqValue::Int(if f > 0.0 {
                        1
                    } else if f < 0.0 {
                        -1
                    } else {
                        0
                    })
                }
            }
        }
        "SQUARE" | "SQR" => {
            need!(1);
            match &args[0] {
                TqValue::Int(i) => TqValue::Int(i.saturating_mul(*i)),
                other => {
                    let f =
                        as_f(other).ok_or_else(|| TaqlError::Eval(format!("square({other:?})")))?;
                    TqValue::Float(f * f)
                }
            }
        }
        "CUBE" => {
            need!(1);
            match &args[0] {
                TqValue::Int(i) => TqValue::Int(i.saturating_mul(*i).saturating_mul(*i)),
                other => {
                    let f =
                        as_f(other).ok_or_else(|| TaqlError::Eval(format!("cube({other:?})")))?;
                    TqValue::Float(f * f * f)
                }
            }
        }
        "INT" | "INTEGER" => TqValue::Int(onef!() as i64),
        "FLOOR" => match &args[0] {
            TqValue::Int(i) if args.len() == 1 => TqValue::Int(*i),
            TqValue::Float(_) if args.len() == 1 => TqValue::Float(onef!().floor()),
            _ => {
                need!(1);
                return Err(TaqlError::Eval(format!("floor({:?})", args[0])));
            }
        },
        "CEIL" => match &args[0] {
            TqValue::Int(i) if args.len() == 1 => TqValue::Int(*i),
            TqValue::Float(_) if args.len() == 1 => TqValue::Float(onef!().ceil()),
            _ => {
                need!(1);
                return Err(TaqlError::Eval(format!("ceil({:?})", args[0])));
            }
        },
        "ROUND" => match &args[0] {
            TqValue::Int(i) if args.len() == 1 => TqValue::Int(*i),
            TqValue::Float(_) if args.len() == 1 => TqValue::Float(onef!().round()),
            _ => {
                need!(1);
                return Err(TaqlError::Eval(format!("round({:?})", args[0])));
            }
        },

        // ---- real-valued unary ----
        "SQRT" => TqValue::Float(onef!().sqrt()),
        "CBRT" => TqValue::Float(onef!().cbrt()),
        "EXP" => TqValue::Float(onef!().exp()),
        "LN" | "LOG" => TqValue::Float(onef!().ln()),
        "LOG10" => TqValue::Float(onef!().log10()),
        "SIN" => TqValue::Float(onef!().sin()),
        "COS" => TqValue::Float(onef!().cos()),
        "TAN" => TqValue::Float(onef!().tan()),
        "ASIN" => TqValue::Float(onef!().asin()),
        "ACOS" => TqValue::Float(onef!().acos()),
        "ATAN" => TqValue::Float(onef!().atan()),
        "SINH" => TqValue::Float(onef!().sinh()),
        "COSH" => TqValue::Float(onef!().cosh()),
        "TANH" => TqValue::Float(onef!().tanh()),

        // ---- binary real-valued ----
        "ATAN2" => {
            need!(2);
            let (a, b) = (
                as_f(&args[0]).ok_or_else(|| TaqlError::Eval("atan2(a,b)".into()))?,
                as_f(&args[1]).ok_or_else(|| TaqlError::Eval("atan2(a,b)".into()))?,
            );
            TqValue::Float(a.atan2(b))
        }
        "POW" => {
            need!(2);
            TqValue::Float(
                as_f(&args[0])
                    .ok_or_else(|| TaqlError::Eval("pow(a,b)".into()))?
                    .powf(as_f(&args[1]).ok_or_else(|| TaqlError::Eval("pow(a,b)".into()))?),
            )
        }
        "FMOD" => {
            need!(2);
            TqValue::Float(
                as_f(&args[0]).ok_or_else(|| TaqlError::Eval("fmod(a,b)".into()))?
                    % as_f(&args[1]).ok_or_else(|| TaqlError::Eval("fmod(a,b)".into()))?,
            )
        }

        // ---- presence / predicates ----
        "ISNAN" => TqValue::Bool(matches!(args.first(), Some(TqValue::Float(f)) if f.is_nan())),
        "ISINF" => {
            TqValue::Bool(matches!(args.first(), Some(TqValue::Float(f)) if f.is_infinite()))
        }
        "ISFINITE" => match args.first() {
            Some(TqValue::Float(f)) => TqValue::Bool(f.is_finite()),
            Some(_) => TqValue::Bool(true),
            None => TqValue::Bool(true),
        },
        "ISNULL" => TqValue::Bool(false),
        "ISDEFINED" => TqValue::Bool(true),

        // ---- complex parts ----
        "REAL" => {
            need!(1);
            TqValue::Float(complex_parts(&args[0]).0)
        }
        "IMAG" => {
            need!(1);
            TqValue::Float(complex_parts(&args[0]).1)
        }
        "AMPL" | "AMPLITUDE" => {
            need!(1);
            let (re, im) = complex_parts(&args[0]);
            TqValue::Float(f64::hypot(re, im))
        }
        "ARG" | "PHASE" => {
            need!(1);
            let (re, im) = complex_parts(&args[0]);
            // atan2(y=im, x=re)
            TqValue::Float(im.atan2(re))
        }
        "CONJ" => {
            need!(1);
            let (re, im) = complex_parts(&args[0]);
            TqValue::Arr(vec![TqValue::Float(re), TqValue::Float(-im)])
        }
        "COMPLEX" | "FORMCOMPLEX" => {
            need!(2);
            let re = as_f(&args[0]).ok_or_else(|| {
                TaqlError::Eval(format!("{fname}(a,b): non-number {:?}", args[0]))
            })?;
            let im = as_f(&args[1]).ok_or_else(|| {
                TaqlError::Eval(format!("{fname}(a,b): non-number {:?}", args[1]))
            })?;
            TqValue::Arr(vec![TqValue::Float(re), TqValue::Float(im)])
        }

        // ---- strings ----
        "STRLENGTH" | "LEN" => TqValue::Int(strarg!(0).chars().count() as i64),
        "UPCASE" | "UPPER" | "TOUPPER" | "TO_UPPER" => {
            TqValue::Str(strarg!(0).to_ascii_uppercase())
        }
        "DOWNCASE" | "LOWER" | "TOLOWER" | "TO_LOWER" => {
            TqValue::Str(strarg!(0).to_ascii_lowercase())
        }
        "CAPITALIZE" => {
            let s = strarg!(0);
            let mut c = s.chars();
            TqValue::Str(match c.next() {
                Some(first) => {
                    let rest: String = c.map(|ch| ch.to_ascii_lowercase()).collect();
                    first.to_ascii_uppercase().to_string() + &rest
                }
                None => String::new(),
            })
        }
        "REVERSESTRING" | "SREVERSE" => TqValue::Str(strarg!(0).chars().rev().collect()),
        "TRIM" | "LTRIM" | "RTRIM" => {
            if args.is_empty() || args.len() > 2 {
                return Err(TaqlError::Eval(format!(
                    "{fname} expects 1 or 2 arguments, got {}",
                    args.len()
                )));
            }
            let s = strarg!(0);
            let cutset = if args.len() == 2 {
                strarg!(1)
            } else {
                String::new()
            };
            let side = match fname {
                "LTRIM" => 2,
                "RTRIM" => 3,
                _ => 1,
            };
            TqValue::Str(str_trim(&s, side, &cutset))
        }
        "SUBSTR" | "SUBSTRING" => {
            if args.len() < 2 || args.len() > 3 {
                return Err(TaqlError::Eval(format!(
                    "{fname} expects 2 or 3 arguments, got {}",
                    args.len()
                )));
            }
            let s = strarg!(0);
            let start = as_i(&args[1])
                .ok_or_else(|| TaqlError::Eval(format!("{fname}: start must be an integer")))?;
            let end =
                if args.len() == 3 {
                    Some(as_i(&args[2]).ok_or_else(|| {
                        TaqlError::Eval(format!("{fname}: end must be an integer"))
                    })?)
                } else {
                    None
                };
            TqValue::Str(substr_of(&s, start, end))
        }
        "REPLACE" => {
            need!(3);
            let s = strarg!(0);
            let from = strarg!(1);
            let to = strarg!(2);
            TqValue::Str(s.replace(&from, &to))
        }
        "STR" | "STRING" => {
            need!(1);
            TqValue::Str(stringify(&args[0]))
        }
        "IIF" => {
            need!(3);
            if args[0].truthy() {
                args[1].clone()
            } else {
                args[2].clone()
            }
        }

        // ---- arrays & shapes (flat 1-D value model) ----
        "ARRAY" => TqValue::Arr(args.to_vec()),
        "SHAPE" => match args.first() {
            Some(TqValue::Arr(items)) => TqValue::Arr(vec![TqValue::Int(items.len() as i64)]),
            _ => TqValue::Arr(vec![]),
        },
        "NDIM" => match args.first() {
            Some(TqValue::Arr(_)) => TqValue::Int(1),
            _ => TqValue::Int(0),
        },
        "NELEMENTS" | "COUNT" => match args.first() {
            Some(TqValue::Arr(items)) => TqValue::Int(items.len() as i64),
            Some(TqValue::Str(s)) => TqValue::Int(s.chars().count() as i64),
            Some(_) => TqValue::Int(1),
            None => TqValue::Int(0),
        },
        "TRANSPOSE" => args.first().cloned().unwrap_or(TqValue::Int(0)),
        "REVERSEARRAY" | "AREVERSE" => match args.first() {
            Some(TqValue::Arr(items)) => {
                let mut v = items.clone();
                v.reverse();
                TqValue::Arr(v)
            }
            other => other.cloned().unwrap_or(TqValue::Int(0)),
        },
        "FLATTEN" | "ARRAYFLATTEN" => {
            let items: Vec<TqValue> = args.iter().flat_map(flat_elements).cloned().collect();
            TqValue::Arr(items)
        }

        // ---- pattern values for the `~` operator ----
        "REGEX" | "PATTERN" | "SQLPATTERN" => {
            need!(1);
            let s = match &args[0] {
                TqValue::Str(s) => s.clone(),
                other => {
                    return Err(TaqlError::Eval(format!(
                        "{fname} expects a string pattern, got {other:?}"
                    )));
                }
            };
            let kind = match fname {
                "REGEX" => RegexKind::Regex,
                "PATTERN" => RegexKind::Pattern,
                _ => RegexKind::SqlPattern,
            };
            TqValue::Regex(kind, s)
        }

        // ---- sexagesimal strings ----
        "HMS" | "DMS" | "HDMS" => {
            need!(1);
            if fname == "HDMS" {
                let items: Vec<TqValue> = match &args[0] {
                    TqValue::Arr(items) => items.clone(),
                    other => vec![other.clone()],
                };
                let out: TResult<Vec<TqValue>> = items
                    .iter()
                    .map(|v| {
                        let f =
                            as_f(v).ok_or_else(|| TaqlError::Eval("hdms on non-number".into()))?;
                        Ok(TqValue::Str(dms_str(f)))
                    })
                    .collect();
                TqValue::Arr(out?)
            } else {
                let v = as_f(&args[0])
                    .ok_or_else(|| TaqlError::Eval(format!("{fname} on non-number")))?;
                TqValue::Str(if fname == "HMS" {
                    hms_str(v)
                } else {
                    dms_str(v)
                })
            }
        }

        // ---- date/time (MJD; casacore MVTime semantics) ----
        "MJDTODATE" | "MJD" | "DATE" | "DATETIME" | "TIME" | "YEAR" | "MONTH" | "DAY"
        | "CMONTH" | "WEEKDAY" | "DOW" | "CWEEKDAY" | "CDOW" | "WEEK" | "CDATE" | "CTIME"
        | "CDATETIME" | "CTOD" => {
            need!(1);
            let mjd = date_arg(&args[0])?;
            match fname {
                "MJD" | "MJDTODATE" | "DATETIME" => TqValue::Float(mjd),
                "DATE" => TqValue::Float(mjd.floor()),
                "TIME" => TqValue::Float(mjd.fract().rem_euclid(1.0) * std::f64::consts::TAU),
                "YEAR" => TqValue::Int(mjd_ymd(mjd).0),
                "MONTH" => TqValue::Int(mjd_ymd(mjd).1),
                "DAY" => TqValue::Int(mjd_ymd(mjd).2),
                "CMONTH" => TqValue::Str(MONTH_ABBREV[(mjd_ymd(mjd).1 - 1) as usize].into()),
                "WEEKDAY" | "DOW" => TqValue::Int(mjd_weekday(mjd)),
                "CWEEKDAY" | "CDOW" => {
                    TqValue::Str(DAY_ABBREV[(mjd_weekday(mjd) - 1) as usize].into())
                }
                "WEEK" => TqValue::Int(mjd_yearweek(mjd)),
                "CDATE" => {
                    let (y, mo, d) = mjd_ymd(mjd);
                    TqValue::Str(format!("{d:02}-{}-{y:04}", MONTH_ABBREV[(mo - 1) as usize]))
                }
                _ => {
                    // CTIME / CDATETIME / CTOD.
                    let (y, mo, d) = mjd_ymd(mjd);
                    let (h, mi, sec) = mjd_hms(mjd);
                    let t = format!("{h:02}:{mi:02}:{sec:06.3}");
                    if fname == "CTIME" {
                        TqValue::Str(t)
                    } else {
                        TqValue::Str(format!("{y:04}/{mo:02}/{d:02}/{t}"))
                    }
                }
            }
        }

        _ => return Err(TaqlError::Unknown(fname.to_string())),
    })
}

/// Apply `base[expr]`: array element indexing or subtable row lookup.
fn index_value(base: TqValue, idx: TqValue) -> TResult<TqValue> {
    let idx = match idx {
        TqValue::Int(i) => i,
        TqValue::Float(f) => f as i64,
        other => {
            return Err(TaqlError::Eval(format!(
                "array index must be an integer, found {other:?}"
            )));
        }
    };
    match base {
        TqValue::Arr(items) => {
            let i = if idx < 0 {
                items.len() as i64 + idx
            } else {
                idx
            };
            Ok(items.get(i as usize).cloned().unwrap_or(TqValue::Int(0)))
        }
        TqValue::Subtable(sub) => {
            let col = sub
                .columns
                .first()
                .ok_or_else(|| TaqlError::Eval("subquery has no column".into()))?;
            Ok(col
                .get(idx as usize)
                .cloned()
                .map(|r| cell_value(&r))
                .unwrap_or(TqValue::Int(0)))
        }
        other => Err(TaqlError::Eval(format!(
            "cannot index {other:?} with [{idx}]"
        ))),
    }
}

fn binary_value(op: &BinOp, l: &TqValue, r: &TqValue) -> TResult<TqValue> {
    use BinOp::*;
    Ok(match op {
        Or => TqValue::Bool(l.truthy() || r.truthy()),
        And => TqValue::Bool(l.truthy() && r.truthy()),
        Add | Sub | Mul | Div | Mod => {
            let (li, lf, lnum) = num(l)?;
            let (ri, rf, rnum) = num(r)?;
            if !lnum || !rnum {
                return Err(TaqlError::Eval("unsupported arithmetic operands".into()));
            }
            match (li, ri) {
                (Some(a), Some(b)) => match op {
                    Add => TqValue::Int(a + b),
                    Sub => TqValue::Int(a - b),
                    Mul => TqValue::Int(a * b),
                    Div => {
                        if b == 0 {
                            return Err(TaqlError::Eval("division by zero".into()));
                        }
                        TqValue::Int(a.div_euclid(b))
                    }
                    Mod => {
                        if b == 0 {
                            return Err(TaqlError::Eval("division by zero".into()));
                        }
                        TqValue::Int(a % b)
                    }
                    _ => unreachable!(),
                },
                _ => TqValue::Float(match op {
                    Add => lf + rf,
                    Sub => lf - rf,
                    Mul => lf * rf,
                    Div => lf / rf,
                    Mod => lf % rf,
                    _ => unreachable!(),
                }),
            }
        }
        Eq | Ne | Lt | Le | Gt | Ge => {
            let ord = compare(l, r);
            TqValue::Bool(match op {
                Eq => ord == std::cmp::Ordering::Equal,
                Ne => ord != std::cmp::Ordering::Equal,
                Lt => ord == std::cmp::Ordering::Less,
                Le => ord != std::cmp::Ordering::Greater,
                Gt => ord == std::cmp::Ordering::Greater,
                Ge => ord != std::cmp::Ordering::Less,
                _ => unreachable!(),
            })
        }
        Like { .. } | In { .. } | Regex { .. } => {
            return Err(TaqlError::Eval(
                "LIKE/IN/~ must be evaluated in their expression context".into(),
            ));
        }
    })
}

/// (int, float, is-number) numeric interpretation of a value.
fn num(v: &TqValue) -> TResult<(Option<i64>, f64, bool)> {
    Ok(match v {
        TqValue::Int(i) => (Some(*i), *i as f64, true),
        TqValue::Float(f) => (None, *f, true),
        TqValue::Bool(b) => (Some(i64::from(*b)), f64::from(*b), true),
        other => {
            return Err(TaqlError::Eval(format!("not numeric: {other:?}")));
        }
    })
}

/// casacore-style comparison: numbers compare numerically across int/float;
/// strings lexicographically; strings sort before numbers.
fn compare(l: &TqValue, r: &TqValue) -> std::cmp::Ordering {
    match (l, r) {
        (TqValue::Int(a), TqValue::Int(b)) => a.cmp(b),
        (TqValue::Float(a), TqValue::Float(b)) => {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        }
        (TqValue::Int(a), TqValue::Float(b)) => (*a as f64)
            .partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal),
        (TqValue::Float(a), TqValue::Int(b)) => a
            .partial_cmp(&(*b as f64))
            .unwrap_or(std::cmp::Ordering::Equal),
        (TqValue::Str(a), TqValue::Str(b)) => a.cmp(b),
        (TqValue::Bool(a), TqValue::Bool(b)) => a.cmp(b),
        (TqValue::Arr(a), TqValue::Arr(b)) => a.len().cmp(&b.len()),
        // Mixed: strings before numbers, matching casacore's 'unsorted'.
        (TqValue::Str(_), _) => std::cmp::Ordering::Less,
        (_, TqValue::Str(_)) => std::cmp::Ordering::Greater,
        (TqValue::Subtable(a), TqValue::Subtable(b)) => a.nrows().cmp(&b.nrows()),
        _ => std::cmp::Ordering::Equal,
    }
}

/// Stable multi-key sort of `rows` by the order-by expressions evaluated at
/// each row; descending keys reverse the comparator (so equal keys swap tie
/// order, matching casacore's DESC).
fn sort_rows(ctx: &EvalCtx<'_>, rows: &mut Vec<i64>, orderby: &[OrderKey]) -> TResult<()> {
    let mut keys: Vec<Vec<TqValue>> = Vec::with_capacity(orderby.len());
    for k in orderby {
        let mut col = Vec::with_capacity(rows.len());
        for &r in rows.iter() {
            col.push(ctx.eval_row(&k.expr, r)?);
        }
        keys.push(col);
    }
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    idx.sort_by(|&a, &b| {
        for (ki, k) in orderby.iter().enumerate() {
            let ord = compare(&keys[ki][a], &keys[ki][b]);
            if ord != std::cmp::Ordering::Equal {
                return if k.desc { ord.reverse() } else { ord };
            }
            // casacore's (unstable) descending sort flips the order of
            // equal-adjacent groups; replicate by breaking the tie with the
            // reversed original index at the first DESC key.
            if k.desc {
                return b.cmp(&a);
            }
        }
        std::cmp::Ordering::Equal
    });
    let sorted: Vec<i64> = idx.iter().map(|&i| rows[i]).collect();
    *rows = sorted;
    Ok(())
}

/// Stable sort of `groups` by the order-by expressions evaluated in each
/// group's context (the grouped-output counterpart of `sort_rows`).
fn sort_groups(ctx: &EvalCtx<'_>, groups: &mut Vec<Vec<i64>>, orderby: &[OrderKey]) -> TResult<()> {
    let mut keys: Vec<Vec<TqValue>> = Vec::with_capacity(orderby.len());
    for k in orderby {
        let mut col = Vec::with_capacity(groups.len());
        for g in groups.iter() {
            col.push(ctx.eval_group(&k.expr, g)?);
        }
        keys.push(col);
    }
    let mut idx: Vec<usize> = (0..groups.len()).collect();
    idx.sort_by(|&a, &b| {
        for (ki, k) in orderby.iter().enumerate() {
            let ord = compare(&keys[ki][a], &keys[ki][b]);
            if ord != std::cmp::Ordering::Equal {
                return if k.desc { ord.reverse() } else { ord };
            }
            if k.desc {
                return b.cmp(&a);
            }
        }
        std::cmp::Ordering::Equal
    });
    let sorted: Vec<Vec<i64>> = idx.iter().map(|&i| groups[i].clone()).collect();
    *groups = sorted;
    Ok(())
}

// ---------------------------------------------------------------------------
// DDL: CREATE TABLE
// ---------------------------------------------------------------------------

/// Parse and execute `CREATE TABLE path [(col, ...)] [KEYWORDS ...] LIMIT n`.
///
/// The column syntax is casacore's DDL subset used by test fixtures:
/// `NAME TYPE [NDIM=n] [SHAPE=[a,b]] [OPTION=n] [MAXLEN=n]`.
/// Supported types (and their casacore abbreviations): `boolean`/`b`,
/// `uchar`, `short`/`sh`, `int`/`i`, `uint`, `float`/`f`, `double`/`d`,
/// `complex`/`c`, `dcomplex`/`cd`, `string`/`s`, plus the fixed-size
/// spellings `I4`, `I8`, `R4`, `R8`, `C8`, `C16` used in fixtures.
pub fn create_table(query: &str) -> Result<PathBuf, TaqlError> {
    let toks = tokenize(query)?;
    let mut p = Parser { toks, pos: 0 };
    let kw = p.expect_ident()?;
    if !kw.eq_ignore_ascii_case("create") {
        return Err(TaqlError::Parse("expected CREATE".into()));
    }
    let kw = p.expect_ident()?;
    if !kw.eq_ignore_ascii_case("table") {
        return Err(TaqlError::Parse("expected TABLE".into()));
    }
    let mut name = String::new();
    loop {
        match p.peek() {
            Some(Tok::Op(o)) if matches!(o.as_str(), "/" | "-" | "." | ":") => {
                name.push_str(o);
                p.next();
            }
            Some(Tok::Ident(w)) if name.is_empty() || !is_clause_word(w) => {
                name.push_str(w);
                p.next();
            }
            Some(Tok::Int(n)) => {
                name.push_str(&n.to_string());
                p.next();
            }
            Some(Tok::Str(s)) if name.is_empty() => {
                name = s.clone();
                p.next();
                break;
            }
            _ => break,
        }
    }
    if name.is_empty() {
        return Err(TaqlError::Parse("expected table path".into()));
    }
    let path: PathBuf = name.into();

    let mut desc = TableDesc {
        name: String::new(),
        version: String::new(),
        comment: String::new(),
        keywords: crate::record::TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        },
        private_keywords: crate::record::TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        },
        columns: Vec::new(),
    };

    // Optional bracketed column list.
    if matches!(p.peek(), Some(Tok::Op(o)) if o == "[") {
        p.next();
        loop {
            desc.columns.push(parse_column_spec(&mut p)?);
            match p.peek() {
                Some(Tok::Op(o)) if o == "," => {
                    p.next();
                }
                Some(Tok::Op(o)) if o == "]" => {
                    p.next();
                    break;
                }
                other => {
                    return Err(TaqlError::Parse(format!(
                        "expected , or ] in column list, found {other:?}"
                    )));
                }
            }
        }
    }

    // LIMIT n — the initial row count.
    let mut limit = 0;
    let mut keywords: Option<crate::record::TableRecord> = None;
    loop {
        match p.peek() {
            Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("limit") => {
                p.next();
                match p.next() {
                    Some(Tok::Int(n)) => limit = n.max(0) as u64,
                    other => {
                        return Err(TaqlError::Parse(format!(
                            "expected integer LIMIT, found {other:?}"
                        )));
                    }
                }
            }
            Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("keywords") => {
                p.next();
                // Parse a small keyword assignment list: name = value, ...
                let mut rec = crate::record::TableRecord {
                    desc: Default::default(),
                    record_type: 0,
                    values: Vec::new(),
                };
                loop {
                    let name = p.expect_ident()?;
                    p.expect_op("=")?;
                    let v = parse_ddl_literal(&mut p)?;
                    rec.set(&name, v);
                    match p.peek() {
                        Some(Tok::Op(o)) if o == "," => {
                            p.next();
                        }
                        _ => break,
                    }
                }
                keywords = Some(rec);
            }
            _ => break,
        }
    }
    if let Some(k) = keywords {
        desc.keywords = k;
    }

    let mut wt = WritableTable::create(path.clone(), desc);
    if limit > 0 {
        wt.addrows(limit);
        // Array columns need a default cell for every row (casacore's CREATE
        // fills them with zeros).
        let array_defs: Vec<(usize, DataType, Vec<i64>)> = wt
            .desc()
            .columns
            .iter()
            .enumerate()
            .filter_map(|(col_idx, col)| {
                if let ColumnKind::Array = col.kind {
                    col.shape.clone().map(|s| (col_idx, col.data_type, s))
                } else {
                    None
                }
            })
            .collect();
        for (col_idx, dt, shape) in array_defs {
            let zero = zero_array(dt, &shape);
            for r in 0..limit {
                wt.putcell(col_idx, r, zero.clone()).unwrap();
            }
        }
    }
    wt.flush()?;
    Ok(path)
}

/// A zero-filled fixed-shape array cell for a column's element type.
fn zero_array(dt: DataType, shape: &[i64]) -> RecordValue {
    let n: usize = shape.iter().map(|&d| d.max(0) as usize).product();
    let data = match dt {
        DataType::Bool => ArrayData::Bool(vec![false; n]),
        DataType::UChar => ArrayData::UChar(vec![0; n]),
        DataType::Short => ArrayData::Short(vec![0; n]),
        DataType::UShort => ArrayData::UShort(vec![0; n]),
        DataType::Int => ArrayData::Int(vec![0; n]),
        DataType::UInt => ArrayData::UInt(vec![0; n]),
        DataType::Int64 => ArrayData::Int64(vec![0; n]),
        DataType::Float => ArrayData::Float(vec![0.0; n]),
        DataType::Double => ArrayData::Double(vec![0.0; n]),
        DataType::Complex => ArrayData::Complex(vec![(0.0, 0.0); n]),
        DataType::DComplex => ArrayData::DComplex(vec![(0.0, 0.0); n]),
        DataType::String => ArrayData::String(vec![String::new(); n]),
        // Unsupported element types produce an empty array.
        _ => ArrayData::Double(Vec::new()),
    };
    RecordValue::Array(ArrayValue {
        shape: shape.iter().map(|&d| d.max(0) as u32).collect(),
        data,
    })
}

fn parse_column_spec(p: &mut Parser) -> Result<ColumnDesc, TaqlError> {
    let name = p.expect_ident()?;
    let type_tok = p.expect_ident()?;
    let (data_type, def) = parse_ddl_type(&type_tok)?;

    let mut ndim = -1i64;
    let mut shape: Option<Vec<i64>> = None;
    let mut option = 0i64;
    let mut maxlen = 0i64;
    let keywords = crate::record::TableRecord {
        desc: Default::default(),
        record_type: 0,
        values: Vec::new(),
    };

    // Column modifiers may be bare (`DATA C8 NDIM...`) or enclosed in
    // brackets (`DATA C8 [NDIM=2, SHAPE=[16,4]]`).
    let mut bracketed = false;
    if matches!(p.peek(), Some(Tok::Op(o)) if o == "[") {
        p.next();
        bracketed = true;
    }
    loop {
        match p.peek() {
            Some(Tok::Op(o)) if o == "]" && bracketed => {
                p.next();
                break;
            }
            Some(Tok::Op(o)) if o == "," && bracketed => {
                p.next();
            }
            Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("ndim") => {
                p.next();
                p.expect_op("=")?;
                match p.next() {
                    Some(Tok::Int(n)) => ndim = n,
                    other => {
                        return Err(TaqlError::Parse(format!(
                            "expected integer NDIM, found {other:?}"
                        )));
                    }
                }
            }
            Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("shape") => {
                p.next();
                p.expect_op("=")?;
                p.expect_op("[")?;
                let mut dims = Vec::new();
                loop {
                    match p.next() {
                        Some(Tok::Int(n)) => dims.push(n),
                        other => {
                            return Err(TaqlError::Parse(format!(
                                "expected integer in SHAPE, found {other:?}"
                            )));
                        }
                    }
                    match p.peek() {
                        Some(Tok::Op(o)) if o == "," => {
                            p.next();
                        }
                        Some(Tok::Op(o)) if o == "]" => {
                            p.next();
                            break;
                        }
                        other => {
                            return Err(TaqlError::Parse(format!(
                                "expected , or ] in SHAPE, found {other:?}"
                            )));
                        }
                    }
                }
                // DDL SHAPE is the logical (row-major) shape; the descriptor
                // stores the CASA order (reversed).
                shape = Some(dims.into_iter().rev().collect());
            }
            Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("option") => {
                p.next();
                p.expect_op("=")?;
                match p.next() {
                    Some(Tok::Int(n)) => option = n,
                    other => {
                        return Err(TaqlError::Parse(format!(
                            "expected integer OPTION, found {other:?}"
                        )));
                    }
                }
            }
            Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("maxlen") => {
                p.next();
                p.expect_op("=")?;
                match p.next() {
                    Some(Tok::Int(n)) => maxlen = n,
                    other => {
                        return Err(TaqlError::Parse(format!(
                            "expected integer MAXLEN, found {other:?}"
                        )));
                    }
                }
            }
            Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("comment") => {
                p.next();
                p.expect_op("=")?;
                // comment 'text' or "text"
                match p.next() {
                    Some(Tok::Str(s)) | Some(Tok::Ident(s)) => {
                        let _ = s;
                    }
                    _ => {}
                }
            }
            _ => break,
        }
    }

    // Scalar vs array: an explicit `NDIM` makes an array column (fixed shape
    // when `SHAPE` is present, variable otherwise); `R8` alone is scalar.
    let column_kind = if ndim == -1 {
        ColumnKind::Scalar(def)
    } else {
        let _ = def;
        ColumnKind::Array
    };

    Ok(ColumnDesc {
        name,
        comment: String::new(),
        data_type,
        data_manager_type: "StandardStMan".to_string(),
        data_manager_group: "StandardStMan".into(),
        options: option as i32,
        ndim: ndim as i32,
        shape,
        max_length: maxlen as i32,
        keywords,
        kind: column_kind,
    })
}

/// Map a DDL type token to a `DataType` and its default scalar value.
fn parse_ddl_type(tok: &str) -> TResult<(DataType, RecordValue)> {
    let t = tok.to_ascii_lowercase();
    let (dt, def) = match t.as_str() {
        // Empirically verified casacore DDL codes (and friendly names):
        // B/BOOL/BOOLEAN, U1/UCHAR, SHORT/I2, I4, I8, U4, R4, R8/DOUBLE,
        // C8=dcomplex, S/STRING.
        "boolean" | "bool" | "b" => (DataType::Bool, RecordValue::Bool(false)),
        "uchar" | "char" | "u1" => (DataType::UChar, RecordValue::UChar(0)),
        "short" | "i2" | "sh" => (DataType::Short, RecordValue::Short(0)),
        "int" | "i" | "i4" => (DataType::Int, RecordValue::Int(0)),
        "int64" | "i8" => (DataType::Int64, RecordValue::Int64(0)),
        "uint" | "u4" => (DataType::UInt, RecordValue::UInt(0)),
        "float" | "r4" | "f" => (DataType::Float, RecordValue::Float(0.0)),
        "double" | "r8" | "d" => (DataType::Double, RecordValue::Double(0.0)),
        "complex" | "c" | "c4" | "cf" => (DataType::Complex, RecordValue::Complex(0.0, 0.0)),
        "dcomplex" | "c8" | "cd" => (DataType::DComplex, RecordValue::DComplex(0.0, 0.0)),
        "string" | "s" => (DataType::String, RecordValue::String(String::new())),
        _ => {
            return Err(TaqlError::Eval(format!("unknown DDL type: {tok}")));
        }
    };
    Ok((dt, def))
}

/// Parse a scalar literal in a DDL KEYWORDS clause.
fn parse_ddl_literal(p: &mut Parser) -> TResult<RecordValue> {
    Ok(match p.next() {
        Some(Tok::Int(n)) => RecordValue::Int(n as i32),
        Some(Tok::Float(f)) => RecordValue::Double(f),
        Some(Tok::Str(s)) => RecordValue::String(s),
        Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("true") => RecordValue::Bool(true),
        Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("false") => RecordValue::Bool(false),
        other => {
            return Err(TaqlError::Parse(format!(
                "expected literal, found {other:?}"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::DataType;
    use crate::tabledesc::{ColumnDesc, ColumnKind};

    fn empty_record() -> crate::record::TableRecord {
        crate::record::TableRecord {
            desc: Default::default(),
            record_type: 0,
            values: Vec::new(),
        }
    }

    fn scalar(name: &str, dt: DataType, def: RecordValue) -> ColumnDesc {
        ColumnDesc {
            name: name.into(),
            comment: String::new(),
            data_type: dt,
            data_manager_type: "StandardStMan".into(),
            data_manager_group: "StandardStMan".into(),
            options: 0,
            ndim: -1,
            shape: None,
            max_length: 0,
            keywords: empty_record(),
            kind: ColumnKind::Scalar(def),
        }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p =
            std::env::temp_dir().join(format!("casacure-taql-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    /// The /tmp/tq probe table: ANT=[1,2,0,2,1] WHAT=[10,20,30,40,50]
    /// VAL=[5.5,2.0,9.0,2.0,1.0] NAME=[a..e].
    fn probe_table() -> (std::path::PathBuf, Table) {
        let dir = temp_dir("probe");
        let mut desc = TableDesc {
            name: String::new(),
            version: String::new(),
            comment: String::new(),
            keywords: empty_record(),
            private_keywords: empty_record(),
            columns: vec![
                scalar("ANT", DataType::Int, RecordValue::Int(0)),
                scalar("WHAT", DataType::Int, RecordValue::Int(0)),
                scalar("VAL", DataType::Double, RecordValue::Double(0.0)),
                scalar("NAME", DataType::String, RecordValue::String(String::new())),
            ],
        };
        desc.columns[0].name = "ANT".into();
        desc.columns[1].name = "WHAT".into();
        desc.columns[2].name = "VAL".into();
        desc.columns[3].name = "NAME".into();
        let mut wt = WritableTable::create(&dir, desc);
        wt.addrows(5);
        for (i, &ant) in [1, 2, 0, 2, 1].iter().enumerate() {
            wt.putcell(0, i as u64, RecordValue::Int(ant)).unwrap();
        }
        for (i, &what) in [10, 20, 30, 40, 50].iter().enumerate() {
            wt.putcell(1, i as u64, RecordValue::Int(what)).unwrap();
        }
        for (i, &val) in [5.5, 2.0, 9.0, 2.0, 1.0].iter().enumerate() {
            wt.putcell(2, i as u64, RecordValue::Double(val)).unwrap();
        }
        for (i, &name) in ["a", "b", "c", "d", "e"].iter().enumerate() {
            wt.putcell(3, i as u64, RecordValue::String(name.into()))
                .unwrap();
        }
        let dir = wt.flush().unwrap();
        let t = Table::open(&dir, true).unwrap();
        (dir, t)
    }

    fn ints(col: &[RecordValue]) -> Vec<i64> {
        col.iter()
            .map(|v| match v {
                RecordValue::Int64(i) => *i,
                RecordValue::Int(i) => i64::from(*i),
                other => panic!("not an int: {other:?}"),
            })
            .collect()
    }

    fn query(t: &Table, q: &str) -> TaqlTable {
        let tables = [t];
        match execute(q, &tables).unwrap() {
            TaqlResult::Query(out) => out,
            other => panic!("expected query result, got {other:?}"),
        }
    }

    #[test]
    fn rowid_orderby_matches_casacore() {
        let (_dir, t) = probe_table();
        let r = query(&t, "SELECT ROWID() AS __tablerow__ FROM $1 ORDERBY ANT");
        assert_eq!(ints(r.getcol("__tablerow__").unwrap()), [2, 0, 4, 1, 3]);
        let r = query(
            &t,
            "SELECT ROWID() AS __tablerow__ FROM $1 ORDERBY ANT, VAL",
        );
        assert_eq!(ints(r.getcol("__tablerow__").unwrap()), [2, 4, 0, 1, 3]);
        let r = query(
            &t,
            "SELECT ROWID() AS __tablerow__ FROM $1 ORDERBY VAL DESC",
        );
        assert_eq!(ints(r.getcol("__tablerow__").unwrap()), [2, 0, 3, 1, 4]);
    }

    #[test]
    fn order_by_spaced_form_matches_orderby() {
        // casacore's `ORDER BY col` (spaced) must behave like `ORDERBY`.
        let (_dir, t) = probe_table();
        let a = query(&t, "SELECT ROWID() AS r FROM $1 ORDER BY ANT");
        let b = query(&t, "SELECT ROWID() AS r FROM $1 ORDERBY ANT");
        assert_eq!(ints(a.getcol("r").unwrap()), ints(b.getcol("r").unwrap()));
        let d = query(&t, "SELECT ROWID() AS r FROM $1 ORDER BY VAL DESC");
        assert_eq!(ints(d.getcol("r").unwrap()), [2, 0, 3, 1, 4]);
    }

    #[test]
    fn where_filters_rows() {
        let (_dir, t) = probe_table();
        let r = query(&t, "SELECT ROWID() AS r FROM $1 WHERE VAL > 2.0");
        assert_eq!(ints(r.getcol("r").unwrap()), [0, 2]);
        let r = query(
            &t,
            "SELECT ROWID() AS r FROM $1 WHERE ANT = 2 AND VAL <= 2.0",
        );
        assert_eq!(ints(r.getcol("r").unwrap()), [1, 3]);
        let r = query(
            &t,
            "SELECT ROWID() AS r FROM $1 WHERE NAME != 'c' ORDERBY WHAT DESC",
        );
        assert_eq!(ints(r.getcol("r").unwrap()), [4, 3, 1, 0]);
    }

    #[test]
    fn groupby_aggregates_correct() {
        let (_dir, t) = probe_table();
        let r = query(
            &t,
            "SELECT ANT, GAGGR(WHAT) AS W, GROWID() AS ROWS_, GCOUNT() AS C, GROWID()[0] AS F FROM $1 GROUPBY ANT",
        );
        assert_eq!(r.nrows(), 3);
        assert_eq!(ints(r.getcol("ANT").unwrap()), [1, 2, 0]);
        // Group 1 rows 0,4: GAGGR(WHAT) = [10,50], GROWID [0,4].
        let g = r.getcell(1, 0).unwrap();
        match g {
            RecordValue::Array(a) => assert_eq!(ints(&a.elements()), [10, 50]),
            other => panic!("expected array, got {other:?}"),
        }
        let g = r.getcell(2, 0).unwrap();
        match g {
            RecordValue::Array(a) => assert_eq!(ints(&a.elements()), [0, 4]),
            other => panic!("expected array, got {other:?}"),
        }
        assert_eq!(ints(r.getcol("C").unwrap()), [2, 2, 1]);
        assert_eq!(ints(r.getcol("F").unwrap()), [0, 1, 2]);
        // WHERE before GROUPBY.
        let r = query(
            &t,
            "SELECT ANT, GCOUNT() AS C FROM $1 WHERE VAL > 2.0 GROUPBY ANT",
        );
        assert_eq!(ints(r.getcol("ANT").unwrap()), [1, 0]);
        assert_eq!(ints(r.getcol("C").unwrap()), [1, 1]);
    }

    #[test]
    fn select_unique_first_occurrence() {
        let (_dir, t) = probe_table();
        let r = query(&t, "SELECT UNIQUE ANT FROM $1");
        assert_eq!(ints(r.getcol("ANT").unwrap()), [1, 2, 0]);
    }

    #[test]
    fn scalar_subquery_index_lookup() {
        let (_dir, t) = probe_table();
        // $2 is the same table; look up NAME by ANT.
        let tables = [&t, &t];
        let out = match execute("SELECT [SELECT NAME FROM $2][ANT] AS N FROM $1", &tables).unwrap()
        {
            TaqlResult::Query(o) => o,
            other => panic!("{other:?}"),
        };
        let n: Vec<String> = out
            .getcol("N")
            .unwrap()
            .iter()
            .map(|v| match v {
                RecordValue::String(s) => s.clone(),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(n, ["b", "c", "a", "c", "b"]);
    }

    #[test]
    fn select_star_and_computed_columns() {
        let (_dir, t) = probe_table();
        let r = query(&t, "SELECT * FROM $1 WHERE ANT = 0");
        assert_eq!(r.colnames, ["ANT", "WHAT", "VAL", "NAME"]);
        assert_eq!(r.nrows(), 1);
        let r = query(&t, "SELECT ANT + 1 AS a1 FROM $1");
        assert_eq!(ints(r.getcol("a1").unwrap()), [2, 3, 1, 3, 2]);
    }

    #[test]
    fn ddl_create_table() {
        let base = temp_dir("ddl");
        let path = base.join("T.tab");
        let q = format!(
            "CREATE TABLE {} [FIELD_ID I4, DATA C8 [NDIM=2, SHAPE=[16,4]], NAME S] LIMIT 3",
            path.display()
        );
        match execute(&q, &[]).unwrap() {
            TaqlResult::Created(p) => assert_eq!(p, path),
            other => panic!("{other:?}"),
        }
        let t = Table::open(&path, true).unwrap();
        assert_eq!(t.nrows(), 3);
        assert_eq!(t.colnames(), ["FIELD_ID", "DATA", "NAME"]);
        // Fixed-shape array column: 16x4 dcomplex, valueType dcomplex.
        let fd_idx = t.colnames().iter().position(|c| c == "FIELD_ID").unwrap();
        let d_idx = t.colnames().iter().position(|c| c == "DATA").unwrap();
        assert!(t
            .getcoldesc(fd_idx)
            .unwrap()
            .contains(r#""valueType":"int""#));
        assert!(t
            .getcoldesc(d_idx)
            .unwrap()
            .contains(r#""valueType":"dcomplex""#));
        assert!(t
            .getcoldesc(d_idx)
            .unwrap()
            .contains(r#""shape":[16,4],"_c_order":true"#));
        // Defaults filled.
        let cell = t.getcell(d_idx, 0).unwrap();
        match cell {
            RecordValue::Array(a) => assert_eq!(a.shape, [4, 16]),
            other => panic!("{other:?}"),
        }
        let s_idx = t.colnames().iter().position(|c| c == "NAME").unwrap();
        assert_eq!(
            t.getcell(s_idx, 2).unwrap(),
            RecordValue::String(String::new())
        );
    }

    /// Numeric (int or float) cell values as f64.
    fn floats(col: &[RecordValue]) -> Vec<f64> {
        col.iter()
            .map(|v| match v {
                RecordValue::Int64(i) => *i as f64,
                RecordValue::Int(i) => f64::from(*i),
                RecordValue::Double(d) => *d,
                RecordValue::Float(f) => f64::from(*f),
                other => panic!("not numeric: {other:?}"),
            })
            .collect()
    }

    fn strings(col: &[RecordValue]) -> Vec<String> {
        col.iter()
            .map(|v| match v {
                RecordValue::String(s) => s.clone(),
                other => panic!("not a string: {other:?}"),
            })
            .collect()
    }

    fn bools(col: &[RecordValue]) -> Vec<bool> {
        col.iter()
            .map(|v| match v {
                RecordValue::Bool(b) => *b,
                other => panic!("not a bool: {other:?}"),
            })
            .collect()
    }

    /// First value of the first column of a one-row query.
    fn v1(t: &Table, q: &str) -> RecordValue {
        query(t, q).getcell(0, 0).unwrap().clone()
    }

    #[test]
    fn scalar_math_functions() {
        let (_dir, t) = probe_table();
        let one = |q: &str| floats(&[v1(&t, q)])[0];
        // Integer-preserving.
        assert_eq!(
            v1(&t, "SELECT ABS(-5) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(5)
        );
        assert_eq!(
            v1(&t, "SELECT SIGN(-3) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(-1)
        );
        assert_eq!(
            v1(&t, "SELECT INT(2.7) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(2)
        );
        assert_eq!(one("SELECT FLOOR(2.7) AS x FROM $1 LIMIT 1"), 2.0);
        assert_eq!(
            v1(&t, "SELECT SQUARE(3) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(9)
        );
        // Real-valued.
        assert_eq!(one("SELECT SQRT(16.0) AS x FROM $1 LIMIT 1"), 4.0);
        assert_eq!(one("SELECT ABS(-2.5) AS x FROM $1 LIMIT 1"), 2.5);
        assert_eq!(one("SELECT POW(2.0, 10.0) AS x FROM $1 LIMIT 1"), 1024.0);
        assert_eq!(one("SELECT ATAN2(0.0, 1.0) AS x FROM $1 LIMIT 1"), 0.0);
        assert_eq!(one("SELECT LOG10(100.0) AS x FROM $1 LIMIT 1"), 2.0);
        assert_eq!(one("SELECT LN(E()) AS x FROM $1 LIMIT 1"), 1.0);
        assert_eq!(one("SELECT LOG(E()) AS x FROM $1 LIMIT 1"), 1.0);
        assert!((one("SELECT PI() AS x FROM $1 LIMIT 1") - std::f64::consts::PI).abs() < 1e-12);
        // Math in WHERE keeps working end-to-end.
        let r = query(
            &t,
            "SELECT ROWID() AS r FROM $1 WHERE ABS(VAL) > 2.0 AND SIN(VAL) < 0.0",
        );
        assert_eq!(ints(r.getcol("r").unwrap()), [0]);
    }

    #[test]
    fn string_functions() {
        let (_dir, t) = probe_table();
        let s = |q: &str| strings(&[v1(&t, q)])[0].clone();
        assert_eq!(
            v1(&t, "SELECT LEN('hello') AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(5)
        );
        assert_eq!(s("SELECT UPPER('aBc') AS x FROM $1 LIMIT 1"), "ABC");
        assert_eq!(s("SELECT LOWER('aBc') AS x FROM $1 LIMIT 1"), "abc");
        assert_eq!(
            s("SELECT CAPITALIZE('hELLO') AS x FROM $1 LIMIT 1"),
            "Hello"
        );
        assert_eq!(s("SELECT REVERSESTRING('abc') AS x FROM $1 LIMIT 1"), "cba");
        assert_eq!(s("SELECT TRIM('  x  ') AS x FROM $1 LIMIT 1"), "x");
        assert_eq!(s("SELECT LTRIM('  x') AS x FROM $1 LIMIT 1"), "x");
        assert_eq!(s("SELECT RTRIM('x  ') AS x FROM $1 LIMIT 1"), "x");
        assert_eq!(
            s("SELECT SUBSTR('abcdef', 1, 3) AS x FROM $1 LIMIT 1"),
            "bcd"
        );
        assert_eq!(s("SELECT SUBSTR('abcdef', 2) AS x FROM $1 LIMIT 1"), "cdef");
        assert_eq!(s("SELECT SUBSTR('abcdef', -3) AS x FROM $1 LIMIT 1"), "def");
        assert_eq!(
            s("SELECT REPLACE('aXbX', 'X', '-') AS x FROM $1 LIMIT 1"),
            "a-b-"
        );
        assert_eq!(s("SELECT STR(1.5) AS x FROM $1 LIMIT 1"), "1.5");
        assert_eq!(s("SELECT STR(NAME) AS x FROM $1 WHERE NAME = 'a'"), "a");
    }

    #[test]
    fn complex_parts_functions() {
        let (_dir, t) = probe_table();
        let one = |q: &str| floats(&[v1(&t, q)])[0];
        assert_eq!(
            one("SELECT REAL(COMPLEX(3.0,4.0)) AS x FROM $1 LIMIT 1"),
            3.0
        );
        assert_eq!(
            one("SELECT IMAG(COMPLEX(3.0,4.0)) AS x FROM $1 LIMIT 1"),
            4.0
        );
        assert_eq!(
            one("SELECT AMPL(COMPLEX(3.0,4.0)) AS x FROM $1 LIMIT 1"),
            5.0
        );
        assert!(
            (one("SELECT ARG(COMPLEX(0.0,1.0)) AS x FROM $1 LIMIT 1")
                - std::f64::consts::FRAC_PI_2)
                .abs()
                < 1e-12
        );
        // conj(3,4) => (3,-4).
        let r = query(&t, "SELECT CONJ(COMPLEX(3.0,4.0)) AS x FROM $1 LIMIT 1");
        match r.getcell(0, 0).unwrap() {
            RecordValue::Array(a) => assert_eq!(floats(&a.elements()), [3.0, -4.0]),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn predicate_and_presence_functions() {
        let (_dir, t) = probe_table();
        let b = |q: &str| bools(&[v1(&t, q)])[0];
        assert!(b("SELECT ISCOLUMN('ANT') AS x FROM $1 LIMIT 1"));
        assert!(!b("SELECT ISCOLUMN('NOPE') AS x FROM $1 LIMIT 1"));
        assert!(!b("SELECT ISNULL(3) AS x FROM $1 LIMIT 1"));
        assert!(b("SELECT ISDEFINED(3) AS x FROM $1 LIMIT 1"));
        assert!(b("SELECT ISFINITE(1.5) AS x FROM $1 LIMIT 1"));
        assert!(b("SELECT ISNAN(SQRT(-1.0)) AS x FROM $1 LIMIT 1"));
        assert!(b("SELECT ISINF(1e308 * 10.0) AS x FROM $1 LIMIT 1"));
    }

    #[test]
    fn array_and_aggregate_functions() {
        let (_dir, t) = probe_table();
        let one = |q: &str| floats(&[v1(&t, q)])[0];
        assert_eq!(
            v1(&t, "SELECT SUM(ARRAY(1,2,3,4)) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(10)
        );
        assert_eq!(
            v1(&t, "SELECT MIN(ARRAY(3,1,2)) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(1)
        );
        assert_eq!(
            v1(&t, "SELECT MAX(ARRAY(3,1,2)) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(3)
        );
        assert_eq!(
            one("SELECT MEAN(ARRAY(1.0,2.0,3.0)) AS x FROM $1 LIMIT 1"),
            2.0
        );
        assert_eq!(
            one("SELECT MEDIAN(ARRAY(1.0,2.0,5.0)) AS x FROM $1 LIMIT 1"),
            2.0
        );
        assert_eq!(
            v1(&t, "SELECT NTRUE(ARRAY(1,0,1,1)) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(3)
        );
        assert!(bools(&[v1(&t, "SELECT ANY(ARRAY(0,0,1)) AS x FROM $1 LIMIT 1")])[0]);
        assert!(bools(&[v1(&t, "SELECT ALL(ARRAY(1,1,1)) AS x FROM $1 LIMIT 1")])[0]);
        // Nelements for a column's array cell.
        let r = query(&t, "SELECT NELEMENTS(GAGGR(WHAT)) AS n FROM $1 GROUPBY ANT");
        assert_eq!(ints(r.getcol("n").unwrap()), [2, 2, 1]);
        // shape() returns a [N] array.
        let r = query(&t, "SELECT SHAPE(ARRAY(1,2,3)) AS x FROM $1 LIMIT 1");
        match r.getcell(0, 0).unwrap() {
            RecordValue::Array(a) => assert_eq!(ints(&a.elements()), [3]),
            other => panic!("{other:?}"),
        }
        // reversearray / transpose / flatten.
        let r = query(&t, "SELECT REVERSEARRAY(ARRAY(1,2,3)) AS x FROM $1 LIMIT 1");
        match r.getcell(0, 0).unwrap() {
            RecordValue::Array(a) => assert_eq!(ints(&a.elements()), [3, 2, 1]),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn group_aggregate_g_functions() {
        let (_dir, t) = probe_table();
        // Groups appear in first-appearance order of the key: ANT 1, 2, 0.
        let r = query(
            &t,
            "SELECT ANT, GMIN(WHAT) AS mn, GMAX(WHAT) AS mx, GSUM(WHAT) AS s, GMEAN(WHAT) AS m, GMEDIAN(WHAT) AS med, GSUMSQR(WHAT) AS sq FROM $1 GROUPBY ANT",
        );
        assert_eq!(ints(r.getcol("ANT").unwrap()), [1, 2, 0]);
        assert_eq!(ints(r.getcol("mn").unwrap()), [10, 20, 30]);
        assert_eq!(ints(r.getcol("mx").unwrap()), [50, 40, 30]);
        assert_eq!(ints(r.getcol("s").unwrap()), [60, 60, 30]);
        assert_eq!(floats(r.getcol("m").unwrap()), [30.0, 30.0, 30.0]);
        assert_eq!(floats(r.getcol("med").unwrap()), [30.0, 30.0, 30.0]);
        // Groups: [10,50] -> 2600; [20,40] -> 2000; [30] -> 900.
        assert_eq!(ints(r.getcol("sq").unwrap()), [2600, 2000, 900]);
        // GFIRST / GLAST keep the raw cell values.
        let r = query(
            &t,
            "SELECT ANT, GFIRST(NAME) AS f, GLAST(NAME) AS l FROM $1 WHERE ANT = 1 GROUPBY ANT",
        );
        assert_eq!(strings(r.getcol("f").unwrap()), ["a"]);
        assert_eq!(strings(r.getcol("l").unwrap()), ["e"]);
        // GFRACTILE over a float column.
        let r = query(
            &t,
            "SELECT ANT, GFRACTILE(VAL, 0.5) AS med FROM $1 WHERE ANT = 1 GROUPBY ANT",
        );
        // VAL for group ANT=1: rows 0 (5.5) and 4 (1.0); median = 3.25.
        assert!((floats(r.getcol("med").unwrap())[0] - 3.25).abs() < 1e-12);
    }

    #[test]
    fn like_pattern_matching() {
        let (_dir, t) = probe_table();
        let rows = |q: &str| ints(query(&t, q).getcol("r").unwrap());
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE NAME LIKE 'a%'"),
            [0]
        );
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE NAME LIKE '%c%'"),
            [2]
        );
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE NAME LIKE '_'"),
            [0, 1, 2, 3, 4]
        );
        assert_eq!(rows("SELECT ROWID() AS r FROM $1 WHERE NAME LIKE 'a'"), [0]);
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE NAME ILIKE 'C'"),
            [2]
        );
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE NAME NOT LIKE 'a%'"),
            [1, 2, 3, 4]
        );
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE NAME LIKE '%'"),
            [0, 1, 2, 3, 4]
        );
        // Escaped literal: NAME LIKE '\%' matches nothing (no literal % names).
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE NAME LIKE '\\%'"),
            []
        );
    }

    #[test]
    fn in_set_membership() {
        let (_dir, t) = probe_table();
        let rows = |q: &str| ints(query(&t, q).getcol("r").unwrap());
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE ANT IN (1, 2)"),
            [0, 1, 3, 4]
        );
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE ANT NOT IN (1)"),
            [1, 2, 3]
        );
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE VAL IN (2.0, 9.0)"),
            [1, 2, 3]
        );
        // Brace style set form.
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE ANT IN [1, 2]"),
            [0, 1, 3, 4]
        );
        // Empty set matches nothing.
        assert_eq!(rows("SELECT ROWID() AS r FROM $1 WHERE ANT IN ()"), []);
        // IN in a projection expression.
        let r = query(&t, "SELECT ANT IN (1) AS b FROM $1 WHERE ANT = 1 LIMIT 1");
        assert_eq!(bools(r.getcol("b").unwrap()), [true]);
    }

    #[test]
    fn having_filters_groups() {
        let (_dir, t) = probe_table();
        let r = query(
            &t,
            "SELECT ANT, GCOUNT() AS C FROM $1 GROUPBY ANT HAVING GCOUNT() > 1",
        );
        assert_eq!(ints(r.getcol("ANT").unwrap()), [1, 2]);
        assert_eq!(ints(r.getcol("C").unwrap()), [2, 2]);
        // HAVING over an aggregate column value.
        let r = query(
            &t,
            "SELECT ANT, GSUM(WHAT) AS s FROM $1 GROUPBY ANT HAVING GSUM(WHAT) > 30",
        );
        assert_eq!(ints(r.getcol("ANT").unwrap()), [1, 2]);
        assert_eq!(ints(r.getcol("s").unwrap()), [60, 60]);
        // WHERE runs before GROUPBY, HAVING after.
        let r = query(
            &t,
            "SELECT ANT FROM $1 WHERE VAL > 2.0 GROUPBY ANT HAVING GCOUNT() > 0",
        );
        assert_eq!(ints(r.getcol("ANT").unwrap()), [1, 0]);
    }

    #[test]
    fn offset_skips_leading_rows() {
        let (_dir, t) = probe_table();
        let rows = |q: &str| ints(query(&t, q).getcol("r").unwrap());
        // Ordered by ANT: rows [2,0,4,1,3]; skipping 1 leaves [0,4,1,3].
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 ORDERBY ANT OFFSET 1"),
            [0, 4, 1, 3]
        );
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 ORDERBY ANT LIMIT 2 OFFSET 1"),
            [0, 4]
        );
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 ORDERBY ANT OFFSET 99"),
            []
        );
    }

    #[test]
    fn grouped_orderby() {
        let (_dir, t) = probe_table();
        let r = query(
            &t,
            "SELECT ANT, GCOUNT() AS C FROM $1 GROUPBY ANT ORDERBY ANT",
        );
        assert_eq!(ints(r.getcol("ANT").unwrap()), [0, 1, 2]);
        let r = query(
            &t,
            "SELECT ANT, GCOUNT() AS C FROM $1 GROUPBY ANT ORDERBY ANT DESC",
        );
        assert_eq!(ints(r.getcol("ANT").unwrap()), [2, 1, 0]);
        // HAVING + ORDERBY + LIMIT compose on grouped output.
        let r = query(
            &t,
            "SELECT ANT FROM $1 GROUPBY ANT HAVING GCOUNT() > 1 ORDERBY ANT DESC",
        );
        assert_eq!(ints(r.getcol("ANT").unwrap()), [2, 1]);
    }

    #[test]
    fn count_command() {
        let (_dir, t) = probe_table();
        let c = |q: &str| ints(query(&t, q).getcol("count").unwrap())[0];
        assert_eq!(c("COUNT * FROM $1"), 5);
        assert_eq!(c("COUNT col FROM $1 WHERE VAL > 2.0"), 2);
        assert_eq!(c("COUNT * FROM $1 WHERE ANT IN (1, 2)"), 4);
    }

    #[test]
    fn date_time_functions() {
        let (_dir, t) = probe_table();
        let one = |q: &str| floats(&[v1(&t, q)])[0];
        let v = |q: &str| v1(&t, q);
        let s = |q: &str| strings(&[v1(&t, q)])[0].clone();
        // Anchored to real casacore 3.8.1 TaQL output.
        assert_eq!(
            v("SELECT YEAR(51544.0) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(2000)
        );
        assert_eq!(
            v("SELECT MONTH(51544.0) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(1)
        );
        assert_eq!(
            v("SELECT DAY(51544.0) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(1)
        );
        assert_eq!(
            v("SELECT WEEKDAY(51544.0) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(6)
        );
        assert_eq!(
            v("SELECT DOW(0.0) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(3)
        );
        assert_eq!(
            v("SELECT WEEK(51544.0) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(0)
        );
        assert_eq!(
            v("SELECT WEEK(58849.0) AS x FROM $1 LIMIT 1"),
            RecordValue::Int64(1)
        );
        assert_eq!(one("SELECT DATE(51544.75) AS x FROM $1 LIMIT 1"), 51544.0);
        assert!(
            (one("SELECT TIME(51544.5) AS x FROM $1 LIMIT 1") - std::f64::consts::PI).abs() < 1e-12
        );
        // String-to-MJD parsing.
        assert_eq!(
            one("SELECT MJD('2000-01-01') AS x FROM $1 LIMIT 1"),
            51544.0
        );
        assert_eq!(
            one("SELECT MJD('2000-01-01 12:00:00') AS x FROM $1 LIMIT 1"),
            51544.5
        );
        assert_eq!(
            one("SELECT DATETIME('1858-11-17') AS x FROM $1 LIMIT 1"),
            0.0
        );
        // Named / formatted output.
        assert_eq!(s("SELECT CMONTH(51544.0) AS x FROM $1 LIMIT 1"), "Jan");
        assert_eq!(s("SELECT CDOW(51544.0) AS x FROM $1 LIMIT 1"), "Sat");
        assert_eq!(
            s("SELECT CDATE(51544.0) AS x FROM $1 LIMIT 1"),
            "01-Jan-2000"
        );
        assert_eq!(
            s("SELECT CTIME(MJD('2000-01-01 06:30:00')) AS x FROM $1 LIMIT 1"),
            "06:30:00.000"
        );
        assert_eq!(
            s("SELECT CTOD(MJD('2000-01-01 06:30:00')) AS x FROM $1 LIMIT 1"),
            "2000/01/01/06:30:00.000"
        );
    }

    #[test]
    fn plural_statistic_variants() {
        let (_dir, t) = probe_table();
        let arr = |q: &str| -> Vec<i64> {
            let r = query(&t, q);
            match r.getcell(0, 0).unwrap() {
                RecordValue::Array(a) => ints(&a.elements()),
                other => panic!("{other:?}"),
            }
        };
        assert_eq!(
            arr("SELECT SUMS(ARRAY(1,2,3), ARRAY(10,20,30)) AS x FROM $1 LIMIT 1"),
            [11, 22, 33]
        );
        assert_eq!(
            arr("SELECT PRODUCTS(ARRAY(1,2), ARRAY(3,4)) AS x FROM $1 LIMIT 1"),
            [3, 8]
        );
        assert_eq!(
            arr("SELECT MINS(ARRAY(3,1), ARRAY(0,5)) AS x FROM $1 LIMIT 1"),
            [0, 1]
        );
        assert_eq!(
            arr("SELECT MAXS(ARRAY(3,1), ARRAY(0,5)) AS x FROM $1 LIMIT 1"),
            [3, 5]
        );
        // Mismatched lengths are an error.
        assert!(matches!(
            execute("SELECT SUMS(ARRAY(1,2), ARRAY(1,2,3)) AS x FROM $1", &[&t]),
            Err(TaqlError::Eval(_))
        ));
    }

    #[test]
    fn update_statement() {
        let (_dir, t) = probe_table();
        let q = format!("UPDATE '{}' SET VAL = 0.0 WHERE ANT = 1", _dir.display());
        match execute(&q, &[&t]).unwrap() {
            TaqlResult::Query(_) => {}
            other => panic!("{other:?}"),
        }
        let t2 = Table::open(&_dir, true).unwrap();
        let val = t2.getcell(2, 0).unwrap();
        assert_eq!(val, RecordValue::Double(0.0));
        assert_eq!(t2.getcell(2, 4).unwrap(), RecordValue::Double(0.0));
        // Unmatched rows untouched.
        assert_eq!(t2.getcell(2, 1).unwrap(), RecordValue::Double(2.0));
        // Computed update expression in a row context.
        let q = format!(
            "UPDATE '{}' SET WHAT = WHAT * 10 WHERE ANT = 2",
            _dir.display()
        );
        execute(&q, &[&t]).unwrap();
        let t3 = Table::open(&_dir, true).unwrap();
        assert_eq!(t3.getcell(1, 1).unwrap(), RecordValue::Int(200));
        assert_eq!(t3.getcell(1, 3).unwrap(), RecordValue::Int(400));
    }

    #[test]
    fn mutating_statements_report_touched_dirs() {
        let (dir, t) = probe_table();
        // UPDATE reports the target directory, so an embedding can
        // invalidate cached state for it (the stale-read / clobber bug the
        // pyo3 layer pinned).
        let mut touched = Vec::new();
        let q = format!("UPDATE '{}' SET VAL = 0.0 WHERE ANT = 1", dir.display());
        execute_into(&q, &[&t], &mut touched).unwrap();
        assert_eq!(touched, vec![dir.clone()]);
        touched.clear();
        let q = format!("DELETE FROM '{}' WHERE ANT = 1", dir.display());
        execute_into(&q, &[&t], &mut touched).unwrap();
        assert_eq!(touched, vec![dir.clone()]);
        touched.clear();
        let q = format!("ALTER TABLE '{}' ADD COLUMN X double", dir.display());
        execute_into(&q, &[&t], &mut touched).unwrap();
        assert_eq!(touched, vec![dir.clone()]);
        touched.clear();
        // Read-only statements report nothing.
        let q = format!("SELECT ANT FROM '{}' WHERE ANT = 1", dir.display());
        execute_into(&q, &[&t], &mut touched).unwrap();
        assert!(touched.is_empty());
        touched.clear();
        let q = format!("SHOW TABLE '{}'", dir.display());
        execute_into(&q, &[&t], &mut touched).unwrap();
        assert!(touched.is_empty());
    }

    #[test]
    fn insert_coerces_result_values_to_column_type() {
        let (_dir, t) = probe_table();
        let mut wt = WritableTable::create(temp_dir("insc"), t.dat.desc.clone());
        let target = wt.flush().unwrap();
        // ANT is an Int column; the SELECT expression yields Int64 values.
        // They must be narrowed to the column type, not stored raw (which
        // wrote zero for the inserted cell before the coercion fix).
        let q = format!(
            "INSERT INTO '{}' (ANT) SELECT ANT*100 FROM $1 WHERE ANT = 1",
            target.display()
        );
        match execute(&q, &[&t]).unwrap() {
            TaqlResult::Query(_) => {}
            other => panic!("{other:?}"),
        }
        let t2 = Table::open(&target, true).unwrap();
        assert_eq!(t2.nrows(), 2);
        assert_eq!(
            t2.getcol(0, 0, 2).unwrap(),
            vec![RecordValue::Int(100), RecordValue::Int(100)]
        );
    }

    #[test]
    fn delete_statement() {
        let (_dir, t) = probe_table();
        let q = format!("DELETE FROM '{}' WHERE ANT = 1", _dir.display());
        execute(&q, &[&t]).unwrap();
        let t2 = Table::open(&_dir, true).unwrap();
        assert_eq!(t2.nrows(), 3);
        // Surviving ANT values in order: [2, 0, 2].
        assert_eq!(
            t2.getcol(0, 0, 3).unwrap(),
            vec![
                RecordValue::Int(2),
                RecordValue::Int(0),
                RecordValue::Int(2)
            ]
        );
        // DELETE with no WHERE removes every row.
        let q = format!("DELETE FROM '{}'", _dir.display());
        execute(&q, &[]).unwrap();
        let t3 = Table::open(&_dir, true).unwrap();
        assert_eq!(t3.nrows(), 0);
    }

    #[test]
    fn insert_into_statement() {
        let (_dir, t) = probe_table();
        let mut wt = WritableTable::create(temp_dir("ins"), t.dat.desc.clone());
        let target = wt.flush().unwrap();
        // INSERT INTO target SELECT ... FROM $1 (the probe) WHERE ANT = 2.
        let q = format!(
            "INSERT INTO '{}' SELECT ANT, WHAT, VAL, NAME FROM $1 WHERE ANT = 2",
            target.display()
        );
        match execute(&q, &[&t]).unwrap() {
            TaqlResult::Query(_) => {}
            other => panic!("{other:?}"),
        }
        let t2 = Table::open(&target, true).unwrap();
        assert_eq!(t2.nrows(), 2);
        assert_eq!(
            t2.getcol(0, 0, 2).unwrap(),
            vec![RecordValue::Int(2), RecordValue::Int(2)]
        );
        assert_eq!(
            t2.getcol(3, 0, 2).unwrap(),
            vec![
                RecordValue::String("b".into()),
                RecordValue::String("d".into())
            ]
        );
    }

    #[test]
    fn select_into_statement() {
        let (_dir, t) = probe_table();
        let out = temp_dir("into");
        let target = out.join("R.tab");
        let q = format!(
            "SELECT ANT, GCOUNT() AS C INTO '{}' FROM $1 GROUPBY ANT",
            target.display()
        );
        match execute(&q, &[&t]).unwrap() {
            TaqlResult::Created(p) => assert_eq!(p, target),
            other => panic!("{other:?}"),
        }
        let t2 = Table::open(&target, true).unwrap();
        assert_eq!(t2.colnames(), ["ANT", "C"]);
        assert_eq!(t2.nrows(), 3);
        // Grouped query results are Int64-typed.
        assert_eq!(
            t2.getcol(0, 0, 3).unwrap(),
            vec![
                RecordValue::Int64(1),
                RecordValue::Int64(2),
                RecordValue::Int64(0)
            ]
        );
        assert_eq!(
            t2.getcol(1, 0, 3).unwrap(),
            vec![
                RecordValue::Int64(2),
                RecordValue::Int64(2),
                RecordValue::Int64(1)
            ]
        );
    }

    #[test]
    fn droptable_statement() {
        let base = temp_dir("drop");
        let path = base.join("T.tab");
        let mut wt = WritableTable::create(
            path.clone(),
            crate::tabledesc::TableDesc {
                name: String::new(),
                version: String::new(),
                comment: String::new(),
                keywords: empty_record(),
                private_keywords: empty_record(),
                columns: vec![scalar("X", DataType::Int, RecordValue::Int(0))],
            },
        );
        wt.addrows(1);
        wt.putcell(0, 0, RecordValue::Int(1)).unwrap();
        let p = wt.flush().unwrap();
        assert!(p.is_dir());
        let q = format!("DROPTABLE '{}'", p.display());
        execute(&q, &[]).unwrap();
        assert!(!p.exists());
    }

    #[test]
    fn alter_table_statement() {
        let (_dir, t) = probe_table();
        // RENAME + ADD + DROP + keyword ops, applied sequentially.
        let q = format!("ALTER TABLE '{}' RENAME COLUMN ANT TO ANT1", _dir.display());
        execute(&q, &[&t]).unwrap();
        let t2 = Table::open(&_dir, true).unwrap();
        assert!(t2.colnames().contains(&"ANT1".to_string()));
        assert!(!t2.colnames().contains(&"ANT".to_string()));

        let q = format!("ALTER TABLE '{}' ADD COLUMN NEW I4", _dir.display());
        execute(&q, &[&t]).unwrap();
        let t2 = Table::open(&_dir, true).unwrap();
        assert!(t2.colnames().contains(&"NEW".to_string()));
        // Existing rows default to 0.
        assert_eq!(t2.getcell(4, 0).unwrap(), RecordValue::Int(0));

        let q = format!("ALTER TABLE '{}' DROP COLUMN WHAT", _dir.display());
        execute(&q, &[&t]).unwrap();
        let t2 = Table::open(&_dir, true).unwrap();
        assert!(!t2.colnames().contains(&"WHAT".to_string()));
        assert_eq!(t2.colnames().len(), 4); // ANT1, VAL, NAME, NEW

        let q = format!("ALTER TABLE '{}' SET MSVER = 3", _dir.display());
        execute(&q, &[&t]).unwrap();
        let t2 = Table::open(&_dir, true).unwrap();
        assert!(t2.getkeywords().contains(r#""MSVER":3"#));
    }

    #[test]
    fn show_table_and_calc() {
        let (_dir, t) = probe_table();
        let q = format!("SHOW TABLE '{}'", _dir.display());
        let r = match execute(&q, &[&t]).unwrap() {
            TaqlResult::Query(r) => r,
            other => panic!("{other:?}"),
        };
        assert_eq!(r.colnames[0], "name");
        assert_eq!(r.nrows(), 4);
        let names: Vec<String> = strings(&r.columns[0]);
        assert_eq!(names, ["ANT", "WHAT", "VAL", "NAME"]);

        // CALC evaluates a row-context expression at row 0.
        let r = match execute("CALC 2 + 3 FROM $1", &[&t]).unwrap() {
            TaqlResult::Query(r) => r,
            other => panic!("{other:?}"),
        };
        assert_eq!(ints(&r.columns[0]), [5]);
        let r = match execute("CALC VAL * 2 FROM $1", &[&t]).unwrap() {
            TaqlResult::Query(r) => r,
            other => panic!("{other:?}"),
        };
        assert_eq!(floats(&r.columns[0]), [11.0]);
    }

    #[test]
    fn running_and_boxed_statistics() {
        let (_dir, t) = probe_table();
        let arr = |q: &str| -> Vec<i64> {
            let r = query(&t, q);
            match r.getcell(0, 0).unwrap() {
                RecordValue::Array(a) => ints(&a.elements()),
                other => panic!("{other:?}"),
            }
        };
        let farr = |q: &str| -> Vec<f64> {
            let r = query(&t, q);
            match r.getcell(0, 0).unwrap() {
                RecordValue::Array(a) => floats(&a.elements()),
                other => panic!("{other:?}"),
            }
        };
        assert_eq!(
            arr("SELECT RUNNINGSUM(ARRAY(1,2,3)) AS x FROM $1 LIMIT 1"),
            [1, 3, 6]
        );
        assert_eq!(
            arr("SELECT RUNNINGMIN(ARRAY(3,1,2)) AS x FROM $1 LIMIT 1"),
            [3, 1, 1]
        );
        assert_eq!(
            arr("SELECT RUNNINGMAX(ARRAY(3,1,2)) AS x FROM $1 LIMIT 1"),
            [3, 3, 3]
        );
        assert_eq!(
            farr("SELECT BOXEDMEAN(ARRAY(1.0,2.0,3.0,4.0)) AS x FROM $1 LIMIT 1"),
            [1.0, 1.5, 2.0, 2.5]
        );
        assert_eq!(
            arr("SELECT RUNNINGNTRUE(ARRAY(0,1,1,0)) AS x FROM $1 LIMIT 1"),
            [0, 1, 2, 2]
        );
    }

    #[test]
    fn sexagesimal_formatting() {
        let (_dir, t) = probe_table();
        let s = |q: &str| strings(&[v1(&t, q)])[0].clone();
        // Anchored to real casacore 3.8.1 output.
        assert_eq!(s("SELECT HMS(PI()/2) AS x FROM $1 LIMIT 1"), "06h00m00.000");
        assert_eq!(
            s("SELECT DMS(PI()/2) AS x FROM $1 LIMIT 1"),
            "+090d00m00.000"
        );
        // HDMS over an array of radians.
        let r = query(&t, "SELECT HDMS(ARRAY(PI()/2, PI())) AS x FROM $1 LIMIT 1");
        match r.getcell(0, 0).unwrap() {
            RecordValue::Array(a) => {
                let v = strings(&a.elements());
                assert_eq!(v, ["+090d00m00.000", "+180d00m00.000"]);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn regex_operator() {
        let (_dir, t) = probe_table();
        let rows = |q: &str| ints(query(&t, q).getcol("r").unwrap());
        assert_eq!(rows("SELECT ROWID() AS r FROM $1 WHERE NAME ~ 'a'"), [0]);
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE NAME ~ regex('[b-d]')"),
            [1, 2, 3]
        );
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE NAME !~ regex('a')"),
            [1, 2, 3, 4]
        );
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE NAME ~ sqlpattern('%c%')"),
            [2]
        );
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE NAME ~ pattern('*e')"),
            [4]
        );
        // Anchored regex.
        assert_eq!(
            rows("SELECT ROWID() AS r FROM $1 WHERE NAME ~ regex('^[a-c]$')"),
            [0, 1, 2]
        );
        // A doubled string rides through LEN: ensure the regex value is not
        // stringified to an empty integer in outputs.
        let r = query(&t, "SELECT NAME ~ regex('x') AS v FROM $1 LIMIT 1");
        assert_eq!(bools(r.getcol("v").unwrap()), [false]);
    }
}
