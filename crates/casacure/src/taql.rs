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
}

impl PartialEq for TqValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TqValue::Bool(a), TqValue::Bool(b)) => a == b,
            (TqValue::Int(a), TqValue::Int(b)) => a == b,
            (TqValue::Float(a), TqValue::Float(b)) => a.to_bits() == b.to_bits(),
            (TqValue::Str(a), TqValue::Str(b)) => a == b,
            (TqValue::Arr(a), TqValue::Arr(b)) => a == b,
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
                } else if ["==", "!=", "<=", ">=", "<>", "&&", "||"].contains(&two(rest)) {
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

/// After an unquoted table-name start, consume trailing path fragment
/// tokens (`/`, `-`, `.`, `:`) joined into the name, stopping at a clause
/// keyword or bracket.
fn join_table_path_chars(p: &mut Parser) -> String {
    let mut s = String::new();
    loop {
        match p.peek() {
            Some(Tok::Op(o)) if matches!(o.as_str(), "/" | "-" | "." | ":") => {
                s.push_str(o);
                p.next();
            }
            Some(Tok::Ident(w)) if !is_clause_word(w) => {
                s.push_str(w);
                p.next();
            }
            _ => break,
        }
    }
    s
}

/// Words that end an unquoted table path.
fn is_clause_word(w: &str) -> bool {
    matches!(
        w.to_ascii_lowercase().as_str(),
        "where"
            | "orderby"
            | "groupby"
            | "limit"
            | "as"
            | "unique"
            | "desc"
            | "asc"
            | "from"
            | "and"
            | "or"
            | "not"
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
}

#[derive(Debug, Clone)]
struct Select {
    unique: bool,
    columns: Option<Vec<(Expr, Option<String>)>>,
    table: TableRef,
    where_: Option<Expr>,
    orderby: Vec<OrderKey>,
    groupby: Vec<Expr>,
    limit: Option<i64>,
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
        let from = self.expect_ident()?;
        if !from.eq_ignore_ascii_case("from") {
            return Err(TaqlError::Parse(format!("expected FROM, found {from}")));
        }
        let table = match self.next() {
            Some(Tok::Ident(s)) if s.starts_with('$') => {
                let n: usize = s[1..]
                    .parse()
                    .map_err(|_| TaqlError::Parse(format!("invalid table reference {s}")))?;
                if n == 0 {
                    return Err(TaqlError::Parse("table references are 1-based".into()));
                }
                TableRef::Table(n)
            }
            Some(Tok::Str(s)) => TableRef::Path(s),
            Some(tok) => {
                // Unquoted table name, possibly with / separators.
                let mut name = match tok {
                    Tok::Ident(s) => s,
                    other => {
                        return Err(TaqlError::Parse(format!(
                            "expected table reference, found {other:?}"
                        )));
                    }
                };
                name.push_str(&join_table_path_chars(self));
                while name.ends_with('/') {
                    name.pop();
                }
                TableRef::Path(name)
            }
            None => {
                return Err(TaqlError::Parse(
                    "expected table reference, found end of query".into(),
                ));
            }
        };

        let mut where_ = None;
        let mut orderby = Vec::new();
        let mut groupby = Vec::new();
        let mut limit = None;
        loop {
            match self.peek() {
                Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("where") => {
                    self.next();
                    where_ = Some(self.parse_expr()?);
                }
                Some(Tok::Ident(w)) if w.eq_ignore_ascii_case("orderby") => {
                    self.next();
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
            limit,
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
        let op = match self.peek() {
            Some(Tok::Op(o)) if o == "==" || o == "=" => Some(BinOp::Eq),
            Some(Tok::Op(o)) if o == "!=" || o == "<>" => Some(BinOp::Ne),
            Some(Tok::Op(o)) if o == "<" => Some(BinOp::Lt),
            Some(Tok::Op(o)) if o == "<=" => Some(BinOp::Le),
            Some(Tok::Op(o)) if o == ">" => Some(BinOp::Gt),
            Some(Tok::Op(o)) if o == ">=" => Some(BinOp::Ge),
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
    let toks = tokenize(query)?;
    let mut p = Parser { toks, pos: 0 };
    let kw = match p.peek() {
        Some(Tok::Ident(w)) => w.clone(),
        _ => return Err(TaqlError::Parse("empty query".into())),
    };
    if kw.eq_ignore_ascii_case("create") {
        return Ok(TaqlResult::Created(create_table(query)?));
    }
    let sel = p.parse_select()?;
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
            let v = ctx.eval_group(e, group)?;
            vals.push(v.to_record());
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
            let v = ctx.eval_row(e, r)?;
            vals.push(v.to_record());
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
            Expr::Binary(op, l, r) => self.eval_binary(op, l, r, row, false),
            Expr::Index(base, idx) => {
                let base_v = self.eval_row(base, row)?;
                let idx_v = self.eval_row(idx, row)?;
                index_value(base_v, idx_v)
            }
            Expr::Subquery(sel) => {
                let sub = run_select(sel, self.tables)?;
                Ok(TqValue::Subtable(std::rc::Rc::new(sub)))
            }
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
            Expr::Binary(op, l, r) => self.eval_binary(op, l, r, group[0], true),
            Expr::Index(base, idx) => {
                let base_v = self.eval_group(base, group)?;
                let idx_v = self.eval_group(idx, group)?;
                index_value(base_v, idx_v)
            }
            Expr::Subquery(sel) => {
                let sub = run_select(sel, self.tables)?;
                Ok(TqValue::Subtable(std::rc::Rc::new(sub)))
            }
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
            "ROWID" => Ok(TqValue::Int(row)),
            "ABS" => match self.eval_row(&args[0], row)? {
                TqValue::Int(i) => Ok(TqValue::Int(i.abs())),
                TqValue::Float(f) => Ok(TqValue::Float(f.abs())),
                other => Err(TaqlError::Eval(format!("abs({other:?})"))),
            },
            "MIN" | "MAX" | "SUM" | "SQRT" | "FLOOR" | "CEIL" | "ROUND" => {
                let mut vals = Vec::new();
                for a in args {
                    vals.push(self.eval_row(a, row)?);
                }
                math_func(&upper, &vals)
            }
            "ISNAN" | "ISINF" => match self.eval_row(&args[0], row)? {
                TqValue::Float(f) => Ok(TqValue::Bool(match upper.as_str() {
                    "ISNAN" => f.is_nan(),
                    _ => f.is_infinite(),
                })),
                _ => Ok(TqValue::Bool(false)),
            },
            _ => Err(TaqlError::Unknown(name.to_string())),
        }
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
            _ => self.call_row(name, args, group.first().copied().unwrap_or(0)),
        }
    }

    fn eval_binary(
        &self,
        op: &BinOp,
        l: &Expr,
        r: &Expr,
        row: i64,
        group: bool,
    ) -> TResult<TqValue> {
        let lv = if group {
            return Err(TaqlError::Eval("group binary not supported".into()));
        } else {
            self.eval_row(l, row)?
        };
        let rv = self.eval_row(r, row)?;
        binary_value(op, &lv, &rv)
    }
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

fn math_func(fname: &str, args: &[TqValue]) -> TResult<TqValue> {
    let mut nums = Vec::with_capacity(args.len());
    for v in args {
        match v {
            TqValue::Int(i) => nums.push(*i as f64),
            TqValue::Float(f) => nums.push(*f),
            other => {
                return Err(TaqlError::Eval(format!("{fname} on non-number {other:?}")));
            }
        }
    }
    Ok(match fname {
        "MIN" => TqValue::Float(nums.iter().copied().fold(f64::INFINITY, f64::min)),
        "MAX" => TqValue::Float(nums.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        "SUM" => TqValue::Float(nums.iter().sum()),
        "SQRT" => TqValue::Float(nums[0].sqrt()),
        "FLOOR" => TqValue::Float(nums[0].floor()),
        "CEIL" => TqValue::Float(nums[0].ceil()),
        "ROUND" => TqValue::Float(nums[0].round()),
        _ => unreachable!(),
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
}
