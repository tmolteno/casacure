//! A minimal FITS reader for the image subsystem (issue #14): the subset
//! DDFacet actually opens — plain single-HDU primary IMAGE units written by
//! astropy (`fits.PrimaryHDU`), BITPIX 8/16/32/64/±float, BSCALE/BZERO, and
//! the WCS header cards.  Hand-rolled on purpose: no new dependency, one
//! header model shared with the CASA-image path, and the data is converted
//! straight from the mapped file (see `datafile::Buffer`).

use std::ops::Deref;

use thiserror::Error;

use crate::record::ArrayData;

#[derive(Debug, Error)]
pub enum FitsError {
    #[error("{path}: not a FITS file (no SIMPLE card)")]
    NotFits { path: std::path::PathBuf },
    #[error("{path}: {source}")]
    Io {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("{path}: malformed card at byte {offset}: {why}")]
    Card {
        path: std::path::PathBuf,
        offset: usize,
        why: String,
    },
    #[error("{path}: unsupported FITS feature: {what}")]
    Unsupported {
        path: std::path::PathBuf,
        what: String,
    },
    #[error("{path}: {msg}")]
    Data {
        path: std::path::PathBuf,
        msg: String,
    },
}

/// A parsed FITS card value.
#[derive(Debug, Clone, PartialEq)]
pub enum CardValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl CardValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            CardValue::Int(i) => Some(*i as f64),
            CardValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            CardValue::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// The header of the primary IMAGE unit: keyword -> value cards in file
/// order (duplicates keep the last), plus the comment text of value cards.
#[derive(Debug, Clone, Default)]
pub struct FitsHeader {
    pub cards: Vec<(String, Option<CardValue>)>,
}

impl FitsHeader {
    pub fn get(&self, keyword: &str) -> Option<&CardValue> {
        let keyword = keyword.to_ascii_uppercase();
        self.cards
            .iter()
            .rev()
            .find(|(k, v)| *k == keyword && v.is_some())
            .and_then(|(_, v)| v.as_ref())
    }

    pub fn f64_of(&self, keyword: &str) -> Option<f64> {
        self.get(keyword).and_then(|v| v.as_f64())
    }

    pub fn string_of(&self, keyword: &str) -> Option<&str> {
        self.get(keyword).and_then(|v| v.as_str())
    }

    /// `CTYPEn`, `CRPIXn` ... — the per-axis WCS cards (n is the FITS axis,
    /// 1-based).
    pub fn axis_f64(&self, prefix: &str, n: usize) -> Option<f64> {
        self.f64_of(&format!("{prefix}{n}"))
    }

    pub fn axis_str(&self, prefix: &str, n: usize) -> Option<&str> {
        self.string_of(&format!("{prefix}{n}"))
    }
}

/// A primary-image FITS file: the parsed header, the data's byte offset in
/// the mapped file, and the axis/data descriptions.
#[derive(Debug)]
pub struct FitsImage {
    pub path: std::path::PathBuf,
    pub header: FitsHeader,
    /// FITS axis lengths, NAXIS1 (fastest) first.
    pub naxes: Vec<usize>,
    pub bitpix: i32,
    pub bscale: f64,
    pub bzero: f64,
    /// Offset of the first data byte within the file.
    data_offset: u64,
    data_len: usize,
    map: crate::datafile::Buffer,
}

impl Deref for FitsImage {
    type Target = FitsHeader;

    fn deref(&self) -> &FitsHeader {
        &self.header
    }
}

fn parse_card(
    card: &[u8],
    path: &std::path::Path,
    offset: usize,
) -> Result<(String, Option<CardValue>), FitsError> {
    let bad = |why: String| FitsError::Card {
        path: path.to_path_buf(),
        offset,
        why,
    };
    let keyword: String = String::from_utf8_lossy(&card[..8]).trim_end().to_string();
    let rest = &card[8..];
    if keyword == "END" || keyword == "COMMENT" || keyword == "HISTORY" || keyword.is_empty() {
        return Ok((keyword, None));
    }
    if rest.first() != Some(&b'=') {
        // A commentary-style card with no value.
        return Ok((keyword, None));
    }
    let value = String::from_utf8_lossy(&rest[1..]);
    let value = value.trim();
    let parsed = if let Some(rest) = value.strip_prefix('\'') {
        // String value: closes at the first unescaped quote; trailing
        // spaces before the closing quote are stripped.
        let mut out = String::new();
        let mut chars = rest.char_indices();
        while let Some((i, ch)) = chars.next() {
            if ch == '\'' {
                if rest[i + 1..].starts_with('\'') {
                    out.push('\'');
                    chars.next();
                } else {
                    break;
                }
            } else {
                out.push(ch);
            }
        }
        CardValue::Str(out.trim_end().to_string())
    } else {
        let token = value.split('/').next().unwrap_or("").trim();
        match token {
            "T" => CardValue::Bool(true),
            "F" => CardValue::Bool(false),
            "" => return Ok((keyword, None)),
            _ => {
                let is_float = token.contains(['.', 'E', 'e', 'D', 'd'])
                    || token.starts_with('-') && token[1..].contains(['.', 'E', 'e']);
                if is_float {
                    CardValue::Float(
                        token
                            .replace(['D', 'd'], "E")
                            .parse::<f64>()
                            .map_err(|_| bad(format!("bad float {token:?}")))?,
                    )
                } else {
                    CardValue::Int(
                        token
                            .parse::<i64>()
                            .map_err(|_| bad(format!("bad integer {token:?}")))?,
                    )
                }
            }
        }
    };
    Ok((keyword, Some(parsed)))
}

impl FitsImage {
    /// Open and map a FITS file, validating the primary HDU.
    pub fn open(path: impl Into<std::path::PathBuf>) -> Result<FitsImage, FitsError> {
        let path = path.into();
        let file = std::fs::File::open(&path).map_err(|e| FitsError::Io {
            path: path.clone(),
            source: e,
        })?;
        let map = crate::datafile::Buffer::from_file(file).map_err(|e| FitsError::Io {
            path: path.clone(),
            source: e,
        })?;
        let bytes = map.as_slice();
        let bad_magic = || FitsError::NotFits { path: path.clone() };
        if bytes.len() < 80 || !bytes.starts_with(b"SIMPLE  =") {
            return Err(bad_magic());
        }
        let mut header = FitsHeader::default();
        let mut offset = 0usize;
        let mut ended = false;
        while offset + 2880 <= bytes.len() && !ended {
            for i in 0..36 {
                let card = &bytes[offset + i * 80..offset + (i + 1) * 80];
                let (k, v) = parse_card(card, &path, offset + i * 80)?;
                if k == "END" {
                    ended = true;
                    break;
                }
                if k.is_empty() || k == "COMMENT" || k == "HISTORY" {
                    continue;
                }
                header.cards.retain(|(ek, _)| ek != &k);
                header.cards.push((k, v));
            }
            offset += 2880;
        }
        if !ended {
            return Err(bad_magic());
        }
        let unsupported = |what: &str| FitsError::Unsupported {
            path: path.clone(),
            what: what.to_string(),
        };
        if header.get("SIMPLE") != Some(&CardValue::Bool(true)) {
            return Err(unsupported("SIMPLE is not T"));
        }
        if header
            .get("GROUPS")
            .is_some_and(|v| *v == CardValue::Bool(true))
        {
            return Err(unsupported("random-groups records"));
        }
        if header.string_of("XTENSION").is_some() {
            return Err(unsupported("extension HDUs"));
        }
        let naxis = match header.f64_of("NAXIS") {
            Some(v) if (0.0..=20.0).contains(&v) => v as usize,
            _ => return Err(unsupported("missing/absurd NAXIS")),
        };
        let mut naxes = Vec::with_capacity(naxis);
        for n in 1..=naxis {
            naxes.push(header.axis_f64("NAXIS", n).unwrap_or(1.0).max(0.0) as usize);
        }
        let bitpix = header.f64_of("BITPIX").unwrap_or(-32.0) as i32;
        if !matches!(bitpix, 8 | 16 | 32 | 64 | -32 | -64) {
            return Err(unsupported(&format!("BITPIX {bitpix}")));
        }
        let pcount = header.f64_of("PCOUNT").unwrap_or(0.0) as usize;
        let gcount = header.f64_of("GCOUNT").unwrap_or(1.0) as usize;
        let elem = bitpix.unsigned_abs() as usize / 8;
        let data_len: usize =
            naxes.iter().product::<usize>() * elem * gcount.max(1) + pcount * elem * gcount.max(1);
        let data_offset = offset as u64;
        if data_offset + data_len as u64 > bytes.len() as u64 {
            return Err(FitsError::Data {
                path: path.clone(),
                msg: format!(
                    "data region {}..{} exceeds file length {}",
                    data_offset,
                    data_offset + data_len as u64,
                    bytes.len()
                ),
            });
        }
        Ok(FitsImage {
            bscale: header.f64_of("BSCALE").unwrap_or(1.0),
            bzero: header.f64_of("BZERO").unwrap_or(0.0),
            path,
            header,
            naxes,
            bitpix,
            data_offset,
            data_len,
            map,
        })
    }

    /// The image data as f64 (scaled to physical values).  `data_array`
    /// keeps the native dtype the way casacore's `getdata` does.
    pub fn data_f64(&self) -> Result<Vec<f64>, FitsError> {
        let bytes = &self.map.as_slice()[self.data_offset as usize..][..self.data_len];
        let bad = |why: String| FitsError::Data {
            path: self.path.clone(),
            msg: why,
        };
        let n: usize = self.naxes.iter().product();
        let mut out = Vec::with_capacity(n);
        macro_rules! be_int {
            ($width:expr, $ty:ty) => {{
                for chunk in bytes.chunks_exact($width) {
                    let mut b: [u8; $width] = [0; $width];
                    b.copy_from_slice(chunk);
                    let raw = <$ty>::from_be_bytes(b);
                    out.push(raw as f64 * self.bscale + self.bzero);
                }
            }};
        }
        match self.bitpix {
            8 => {
                for &b in bytes {
                    out.push(f64::from(b) * self.bscale + self.bzero);
                }
            }
            16 => be_int!(2, i16),
            32 => be_int!(4, i32),
            64 => be_int!(8, i64),
            -32 => {
                for chunk in bytes.as_chunks::<4>().0 {
                    out.push(f32::from_be_bytes(*chunk) as f64);
                }
            }
            -64 => {
                for chunk in bytes.as_chunks::<8>().0 {
                    out.push(f64::from_be_bytes(*chunk));
                }
            }
            other => return Err(bad(format!("BITPIX {other}"))),
        }
        if out.len() != n {
            return Err(bad(format!("decoded {} pixels, expected {n}", out.len())));
        }
        Ok(out)
    }

    /// The image data keeping its native dtype (float32 FITS -> f32, like
    /// casacore's `getdata`); integer pixels decode as f64 (scaled).
    pub fn data_array(&self) -> Result<ArrayData, FitsError> {
        if self.bitpix == -32 {
            let bytes = &self.map.as_slice()[self.data_offset as usize..][..self.data_len];
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.as_chunks::<4>().0 {
                out.push(f32::from_be_bytes(*chunk));
            }
            return Ok(ArrayData::Float(out));
        }
        if self.bitpix == -64 {
            let bytes = &self.map.as_slice()[self.data_offset as usize..][..self.data_len];
            let mut out = Vec::with_capacity(bytes.len() / 8);
            for chunk in bytes.as_chunks::<8>().0 {
                out.push(f64::from_be_bytes(*chunk));
            }
            return Ok(ArrayData::Double(out));
        }
        Ok(ArrayData::Double(self.data_f64()?))
    }
}
