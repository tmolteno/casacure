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
    /// The bytes do not start a FITS primary header.  `why` distinguishes a
    /// file that never had a `SIMPLE` card from one whose header was cut
    /// short before its `END` card — reporting the latter as "no SIMPLE
    /// card" sent readers looking for a card that was already there.
    #[error("{path}: not a FITS file ({why})")]
    NotFits {
        path: std::path::PathBuf,
        why: String,
    },
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
        let bad_magic = |why: &str| FitsError::NotFits {
            path: path.clone(),
            why: why.to_string(),
        };
        if bytes.len() < 80 || !bytes.starts_with(b"SIMPLE  =") {
            return Err(bad_magic("no SIMPLE card"));
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
            return Err(bad_magic("no END card"));
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
        // Contract (D4): a truncated file is rejected here, at open time, so
        // that every later decode is known to fit inside the mapping.  The
        // decode paths still re-check through `Self::data` rather than
        // slicing blindly, because a panic on a short read is not a useful
        // error for a caller such as DDFacet's FITS ingestion (`fits2png.py`,
        // `ClassCasaImage.py`).
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

    /// The pixel-data byte range, re-validated against the mapping.
    ///
    /// `open` already rejects a truncated data region, so this is normally
    /// infallible; it exists so that the decode paths return
    /// [`FitsError::Data`] instead of panicking if a `FitsImage` is ever
    /// built another way.
    fn data(&self) -> Result<&[u8], FitsError> {
        let bytes = self.map.as_slice();
        let start = self.data_offset as usize;
        let end = start
            .checked_add(self.data_len)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| FitsError::Data {
                path: self.path.clone(),
                msg: format!(
                    "data region {}..{} exceeds file length {}",
                    start,
                    start.saturating_add(self.data_len),
                    bytes.len()
                ),
            })?;
        Ok(&bytes[start..end])
    }

    /// The image data as f64 (scaled to physical values).  `data_array`
    /// keeps the native dtype the way casacore's `getdata` does.
    pub fn data_f64(&self) -> Result<Vec<f64>, FitsError> {
        let bytes = self.data()?;
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
            let bytes = self.data()?;
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.as_chunks::<4>().0 {
                out.push(f32::from_be_bytes(*chunk));
            }
            return Ok(ArrayData::Float(out));
        }
        if self.bitpix == -64 {
            let bytes = self.data()?;
            let mut out = Vec::with_capacity(bytes.len() / 8);
            for chunk in bytes.as_chunks::<8>().0 {
                out.push(f64::from_be_bytes(*chunk));
            }
            return Ok(ArrayData::Double(out));
        }
        Ok(ArrayData::Double(self.data_f64()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn par(text: &str) -> Result<(String, Option<CardValue>), FitsError> {
        let mut card = [b' '; 80];
        let bytes = text.as_bytes();
        let n = bytes.len().min(80);
        card[..n].copy_from_slice(&bytes[..n]);
        parse_card(&card, std::path::Path::new("t.fits"), 0)
    }

    fn value(text: &str) -> Option<CardValue> {
        par(text).expect("parses").1
    }

    #[test]
    fn parse_card_reads_logicals_integers_and_floats() {
        assert_eq!(
            value("SIMPLE  =                    T"),
            Some(CardValue::Bool(true))
        );
        assert_eq!(
            value("EXTEND  =                    F"),
            Some(CardValue::Bool(false))
        );
        assert_eq!(
            value("NAXIS   =                    4"),
            Some(CardValue::Int(4))
        );
        assert_eq!(
            value("NAXIS   =                   -3"),
            Some(CardValue::Int(-3))
        );
        assert_eq!(
            value("CRVAL1  =      1.750000000E+00"),
            Some(CardValue::Float(1.75))
        );
    }

    #[test]
    fn parse_card_accepts_fortran_d_exponents() {
        // FITS writers legitimately emit `D` for a double exponent.
        assert_eq!(
            value("BZERO   =      1.000000000D+03"),
            Some(CardValue::Float(1000.0))
        );
    }

    #[test]
    fn parse_card_reads_quoted_strings_and_strips_trailing_spaces() {
        assert_eq!(
            value("BUNIT   = 'Jy/beam'"),
            Some(CardValue::Str("Jy/beam".into()))
        );
        // A slash inside the quotes is part of the value, not a comment.
        assert_eq!(
            value("BUNIT   = 'Jy/beam '           /Brightness unit"),
            Some(CardValue::Str("Jy/beam".into()))
        );
    }

    #[test]
    fn parse_card_unescapes_a_doubled_quote() {
        assert_eq!(
            value("OBJECT  = 'it''s here'"),
            Some(CardValue::Str("it's here".into()))
        );
    }

    #[test]
    fn parse_card_treats_commentary_and_blank_cards_as_valueless() {
        assert_eq!(par("COMMENT any text at all").unwrap().1, None);
        assert_eq!(par("HISTORY produced by x").unwrap().1, None);
        assert_eq!(par("END").unwrap().0, "END");
        assert_eq!(par("        ").unwrap().1, None);
        // A keyword with no `=` is a commentary card, not an error.
        assert_eq!(par("NOEQUALS").unwrap().1, None);
    }

    #[test]
    fn parse_card_reports_a_bad_number_with_its_offset() {
        let err = par("NAXIS   =                  abc").unwrap_err();
        match err {
            FitsError::Card { offset, why, .. } => {
                assert_eq!(offset, 0);
                assert!(why.contains("bad integer"), "{why}");
            }
            other => panic!("expected a card error, got {other:?}"),
        }
        let err = par("CRVAL1  =                  x.y").unwrap_err();
        assert!(err.to_string().contains("bad float"), "{err}");
    }

    /// A minimal primary HDU on disk: `cards` plus `data`, padded to 2880.
    fn write_fits(
        dir: &std::path::Path,
        name: &str,
        cards: &[&str],
        data: &[u8],
    ) -> std::path::PathBuf {
        let mut body: Vec<u8> = Vec::new();
        for c in cards {
            let mut card = [b' '; 80];
            let b = c.as_bytes();
            card[..b.len().min(80)].copy_from_slice(&b[..b.len().min(80)]);
            body.extend_from_slice(&card);
        }
        let mut end = [b' '; 80];
        end[..3].copy_from_slice(b"END");
        body.extend_from_slice(&end);
        while !body.len().is_multiple_of(2880) {
            body.push(b' ');
        }
        body.extend_from_slice(data);
        let path = dir.join(name);
        std::fs::write(&path, &body).unwrap();
        path
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("casacure-fits-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn open_reads_axes_bitpix_and_scaling() {
        let dir = scratch("open");
        let data: Vec<u8> = [1i16, -2, 3].iter().flat_map(|v| v.to_be_bytes()).collect();
        let path = write_fits(
            &dir,
            "a.fits",
            &[
                "SIMPLE  =                    T",
                "BITPIX  =                   16",
                "NAXIS   =                    1",
                "NAXIS1  =                    3",
                "BSCALE  =      2.000000000E+00",
                "BZERO   =      1.000000000E+02",
            ],
            &data,
        );
        let img = FitsImage::open(&path).unwrap();
        assert_eq!(img.naxes, vec![3]);
        assert_eq!(img.bitpix, 16);
        assert_eq!(img.bscale, 2.0);
        assert_eq!(img.bzero, 100.0);
        assert_eq!(img.data_f64().unwrap(), vec![102.0, 96.0, 106.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn data_array_keeps_float32_and_integers_become_f64() {
        let dir = scratch("dtype");
        let floats: Vec<u8> = [1.5f32, -2.5]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let f32_path = write_fits(
            &dir,
            "f.fits",
            &[
                "SIMPLE  =                    T",
                "BITPIX  =                  -32",
                "NAXIS   =                    1",
                "NAXIS1  =                    2",
            ],
            &floats,
        );
        let img = FitsImage::open(&f32_path).unwrap();
        match img.data_array().unwrap() {
            ArrayData::Float(v) => assert_eq!(v, vec![1.5, -2.5]),
            other => panic!("expected f32, got {other:?}"),
        }

        let ints: Vec<u8> = [7i32, -9].iter().flat_map(|v| v.to_be_bytes()).collect();
        let i32_path = write_fits(
            &dir,
            "i.fits",
            &[
                "SIMPLE  =                    T",
                "BITPIX  =                   32",
                "NAXIS   =                    1",
                "NAXIS1  =                    2",
            ],
            &ints,
        );
        match FitsImage::open(&i32_path).unwrap().data_array().unwrap() {
            ArrayData::Double(v) => assert_eq!(v, vec![7.0, -9.0]),
            other => panic!("expected f64, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_rejects_a_truncated_data_region() {
        let dir = scratch("trunc");
        let data = [0u8; 16];
        let path = write_fits(
            &dir,
            "t.fits",
            &[
                "SIMPLE  =                    T",
                "BITPIX  =                  -32",
                "NAXIS   =                    2",
                "NAXIS1  =                    2",
                "NAXIS2  =                    2",
            ],
            &data,
        );
        // Cut six bytes off the pixel data.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 6);
        std::fs::write(&path, &bytes).unwrap();

        let err = FitsImage::open(&path).unwrap_err();
        match &err {
            FitsError::Data { msg, .. } => assert!(msg.contains("exceeds file length"), "{msg}"),
            other => panic!("expected a data error, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_rejects_a_header_without_an_end_card() {
        let dir = scratch("noend");
        let path = dir.join("noend.fits");
        let mut body = vec![b' '; 2880];
        let simple = b"SIMPLE  =                    T";
        body[..simple.len()].copy_from_slice(simple);
        std::fs::write(&path, &body).unwrap();

        let err = FitsImage::open(&path).unwrap_err();
        assert!(err.to_string().contains("no END card"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_rejects_non_fits_bytes_and_bad_bitpix() {
        let dir = scratch("bad");
        let path = dir.join("x.bin");
        std::fs::write(&path, b"not a fits file at all").unwrap();
        assert!(FitsImage::open(&path)
            .unwrap_err()
            .to_string()
            .contains("no SIMPLE card"));

        let path = write_fits(
            &dir,
            "bp.fits",
            &[
                "SIMPLE  =                    T",
                "BITPIX  =                  -16",
                "NAXIS   =                    0",
            ],
            &[],
        );
        assert!(FitsImage::open(&path)
            .unwrap_err()
            .to_string()
            .contains("BITPIX -16"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_accessors_read_axis_cards() {
        let mut hdr = FitsHeader::default();
        for (k, v) in [
            ("NAXIS", CardValue::Int(2)),
            ("NAXIS1", CardValue::Int(7)),
            ("NAXIS2", CardValue::Float(5.0)),
            ("CTYPE1", CardValue::Str("RA---SIN".into())),
            ("BUNIT", CardValue::Str("Jy/beam".into())),
        ] {
            hdr.cards.push((k.to_string(), Some(v)));
        }
        assert_eq!(hdr.f64_of("NAXIS"), Some(2.0));
        assert_eq!(hdr.axis_f64("NAXIS", 1), Some(7.0));
        assert_eq!(hdr.axis_f64("NAXIS", 2), Some(5.0));
        assert_eq!(hdr.axis_f64("NAXIS", 3), None);
        assert_eq!(hdr.axis_str("CTYPE", 1), Some("RA---SIN"));
        assert_eq!(hdr.string_of("BUNIT"), Some("Jy/beam"));
        assert_eq!(hdr.f64_of("MISSING"), None);
    }
}
