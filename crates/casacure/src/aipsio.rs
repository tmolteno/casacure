//! Minimal reader for casacore's AipsIO canonical byte format.
//!
//! An AipsIO stream holds nested typed objects. The root object starts with
//! the magic value `0xbebebebe`; every object (root included) is then laid
//! out as `[u32 length][u32 type-length + type bytes][u32 version][payload]`
//! where `length` counts every byte of the object after the magic, including
//! the 4 bytes of the length field itself.
//! All integers are big-endian; strings are a `u32` length followed by raw
//! bytes. See `casacore/casa/IO/AipsIO.cc` (`putstart`/`putend`/`getstart`).

use thiserror::Error;

/// Magic value written before the root object (`AipsIO::magicval_p`).
pub const MAGIC: u32 = 0xbebebebe;

/// Errors from decoding an AipsIO byte stream.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AipsIoError {
    #[error("buffer too short: need {needed} bytes at offset {offset}, have {len}")]
    Truncated {
        needed: usize,
        offset: usize,
        len: usize,
    },
    #[error("bad AipsIO magic at offset {offset}: expected {MAGIC:#010x}, found {found:#010x}")]
    BadMagic { offset: usize, found: u32 },
    #[error("invalid string at offset {offset}: not valid UTF-8")]
    BadString { offset: usize },
    #[error("unexpected object type at offset {offset}: expected {expected:?}, found {found:?}")]
    UnexpectedType {
        offset: usize,
        expected: String,
        found: String,
    },
    #[error("unsupported object version {0}")]
    UnsupportedVersion(u32),
}

/// Header of a typed object in an AipsIO stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectStart {
    /// Payload length in bytes (type name and version included).
    pub length: u32,
    pub type_name: String,
    pub version: u32,
    /// Byte offset of the payload in the underlying buffer.
    pub payload_offset: usize,
}

/// Cursor over an AipsIO byte stream.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], AipsIoError> {
        if self.buf.len() - self.pos < n {
            return Err(AipsIoError::Truncated {
                needed: n,
                offset: self.pos,
                len: self.buf.len(),
            });
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    pub fn read_u32(&mut self) -> Result<u32, AipsIoError> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_i32(&mut self) -> Result<i32, AipsIoError> {
        Ok(self.read_u32()? as i32)
    }

    pub fn read_u64(&mut self) -> Result<u64, AipsIoError> {
        let b = self.take(8)?;
        Ok(u64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn read_i64(&mut self) -> Result<i64, AipsIoError> {
        Ok(self.read_u64()? as i64)
    }

    pub fn read_u8(&mut self) -> Result<u8, AipsIoError> {
        Ok(self.take(1)?[0])
    }

    pub fn read_i16(&mut self) -> Result<i16, AipsIoError> {
        let b = self.take(2)?;
        Ok(i16::from_be_bytes([b[0], b[1]]))
    }

    pub fn read_u16(&mut self) -> Result<u16, AipsIoError> {
        Ok(self.read_i16()? as u16)
    }

    pub fn read_f32(&mut self) -> Result<f32, AipsIoError> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    pub fn read_f64(&mut self) -> Result<f64, AipsIoError> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    /// AipsIO Bool: one bit-packed byte; a scalar occupies a full byte with
    /// the value in bit 0 (`TypeIO::write` via `Conversion::boolToBit`).
    pub fn read_bool(&mut self) -> Result<bool, AipsIoError> {
        Ok(self.read_u8()? & 1 != 0)
    }

    /// An opaque data block: `u32` byte length + raw bytes (`AipsIO::getnew`
    /// / `ByteIO` multi-byte write), e.g. the per-data-manager blobs in
    /// `table.dat`. The returned slice is a self-contained AipsIO stream.
    pub fn read_opaque(&mut self) -> Result<&'a [u8], AipsIoError> {
        let n = self.read_u32()? as usize;
        self.take(n)
    }

    /// AipsIO string: `u32` length + raw bytes (no NUL terminator).
    pub fn read_string(&mut self) -> Result<String, AipsIoError> {
        let offset = self.pos;
        let len = self.read_u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| AipsIoError::BadString { offset })
    }

    /// Read an object header. `root` objects are preceded by the magic value.
    pub fn read_object_start(&mut self, root: bool) -> Result<ObjectStart, AipsIoError> {
        if root {
            let offset = self.pos;
            let found = self.read_u32()?;
            if found != MAGIC {
                return Err(AipsIoError::BadMagic { offset, found });
            }
        }
        let length = self.read_u32()?;
        let type_name = self.read_string()?;
        let version = self.read_u32()?;
        Ok(ObjectStart {
            length,
            type_name,
            version,
            payload_offset: self.pos,
        })
    }

    /// Read a framed object of an expected type, returning its version and
    /// payload. `root` objects are preceded by the magic value.
    pub fn read_object(
        &mut self,
        root: bool,
        expected: &str,
    ) -> Result<(u32, ObjectStart), AipsIoError> {
        let obj = self.read_object_start(root)?;
        if obj.type_name != expected {
            return Err(AipsIoError::UnexpectedType {
                offset: obj.payload_offset,
                expected: expected.to_string(),
                found: obj.type_name,
            });
        }
        Ok((obj.version, obj))
    }

    /// IPosition: framed `"IPosition"` object; v1 = `u32 nelem` + i32 dims,
    /// v2 (huge dims) = i64 dims (`casacore/casa/IO/IPositionIO.cc`).
    pub fn read_iposition(&mut self) -> Result<Vec<i64>, AipsIoError> {
        let (version, _) = self.read_object(false, "IPosition")?;
        let n = self.read_u32()? as usize;
        let mut dims = Vec::with_capacity(n);
        for _ in 0..n {
            dims.push(match version {
                1 => i64::from(self.read_i32()?),
                2 => self.read_i64()?,
                v => return Err(AipsIoError::UnsupportedVersion(v)),
            });
        }
        Ok(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object_bytes(root: bool, type_name: &str, version: u32, payload: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        if root {
            buf.extend_from_slice(&MAGIC.to_be_bytes());
        }
        let length = (4 + 4 + type_name.len() + 4 + payload.len()) as u32;
        buf.extend_from_slice(&length.to_be_bytes());
        buf.extend_from_slice(&(type_name.len() as u32).to_be_bytes());
        buf.extend_from_slice(type_name.as_bytes());
        buf.extend_from_slice(&version.to_be_bytes());
        buf.extend_from_slice(payload);
        buf
    }

    #[test]
    fn reads_root_object() {
        let buf = object_bytes(true, "Table", 2, &[0, 0, 0, 1]);
        let mut r = Reader::new(&buf);
        let obj = r.read_object_start(true).unwrap();
        assert_eq!(obj.type_name, "Table");
        assert_eq!(obj.version, 2);
        assert_eq!(obj.length as usize, buf.len() - 4);
        assert_eq!(obj.payload_offset, 21);
        assert_eq!(r.read_u32().unwrap(), 1);
    }

    #[test]
    fn nested_object_has_no_magic() {
        let buf = object_bytes(false, "TableDesc", 2, &[]);
        let mut r = Reader::new(&buf);
        let obj = r.read_object_start(false).unwrap();
        assert_eq!(obj.type_name, "TableDesc");
        assert_eq!(obj.version, 2);
    }

    #[test]
    fn rejects_bad_magic() {
        let buf = object_bytes(true, "Table", 2, &[]);
        let mut r = Reader::new(&buf[4..]); // drop the magic
        assert!(matches!(
            r.read_object_start(true),
            Err(AipsIoError::BadMagic { offset: 0, .. })
        ));
    }

    #[test]
    fn rejects_truncated_buffer() {
        let buf = object_bytes(true, "Table", 2, &[1, 2, 3, 4]);
        let mut r = Reader::new(&buf[..buf.len() - 2]);
        r.read_object_start(true).unwrap();
        assert!(matches!(r.read_u32(), Err(AipsIoError::Truncated { .. })));
    }

    #[test]
    fn reads_u64_big_endian() {
        let bytes = 0x0102030405060708u64.to_be_bytes();
        let mut r = Reader::new(&bytes);
        assert_eq!(r.read_u64().unwrap(), 0x0102030405060708);
    }
}
