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

    pub fn read_u64(&mut self) -> Result<u64, AipsIoError> {
        let b = self.take(8)?;
        Ok(u64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
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
