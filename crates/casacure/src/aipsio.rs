//! Reader and writer for casacore's AipsIO canonical byte format.
//!
//! An AipsIO stream holds nested typed objects. The root object starts with
//! the magic value `0xbebebebe`; every object (root included) is then laid
//! out as `[u32 length][u32 type-length + type bytes][u32 version][payload]`
//! where `length` counts every byte of the object after the magic, including
//! the 4 bytes of the length field itself (the length is patched in when the
//! object is closed). Multi-byte values are big-endian by default and
//! little-endian for the StandardStMan data files; strings are a `u32`
//! length followed by raw bytes. See `casacore/casa/IO/AipsIO.cc`
//! (`putstart`/`putend`/`getstart`).

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
    /// Multi-byte values with this endianness. `table.dat` is always
    /// big-endian canonical AipsIO; the StandardStMan data files use the
    /// table's data-file endianness (big or little).
    little_endian: bool,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader {
            buf,
            pos: 0,
            little_endian: false,
        }
    }

    /// A reader over a little-endian AipsIO stream (`LECanonicalIO`), used
    /// for the StandardStMan data files on little-endian hosts.
    pub fn new_le(buf: &'a [u8]) -> Self {
        Reader {
            buf,
            pos: 0,
            little_endian: true,
        }
    }

    pub fn little_endian(&self) -> bool {
        self.little_endian
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
        Ok(if self.little_endian {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        })
    }

    pub fn read_i32(&mut self) -> Result<i32, AipsIoError> {
        Ok(self.read_u32()? as i32)
    }

    pub fn read_u64(&mut self) -> Result<u64, AipsIoError> {
        let b = self.take(8)?;
        Ok(if self.little_endian {
            u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        } else {
            u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        })
    }

    pub fn read_i64(&mut self) -> Result<i64, AipsIoError> {
        Ok(self.read_u64()? as i64)
    }

    pub fn read_u8(&mut self) -> Result<u8, AipsIoError> {
        Ok(self.take(1)?[0])
    }

    pub fn read_i16(&mut self) -> Result<i16, AipsIoError> {
        let b = self.take(2)?;
        Ok(if self.little_endian {
            i16::from_le_bytes([b[0], b[1]])
        } else {
            i16::from_be_bytes([b[0], b[1]])
        })
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

/// Serializer for casacore's canonical AipsIO byte format.
///
/// Objects are opened with `put_object_start` and closed with
/// `put_object_end`; the length field is patched in on close, mirroring
/// `AipsIO::putstart`/`putend`. Root objects are additionally preceded by
/// the magic value.
#[derive(Debug)]
pub struct Writer {
    buf: Vec<u8>,
    little_endian: bool,
    /// Positions of open objects' length words, innermost last.
    stack: Vec<usize>,
}

impl Writer {
    pub fn new() -> Writer {
        Writer {
            buf: Vec::new(),
            little_endian: false,
            stack: Vec::new(),
        }
    }

    /// A writer producing a little-endian stream (`LECanonicalIO`), used for
    /// the StandardStMan data files on little-endian hosts.
    pub fn new_le() -> Writer {
        Writer {
            buf: Vec::new(),
            little_endian: true,
            stack: Vec::new(),
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    pub fn put_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn put_u32(&mut self, v: u32) {
        if self.little_endian {
            self.buf.extend_from_slice(&v.to_le_bytes());
        } else {
            self.buf.extend_from_slice(&v.to_be_bytes());
        }
    }

    pub fn put_i32(&mut self, v: i32) {
        self.put_u32(v as u32);
    }

    pub fn put_u64(&mut self, v: u64) {
        if self.little_endian {
            self.buf.extend_from_slice(&v.to_le_bytes());
        } else {
            self.buf.extend_from_slice(&v.to_be_bytes());
        }
    }

    pub fn put_i64(&mut self, v: i64) {
        self.put_u64(v as u64);
    }

    pub fn put_i16(&mut self, v: i16) {
        if self.little_endian {
            self.buf.extend_from_slice(&v.to_le_bytes());
        } else {
            self.buf.extend_from_slice(&v.to_be_bytes());
        }
    }

    pub fn put_u16(&mut self, v: u16) {
        if self.little_endian {
            self.buf.extend_from_slice(&v.to_le_bytes());
        } else {
            self.buf.extend_from_slice(&v.to_be_bytes());
        }
    }

    pub fn put_f32(&mut self, v: f32) {
        self.put_u32(v.to_bits());
    }

    pub fn put_f64(&mut self, v: f64) {
        self.put_u64(v.to_bits());
    }

    /// AipsIO Bool: one byte with the value in bit 0.
    pub fn put_bool(&mut self, v: bool) {
        self.put_u8(v as u8);
    }

    /// AipsIO string: `u32` length + raw bytes (no NUL terminator).
    pub fn put_string(&mut self, s: &str) {
        self.put_u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }

    /// An opaque data block: `u32` byte length + raw bytes
    /// (`AipsIO::put` / `ByteIO` multi-byte write).
    pub fn put_opaque(&mut self, bytes: &[u8]) {
        self.put_u32(bytes.len() as u32);
        self.buf.extend_from_slice(bytes);
    }

    /// Start a nested (unrooted) typed object.
    pub fn put_object_start(&mut self, type_name: &str, version: u32) {
        let start = self.buf.len();
        self.buf.extend_from_slice(&[0; 4]); // length placeholder
        self.put_string(type_name);
        self.put_u32(version);
        self.stack.push(start);
    }

    /// Start the stream's root object: magic, then the object itself.
    pub fn put_root_object_start(&mut self, type_name: &str, version: u32) {
        self.put_u32(MAGIC);
        self.put_object_start(type_name, version);
    }

    /// Close the innermost open object, patching its length word.
    pub fn put_object_end(&mut self) {
        let start = self
            .stack
            .pop()
            .expect("put_object_end without put_object_start");
        let len = (self.buf.len() - start) as u32;
        if self.little_endian {
            self.buf[start..start + 4].copy_from_slice(&len.to_le_bytes());
        } else {
            self.buf[start..start + 4].copy_from_slice(&len.to_be_bytes());
        }
    }
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
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

    #[test]
    fn hexdump() {
        let mut w = Writer::new();
        w.put_root_object_start("Table", 2);
        w.put_u32(1);
        w.put_object_end();
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert_eq!(r.read_object_start(true).unwrap().type_name, "Table");
        assert_eq!(r.read_u32().unwrap(), 1);
    }

    #[test]
    fn writes_nested_objects_with_checked_lengths() {
        let mut w = Writer::new();
        w.put_root_object_start("A", 1);
        w.put_object_start("B", 2);
        w.put_string("inner");
        w.put_object_end();
        w.put_object_end();
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        let root = r.read_object_start(true).unwrap();
        assert_eq!((root.type_name.as_str(), root.version), ("A", 1));
        assert_eq!(root.length as usize, bytes.len() - 4);
        let nested = r.read_object_start(false).unwrap();
        assert_eq!((nested.type_name.as_str(), nested.version), ("B", 2));
        assert_eq!(r.read_string().unwrap(), "inner");
    }

    #[test]
    fn writes_little_endian_streams() {
        let mut w = Writer::new_le();
        w.put_u32(0x01020304);
        w.put_i16(-2);
        w.put_bool(true);
        let bytes = w.into_bytes();
        let mut r = Reader::new_le(&bytes);
        assert_eq!(r.read_u32().unwrap(), 0x01020304);
        assert_eq!(r.read_i16().unwrap(), -2);
        assert!(r.read_bool().unwrap());
        // Same bytes also parse under the big-endian reader as the flipped
        // values, confirming the endianness is actually applied.
        let mut r = Reader::new(&bytes);
        assert_eq!(r.read_u32().unwrap(), 0x04030201);
    }

    #[test]
    fn writer_round_trips_opaque() {
        let payload = b"\x00\x01\x02\x03";
        let mut w = Writer::new();
        w.put_opaque(payload);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes);
        assert_eq!(r.read_opaque().unwrap(), payload);
    }
}
