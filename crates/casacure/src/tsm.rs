//! Reader for the TiledColumnStMan data file (`table.f{seq}` header plus the
//! tile data file `table.f{seq}_TSM{fileSeqNr}`)
//! (`casacore/tables/DataMan/TiledColumnStMan.cc`, `TiledStMan.cc`,
//! `TSMCube.cc`, `TSMFile.cc`).
//!
//! Layout:
//!
//! - The header file (`table.f{seq}`) is always canonical **big-endian**
//!   AipsIO. Its root is `"TiledColumnStMan"` with the fixed cell shape,
//!   then a nested `"TiledStMan"` object: sequence nr, row count, column
//!   data types, hypercolumn name, dimensionality, the tile files
//!   (sequence nr + length), and one hypercube per column group.
//! - Each hypercube carries its shape (`cubeShape`), tile shape
//!   (`tileShape`), the tile-file sequence number, and a byte offset into
//!   that file. The last cube dimension is the (extensible) row axis; the
//!   earlier dimensions are the fixed per-row array shape.
//! - The tile data file (`table.f{seq}_TSM{fileSeqNr}`) is a bucket file in
//!   the table's *data-file* endianness: tile `t` occupies
//!   `fileOffset + t * tileSize * elemSize` bytes. A row's array cell spans
//!   the full first `nrdim-1` tile dimensions (the common fixed-shape case).

use crate::aipsio::{AipsIoError, Reader};
use crate::record::{ArrayData, ArrayValue, DataType, RecordValue};
use crate::tabledesc::ColumnDesc;
use thiserror::Error;

/// Errors from reading a TiledColumnStMan data file.
#[derive(Debug, Error)]
pub enum TsmError {
    #[error(transparent)]
    AipsIo(#[from] AipsIoError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("unexpected object type {found:?}, expected {expected:?}")]
    UnexpectedType { expected: String, found: String },
    #[error("row {row} outside the {nrow} rows of the tiled data")]
    RowOutOfRange { row: u64, nrow: u64 },
    #[error(
        "tiles smaller than the cell are not supported (cube {cube_shape:?}, tile {tile_shape:?})"
    )]
    TileTooSmall {
        cube_shape: Vec<i64>,
        tile_shape: Vec<i64>,
    },
    #[error("tile data file {0} does not exist")]
    MissingTileFile(String),
    #[error("unsupported tiled element type {0:?}")]
    UnsupportedType(DataType),
    #[error(transparent)]
    Record(#[from] crate::record::RecordError),
}

/// A tile file mentioned in the TSM header (`TSMFile`).
#[derive(Debug, Clone, PartialEq)]
pub struct TsmTileFile {
    pub sequence_nr: u32,
    pub length: u64,
}

/// One hypercube (`TSMCube`).
#[derive(Debug, Clone, PartialEq)]
pub struct TsmCube {
    pub extensible: bool,
    pub nrdim: i32,
    /// Hypercube shape; the last dimension is the (extensible) row axis.
    pub cube_shape: Vec<i64>,
    pub tile_shape: Vec<i64>,
    pub file_seq_nr: i32,
    pub file_offset: u64,
}

/// The parsed TiledStMan header.
#[derive(Debug, Clone)]
pub struct TsmHeader {
    pub version: u32,
    pub seq_nr: u32,
    pub nrrow: u64,
    pub data_types: Vec<DataType>,
    pub hypercolumn_name: String,
    pub nrdim: i32,
    pub files: Vec<TsmTileFile>,
    pub cubes: Vec<TsmCube>,
}

/// A parsed TiledColumnStMan storage manager: header plus the tile file.
#[derive(Debug)]
pub struct TsmFile {
    pub header: TsmHeader,
    /// `table.f{seq}_TSM{fileSeqNr}` tile data, in the table's data-file
    /// byte order (memory-mapped so chunked reads only touch their pages).
    tile_data: crate::datafile::Buffer,
    big_endian: bool,
}

impl TsmFile {
    /// Read `<table_dir>/table.f{seq}` (header) plus its first tile data
    /// file, and parse the geometry.
    pub fn open(
        table_dir: impl AsRef<std::path::Path>,
        seq_nr: u32,
        table_big_endian: bool,
    ) -> Result<TsmFile, TsmError> {
        let dir = table_dir.as_ref();
        let data = std::fs::read(dir.join(format!("table.f{seq_nr}")))?;
        let header = parse_header(&data)?;
        let file_seq = header.cubes.first().map(|c| c.file_seq_nr).unwrap_or(-1);
        let path = if file_seq >= 0 {
            dir.join(format!("table.f{seq_nr}_TSM{file_seq}"))
        } else {
            dir.join(format!("table.f{seq_nr}_TSM0"))
        };
        let tile_file = std::fs::File::open(&path)
            .map_err(|_| TsmError::MissingTileFile(path.display().to_string()))?;
        let tile_data = crate::datafile::Buffer::from_file(tile_file)?;
        Ok(TsmFile {
            header,
            tile_data,
            big_endian: table_big_endian,
        })
    }

    /// Drop this file's mapped tile pages (used after a bulk read has copied
    /// the cells out, to keep streaming scans resident at ~the working set).
    pub fn drop_data_pages(&self) {
        self.tile_data.drop_pages();
    }

    /// Read the fixed-shape array cell of `desc` (an array column of the
    /// hypercolumn) at `row`, returning the logical shape and values.
    pub fn read_cell(&self, desc: &ColumnDesc, row: u64) -> Result<RecordValue, TsmError> {
        if row >= self.header.nrrow {
            return Err(TsmError::RowOutOfRange {
                row,
                nrow: self.header.nrrow,
            });
        }
        let cube = self.header.cubes.first().ok_or(TsmError::RowOutOfRange {
            row,
            nrow: self.header.nrrow,
        })?;
        let nrdim = cube.nrdim as usize;
        if nrdim < 1 || cube.cube_shape.len() != nrdim || cube.tile_shape.len() != nrdim {
            return Err(TsmError::TileTooSmall {
                cube_shape: cube.cube_shape.clone(),
                tile_shape: cube.tile_shape.clone(),
            });
        }
        // The per-row cell spans all but the (extensible) row dimension.
        let cell_size: i64 = cube.cube_shape[..nrdim - 1].iter().product::<i64>();
        let row_tiles: i64 = cube.tile_shape[nrdim - 1];
        if row_tiles <= 0 {
            return Err(TsmError::TileTooSmall {
                cube_shape: cube.cube_shape.clone(),
                tile_shape: cube.tile_shape.clone(),
            });
        }
        // Tiles must match the cell in the non-row dimensions.
        for i in 0..nrdim - 1 {
            if cube.tile_shape[i] != cube.cube_shape[i] {
                return Err(TsmError::TileTooSmall {
                    cube_shape: cube.cube_shape.clone(),
                    tile_shape: cube.tile_shape.clone(),
                });
            }
        }
        let tile_nr = row / row_tiles as u64;
        let row_in_tile = (row % row_tiles as u64) as i64;
        let elem_size = tsm_elem_size(desc.data_type)?;
        let tile_size: i64 = cube.tile_shape.iter().product();
        let bucket_size = tile_size as usize * elem_size;
        let cell_elems = cell_size as usize;
        let in_tile_elems = row_in_tile as usize * cell_elems;
        let offset =
            cube.file_offset as usize + tile_nr as usize * bucket_size + in_tile_elems * elem_size;
        let end = offset + cell_elems * elem_size;
        if end > self.tile_data.len() {
            return Err(TsmError::RowOutOfRange {
                row,
                nrow: self.header.nrrow,
            });
        }
        let slice = &self.tile_data[offset..end];
        // Logical shape = reverse of the on-disk (CASA) cell shape.
        let logical: Vec<u32> = cube.cube_shape[..nrdim - 1]
            .iter()
            .rev()
            .map(|&d| d as u32)
            .collect();
        let data = decode_tile_data(slice, desc.data_type, cell_elems, self.big_endian)?;
        Ok(RecordValue::Array(ArrayValue {
            shape: logical,
            data,
        }))
    }
}

/// Bytes of one tiled element in the tile file.
fn tsm_elem_size(dt: DataType) -> Result<usize, TsmError> {
    match dt {
        DataType::Bool | DataType::UChar | DataType::Char => Ok(1),
        DataType::Short | DataType::UShort => Ok(2),
        DataType::Int | DataType::UInt | DataType::Float => Ok(4),
        DataType::Int64 | DataType::Double | DataType::Complex => Ok(8),
        DataType::DComplex => Ok(16),
        other => Err(TsmError::UnsupportedType(other)),
    }
}

/// Decode `nelem` elements of `dt` from the tile slice (data-file endian).
fn decode_tile_data(
    slice: &[u8],
    dt: DataType,
    nelem: usize,
    big_endian: bool,
) -> Result<ArrayData, TsmError> {
    let mut r = if big_endian {
        Reader::new(slice)
    } else {
        Reader::new_le(slice)
    };
    Ok(match dt {
        DataType::Bool => {
            let mut v = Vec::with_capacity(nelem);
            for _ in 0..nelem {
                v.push(r.read_bool()?);
            }
            ArrayData::Bool(v)
        }
        DataType::Char | DataType::UChar => {
            ArrayData::UChar((0..nelem).map(|_| r.read_u8()).collect::<Result<_, _>>()?)
        }
        DataType::Short => {
            ArrayData::Short((0..nelem).map(|_| r.read_i16()).collect::<Result<_, _>>()?)
        }
        DataType::UShort => {
            ArrayData::UShort((0..nelem).map(|_| r.read_u16()).collect::<Result<_, _>>()?)
        }
        DataType::Int => {
            ArrayData::Int((0..nelem).map(|_| r.read_i32()).collect::<Result<_, _>>()?)
        }
        DataType::UInt => {
            ArrayData::UInt((0..nelem).map(|_| r.read_u32()).collect::<Result<_, _>>()?)
        }
        DataType::Int64 => {
            ArrayData::Int64((0..nelem).map(|_| r.read_i64()).collect::<Result<_, _>>()?)
        }
        DataType::Float => {
            ArrayData::Float((0..nelem).map(|_| r.read_f32()).collect::<Result<_, _>>()?)
        }
        DataType::Double => {
            ArrayData::Double((0..nelem).map(|_| r.read_f64()).collect::<Result<_, _>>()?)
        }
        DataType::Complex => {
            let mut v = Vec::with_capacity(nelem);
            for _ in 0..nelem {
                v.push((r.read_f32()?, r.read_f32()?));
            }
            ArrayData::Complex(v)
        }
        DataType::DComplex => {
            let mut v = Vec::with_capacity(nelem);
            for _ in 0..nelem {
                v.push((r.read_f64()?, r.read_f64()?));
            }
            ArrayData::DComplex(v)
        }
        other => return Err(TsmError::UnsupportedType(other)),
    })
}

/// Parse the TSM header file (always canonical big endian).
fn parse_header(data: &[u8]) -> Result<TsmHeader, TsmError> {
    let mut r = Reader::new(data);
    let obj = r.read_object_start(true)?;
    if obj.type_name != "TiledColumnStMan" {
        return Err(TsmError::UnexpectedType {
            expected: "TiledColumnStMan".into(),
            found: obj.type_name,
        });
    }
    // Subclass payload: the fixed cell shape as an IPosition.
    r.read_iposition()?;
    let base = r.read_object_start(false)?;
    if base.type_name != "TiledStMan" {
        return Err(TsmError::UnexpectedType {
            expected: "TiledStMan".into(),
            found: base.type_name,
        });
    }
    let version = base.version;
    let _stored_big = if version >= 2 { r.read_bool()? } else { true };
    let seq_nr = r.read_u32()?;
    let nrrow = if version >= 3 {
        r.read_u64()?
    } else {
        u64::from(r.read_u32()?)
    };
    let ncol = r.read_u32()?;
    let mut data_types = Vec::with_capacity(ncol as usize);
    for _ in 0..ncol {
        data_types.push(DataType::from_i32(r.read_i32()?)?);
    }
    let hypercolumn_name = r.read_string()?;
    let _pers_cache = r.read_u32()?;
    let nrdim = r.read_i32()?;
    let nrfile = if version >= 3 {
        r.read_u64()?
    } else {
        u64::from(r.read_u32()?)
    };
    let mut files = Vec::with_capacity(nrfile as usize);
    for _ in 0..nrfile {
        let exists = r.read_bool()?;
        if exists {
            let _fver = r.read_u32()?;
            let sequence_nr = r.read_u32()?;
            let length = if version >= 3 {
                r.read_u64()?
            } else {
                u64::from(r.read_u32()?)
            };
            files.push(TsmTileFile {
                sequence_nr,
                length,
            });
        }
    }
    let nrcube = if version >= 3 {
        r.read_u64()?
    } else {
        u64::from(r.read_u32()?)
    };
    let mut cubes = Vec::with_capacity(nrcube as usize);
    for _ in 0..nrcube {
        let cver = r.read_u32()?;
        // Hypercolumn 'values' Record: skip by object length.
        let vobj = r.read_object_start(false)?;
        let vpayload = vobj.length as usize - (4 + 4 + vobj.type_name.len() + 4);
        r.skip(vpayload)?;
        let extensible = r.read_bool()?;
        let cnrdim = r.read_i32()?;
        let cube_shape = r.read_iposition()?;
        let tile_shape = r.read_iposition()?;
        let file_seq_nr = r.read_i32()?;
        let file_offset = if cver == 1 {
            u64::from(r.read_u32()?)
        } else {
            r.read_u64()?
        };
        cubes.push(TsmCube {
            extensible,
            nrdim: cnrdim,
            cube_shape,
            tile_shape,
            file_seq_nr,
            file_offset,
        });
    }
    Ok(TsmHeader {
        version,
        seq_nr,
        nrrow,
        data_types,
        hypercolumn_name,
        nrdim,
        files,
        cubes,
    })
}
/// Default tile bucket size casacore uses (~512 KiB tiles).
const DEFAULT_TILE_BYTES: usize = 524288;

/// Serialize a TiledColumnStMan storage manager for a single fixed-shape
/// array column: the header file (canonical big-endian AipsIO) and the tile
/// data file. `cell_shape` is the fixed per-row cell shape in CASA
/// (reversed logical) dim order; `cells[i]` holds the encoded bytes of that
/// row's cell. Returns `(table.f{seq} bytes, table.f{seq}_TSM0 bytes)`.
pub fn write_tsm_file(
    big_endian: bool,
    seq_nr: u32,
    hypercolumn_name: &str,
    data_type: DataType,
    cell_shape: &[i64],
    cells: &[Vec<u8>],
) -> Result<(Vec<u8>, Vec<u8>), TsmError> {
    let elem_size = tsm_elem_size(data_type)?;
    let cell_elems: i64 = cell_shape.iter().product::<i64>();
    let cell_bytes = cell_elems as usize * elem_size;
    if cells.iter().any(|c| c.len() != cell_bytes) {
        return Err(TsmError::UnsupportedType(data_type));
    }
    let nrow = cells.len() as u64;
    let rows_per_tile = (DEFAULT_TILE_BYTES / cell_bytes.max(1)).max(1) as u64;
    let n_tiles = nrow.div_ceil(rows_per_tile).max(1);
    let tile_elems = cell_elems * rows_per_tile as i64;
    let bucket_size = tile_elems as usize * elem_size;

    // Tile data file (data-file endianness; rows fill the tile grid).
    let mut tile_file = vec![0u8; bucket_size * n_tiles as usize];
    for (row, cell) in cells.iter().enumerate() {
        let tile = row as u64 / rows_per_tile;
        let in_tile = row as u64 % rows_per_tile;
        let off = tile as usize * bucket_size + in_tile as usize * cell_bytes;
        tile_file[off..off + cell_bytes].copy_from_slice(cell);
    }

    let mut cube_shape = cell_shape.to_vec();
    cube_shape.push(nrow as i64);
    let mut tile_shape = cell_shape.to_vec();
    tile_shape.push(rows_per_tile as i64);

    // Header: canonical big-endian AipsIO.
    let mut hw = crate::aipsio::Writer::new();
    hw.put_root_object_start("TiledColumnStMan", 1);
    // Subclass payload: an empty IPosition (fixed cell shape).
    hw.put_object_start("IPosition", 1);
    hw.put_u32(0);
    hw.put_object_end();
    if big_endian {
        hw.put_object_start("TiledStMan", 1);
    } else {
        hw.put_object_start("TiledStMan", 2);
        hw.put_bool(false); // little endian
    }
    hw.put_u32(seq_nr); // DM sequence nr (checked on read)
    hw.put_u32(nrow as u32); // nrrow (v1/v2)
    hw.put_u32(1); // ncolumn
    hw.put_i32(casaure_dtype_code(data_type));
    hw.put_string(hypercolumn_name);
    hw.put_u32(0); // pers max cache size
    hw.put_i32(cell_shape.len() as i32 + 1); // nrdim = cell dims + rows
    hw.put_u32(1); // nrfile
    hw.put_bool(true);
    hw.put_u32(1); // TSMFile version
    hw.put_u32(0); // file sequence nr
    hw.put_u32(tile_file.len() as u32); // file length
    hw.put_u32(1); // nrcube
                   // One TSMCube.
    hw.put_u32(1); // cube version
                   // Empty hypercolumn values Record.
    hw.put_object_start("Record", 1);
    hw.put_object_start("RecordDesc", 2);
    hw.put_i32(0); // no fields
    hw.put_object_end();
    hw.put_i32(0); // record type (Fixed)
    hw.put_object_end();
    hw.put_bool(true); // extensible
    hw.put_i32(cube_shape.len() as i32);
    put_iposition(&mut hw, &cube_shape);
    put_iposition(&mut hw, &tile_shape);
    hw.put_i32(0); // file sequence nr
    hw.put_u32(0); // file offset (cube version 1)
    hw.put_object_end(); // TiledStMan
    hw.put_object_end(); // TiledColumnStMan

    Ok((hw.into_bytes(), tile_file))
}

fn put_iposition(w: &mut crate::aipsio::Writer, dims: &[i64]) {
    w.put_object_start("IPosition", 1);
    w.put_u32(dims.len() as u32);
    for d in dims {
        w.put_i32(*d as i32);
    }
    w.put_object_end();
}

fn casaure_dtype_code(dt: DataType) -> i32 {
    match dt {
        DataType::Bool => 0,
        DataType::Char => 1,
        DataType::UChar => 2,
        DataType::Short => 3,
        DataType::UShort => 4,
        DataType::Int => 5,
        DataType::UInt => 6,
        DataType::Float => 7,
        DataType::Double => 8,
        DataType::Complex => 9,
        DataType::DComplex => 10,
        DataType::String => 11,
        DataType::Int64 => 29,
        _ => 12,
    }
}

/// One row's tile-payload bytes for a TiledColumnStMan cell: the same as the
/// SSM array-data encoding, but Bool elements are stored one byte per element
/// (TiledColumnStMan tiles do not bit-pack Bool — see the tile reader).
pub fn tsm_encode_cell(
    big_endian: bool,
    data_type: DataType,
    data: &crate::record::ArrayData,
) -> Result<Vec<u8>, TsmError> {
    use crate::record::ArrayData;
    if data_type == DataType::Bool {
        let ArrayData::Bool(v) = data else {
            return Err(TsmError::UnsupportedType(data_type));
        };
        let mut out = Vec::with_capacity(v.len());
        for b in v {
            out.push(u8::from(*b));
        }
        return Ok(out);
    }
    crate::ssm::encode_array_data(big_endian, data)
        .map_err(|_| TsmError::UnsupportedType(data_type))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tabledesc::{ColumnDesc, ColumnKind};

    /// The fixed descriptor for a `TiledColumnStMan` array column: CASA
    /// (reversed logical) shape `[3, 2]` for a logical 2x3 cell. `read_cell`
    /// only consults `data_type`/`data_manager_type`, but keep the rest
    /// realistic.
    fn array_desc(dt: DataType) -> ColumnDesc {
        ColumnDesc {
            name: "DATA".into(),
            comment: String::new(),
            data_type: dt,
            data_manager_type: "TiledColumnStMan".into(),
            data_manager_group: "TiledData_GROUP".into(),
            options: 0,
            ndim: 2,
            shape: Some(vec![3, 2]),
            max_length: 0,
            keywords: crate::record::TableRecord {
                desc: Default::default(),
                record_type: 0,
                values: Vec::new(),
            },
            kind: ColumnKind::Array,
        }
    }

    /// Every type TiledColumnStMan can store on disk, each with a per-row
    /// value so offset errors surface.
    fn sample_cell(dt: DataType, row: i32) -> ArrayData {
        let r = row as f32;
        let d = row as f64;
        match dt {
            DataType::Bool => ArrayData::Bool(vec![true, false, true, true, false, false]),
            DataType::UChar => ArrayData::UChar(vec![row as u8, 1, 2, 3, 4, 5]),
            DataType::Short => ArrayData::Short(vec![-(row as i16), 1, -2, 3, -4, 5]),
            DataType::UShort => ArrayData::UShort(vec![row as u16, 1, 2, 3, 4, 5]),
            DataType::Int => ArrayData::Int(vec![row, -1, 2, -3, 4, 5]),
            DataType::UInt => ArrayData::UInt(vec![row as u32, 1, 2, 3, 4, 5]),
            DataType::Int64 => ArrayData::Int64(vec![i64::from(row), -1, 2, -3, 4, 5]),
            DataType::Float => ArrayData::Float(vec![r, 1.5, -2.5, 3.5, -4.5, 5.5]),
            DataType::Double => ArrayData::Double(vec![d, 1.5, -2.5, 3.5, -4.5, 5.5]),
            DataType::Complex => ArrayData::Complex(vec![
                (r, 1.0),
                (2.0, -3.0),
                (4.0, 5.0),
                (-6.0, 7.0),
                (8.0, -9.0),
                (10.0, 11.0),
            ]),
            DataType::DComplex => ArrayData::DComplex(vec![
                (d, 1.0),
                (2.0, -3.0),
                (4.0, 5.0),
                (-6.0, 7.0),
                (8.0, -9.0),
                (10.0, 11.0),
            ]),
            other => panic!("{other:?} is not a tiled element type"),
        }
    }

    /// Write `data` (one logical 2x3 cell per row) through the serializers,
    /// parse the header back, and check every row reads back exactly.
    fn round_trip(big_endian: bool, dt: DataType, data: &[ArrayData]) {
        // CASA cell shape (reversed logical): 2 rows x 3 cols.
        let casa_shape: Vec<i64> = vec![3, 2];
        let cells: Vec<Vec<u8>> = data
            .iter()
            .map(|d| tsm_encode_cell(big_endian, dt, d).unwrap())
            .collect();
        let (header_bytes, tile_data) =
            write_tsm_file(big_endian, 0, "TiledData_GROUP", dt, &casa_shape, &cells).unwrap();
        let header = parse_header(&header_bytes).unwrap();

        // Header geometry: fixed cell dims + the extensible row axis.
        assert_eq!(header.nrrow, data.len() as u64);
        assert_eq!(header.hypercolumn_name, "TiledData_GROUP");
        assert_eq!(header.data_types, vec![dt]);
        assert_eq!(header.nrdim, 3);
        assert_eq!(header.files.len(), 1);
        assert_eq!(header.files[0].sequence_nr, 0);
        let cube = &header.cubes[0];
        assert_eq!(cube.nrdim, 3);
        assert_eq!(cube.cube_shape, vec![3, 2, data.len() as i64]);
        assert_eq!(cube.tile_shape[..2], [3, 2]);
        assert_eq!(cube.file_seq_nr, 0);
        assert_eq!(cube.file_offset, 0);
        // If it were version >= 2 the stored big-endian flag must agree.
        if big_endian {
            assert_eq!(header.version, 1, "big-endian TSM header has no flag");
        } else {
            assert_eq!(
                header.version, 2,
                "little-endian TSM header carries the flag"
            );
        }

        let tsm = TsmFile {
            header,
            tile_data: crate::datafile::Buffer::from(tile_data),
            big_endian,
        };
        let desc = array_desc(dt);
        for (row, d) in data.iter().enumerate() {
            let want = RecordValue::Array(ArrayValue {
                shape: vec![2, 3],
                data: d.clone(),
            });
            assert_eq!(
                tsm.read_cell(&desc, row as u64).unwrap(),
                want,
                "row {row}, dtype {dt:?}, endian {big_endian}"
            );
        }
    }

    #[test]
    fn write_read_cell_round_trip_all_types_both_endians() {
        for &big in &[false, true] {
            for dt in [
                DataType::Bool,
                DataType::UChar,
                DataType::Short,
                DataType::UShort,
                DataType::Int,
                DataType::UInt,
                DataType::Int64,
                DataType::Float,
                DataType::Double,
                DataType::Complex,
                DataType::DComplex,
            ] {
                let data: Vec<ArrayData> = (0..3).map(|row| sample_cell(dt, row)).collect();
                round_trip(big, dt, &data);
            }
        }
    }

    /// Bool tiles must store one byte per element (not SSM's bit-packing).
    #[test]
    fn bool_tiles_store_one_byte_per_element() {
        let data = ArrayData::Bool(vec![true, false, true, true, false, false]);
        let cells = vec![tsm_encode_cell(false, DataType::Bool, &data).unwrap()];
        let (header, tile_data) =
            write_tsm_file(false, 0, "g", DataType::Bool, &[3, 2], &cells).unwrap();
        // First cell starts at tile offset 0: the six raw bool bytes.
        assert_eq!(&tile_data[..6], &[1, 0, 1, 1, 0, 0]);
        // And it still decodes to the original cell.
        let header = parse_header(&header).unwrap();
        let tsm = TsmFile {
            header,
            tile_data: crate::datafile::Buffer::from(tile_data),
            big_endian: false,
        };
        let want = RecordValue::Array(ArrayValue {
            shape: vec![2, 3],
            data,
        });
        assert_eq!(tsm.read_cell(&array_desc(DataType::Bool), 0).unwrap(), want);
    }

    /// A cell large enough to force more than one tile per row-spill boundary:
    /// 1024 dcomplex = 16 KiB/cell -> 32 rows per 512 KiB tile; 65 rows
    /// straddle three tiles. Exercises `tile_nr * bucket_size` offsets.
    #[test]
    fn read_spans_multiple_tiles() {
        let dt = DataType::DComplex;
        let data: Vec<ArrayData> = (0..65)
            .map(|row| ArrayData::DComplex((0..1024).map(|k| (row as f64, k as f64)).collect()))
            .collect();
        let cells: Vec<Vec<u8>> = data
            .iter()
            .map(|d| tsm_encode_cell(false, dt, d).unwrap())
            .collect();
        // 1-D cell of 1024 dcomplex (CASA shape).
        let (header, tile_data) = write_tsm_file(false, 0, "g", dt, &[1024], &cells).unwrap();
        let header = parse_header(&header).unwrap();
        let rows_per_tile = header.cubes[0].tile_shape[1];
        assert_eq!(rows_per_tile, 32);
        assert!(rows_per_tile * 2 < 65, "test must span tiles");
        assert_eq!(header.cubes[0].cube_shape, vec![1024, 65]);

        let tsm = TsmFile {
            header,
            tile_data: crate::datafile::Buffer::from(tile_data),
            big_endian: false,
        };
        let desc = array_desc(dt);
        for (row, d) in data.iter().enumerate() {
            let want = RecordValue::Array(ArrayValue {
                shape: vec![1024],
                data: d.clone(),
            });
            assert_eq!(tsm.read_cell(&desc, row as u64).unwrap(), want, "row {row}");
        }
    }

    #[test]
    fn rejects_row_out_of_range() {
        let data = [sample_cell(DataType::Int, 0)];
        let cells = vec![tsm_encode_cell(false, DataType::Int, &data[0]).unwrap()];
        let (header, tile_data) =
            write_tsm_file(false, 0, "g", DataType::Int, &[3, 2], &cells).unwrap();
        let header = parse_header(&header).unwrap();
        let tsm = TsmFile {
            header,
            tile_data: crate::datafile::Buffer::from(tile_data),
            big_endian: false,
        };
        assert!(matches!(
            tsm.read_cell(&array_desc(DataType::Int), 1),
            Err(TsmError::RowOutOfRange { row: 1, nrow: 1 })
        ));
    }

    #[test]
    fn rejects_unsupported_tiled_element_type() {
        assert!(matches!(
            tsm_elem_size(DataType::String),
            Err(TsmError::UnsupportedType(DataType::String))
        ));
    }

    #[test]
    fn write_rejects_inconsistent_cell_sizes() {
        // One 1-byte and one 2-byte (u16) cell: 4-element Int cells are 16 B.
        let cells = vec![vec![0u8; 8], vec![0u8; 9]];
        assert!(matches!(
            write_tsm_file(false, 0, "g", DataType::Int, &[3, 2], &cells),
            Err(TsmError::UnsupportedType(DataType::Int))
        ));
    }

    #[test]
    fn header_rejects_wrong_root_type() {
        let cells = vec![vec![0u8; 4]];
        let (header, _) = write_tsm_file(false, 0, "g", DataType::Int, &[1], &cells).unwrap();
        // Corrupt the root type name "TiledColumnStMan" in place.
        let mut bad = header;
        bad[12] = b'X';
        assert!(matches!(
            parse_header(&bad),
            Err(TsmError::UnexpectedType { .. })
        ));
    }

    #[test]
    fn rejects_tiles_smaller_than_cell() {
        // cube_shape[0] != tile_shape[0]: non-data dims must match the tile.
        let header = TsmHeader {
            version: 2,
            seq_nr: 0,
            nrrow: 7,
            data_types: vec![DataType::Int],
            hypercolumn_name: "g".into(),
            nrdim: 3,
            files: Vec::new(),
            cubes: vec![TsmCube {
                extensible: true,
                nrdim: 3,
                cube_shape: vec![2, 3, 7],
                tile_shape: vec![1, 3, 4],
                file_seq_nr: 0,
                file_offset: 0,
            }],
        };
        let tsm = TsmFile {
            header,
            tile_data: crate::datafile::Buffer::from(vec![0u8; 1024]),
            big_endian: false,
        };
        assert!(matches!(
            tsm.read_cell(&array_desc(DataType::Int), 0),
            Err(TsmError::TileTooSmall { .. })
        ));
    }
}
