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
//!   `fileOffset + t * bucketSize` bytes, `bucketSize` being
//!   `tileSize * elemSize` (Bool: `ceil(tileSize / 8)`, bit-packed).
//!   Tiles are numbered first-axis-fastest over the tile grid
//!   (`ceil(cubeShape / tileShape)` tiles per axis, `TSMCube::setupNrTiles`),
//!   every tile a full bucket even at the cube edge; within a tile the
//!   elements are first-axis-fastest in tile coordinates.
//! - When the tile covers the whole cell (tile == cell in every non-row
//!   dimension) a row's cell is one contiguous run inside one tile
//!   ([`CellLocation`], the fast path).  Otherwise — dask-ms tiles the
//!   channel axis at <= 64 channels, so this is the common MS case — a cell
//!   is split over the tiles of its row layer ([`TiledCell`]): in each tile
//!   the row's part starts at element `row_in_tile * prod(tile cell dims)`,
//!   in tile-cell coordinates, of which only the in-cube part is valid.
//! - Cubes created extensible (TiledShapeStMan, one per cell shape) each
//!   get their own tile file (`TiledStMan::makeHypercube`), so the tile
//!   data is kept per file sequence number.

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
    #[error(transparent)]
    FileIo(#[from] crate::datafile::FileIoError),
    #[error("unexpected object type {found:?}, expected {expected:?}")]
    UnexpectedType { expected: String, found: String },
    #[error("row {row} outside the {nrow} rows of the tiled data")]
    RowOutOfRange { row: u64, nrow: u64 },
    /// The cell spans several tiles, so it has no single contiguous span:
    /// the single-span accessors ([`TsmFile::cell_span`],
    /// [`TsmFile::cell_location`]) return this; [`TsmFile::cell_place`]
    /// describes such a cell as per-tile segments instead.
    #[error(
        "the cell spans several tiles (cube {cube_shape:?}, tile {tile_shape:?}); \
         use TsmFile::cell_place"
    )]
    TileTooSmall {
        cube_shape: Vec<i64>,
        tile_shape: Vec<i64>,
    },
    #[error("invalid hypercube geometry (cube {cube_shape:?}, tile {tile_shape:?})")]
    BadGeometry {
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
    /// The storage-manager root object the header was written with
    /// (`TiledColumnStMan` or `TiledShapeStMan`); the same `TiledStMan`
    /// payload follows either way.
    pub root_type: String,
    /// The subclass IPosition payload: the fixed cell shape for
    /// `TiledColumnStMan`, the default tile shape for `TiledShapeStMan`.
    pub subclass_shape: Vec<i64>,
    pub version: u32,
    pub seq_nr: u32,
    pub nrrow: u64,
    pub data_types: Vec<DataType>,
    pub hypercolumn_name: String,
    pub nrdim: i32,
    pub files: Vec<TsmTileFile>,
    pub cubes: Vec<TsmCube>,
    /// TiledShapeStMan's row map: for every stored cell (in write order),
    /// the row it belongs to, the cube holding it, and the cell's linear
    /// position within that cube. Empty for TiledColumnStMan, whose single
    /// cube maps rows arithmetically.
    pub row_map: Vec<u32>,
    pub cube_map: Vec<u32>,
    pub pos_map: Vec<u32>,
}

/// A parsed TiledColumnStMan storage manager: header plus the tile file(s).
#[derive(Debug)]
pub struct TsmFile {
    pub header: TsmHeader,
    /// `table.f{seq}_TSM{fileSeqNr}` tile data per file sequence number, in
    /// the table's data-file byte order (memory-mapped so chunked reads only
    /// touch their pages).  One entry for TiledColumnStMan; TiledShapeStMan
    /// keeps one file per extensible cube.
    tile_files: Vec<(i32, crate::datafile::Buffer)>,
    /// Per cube (same index as `header.cubes`): which `tile_files` entry
    /// holds its data and, for a cube whose tiles are smaller than the cell,
    /// the precomputed per-tile gather plan.
    geoms: Vec<CubeGeom>,
    big_endian: bool,
}

/// A stored cell's position in the tile file (see
/// [`TsmFile::cell_location`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellLocation {
    /// Offset of the first byte covering the cell.
    pub byte_off: usize,
    /// Bytes covering the cell (for Bool, `ceil((skip + nelem) / 8)`).
    pub nbytes: usize,
    /// Bit offset of the cell within its first byte (Bool only; else 0).
    pub skip: usize,
    /// Elements in the cell.
    pub nelem: usize,
    /// The tile file (`table.f{seq}_TSM{file_seq}`) holding the bytes.
    pub file_seq: i32,
}

/// A stored cell's bytes in the tile file (see [`TsmFile::cell_span`]).
#[derive(Debug, Clone, Copy)]
pub struct CellSpan<'a> {
    /// The bytes covering the cell (whole bytes; a Bool cell's bits start
    /// `skip` bits into the first byte).
    pub bytes: &'a [u8],
    /// Bit offset of the cell within `bytes[0]` (Bool cells only; else 0).
    pub skip: usize,
    /// Elements in the cell.
    pub nelem: usize,
}

/// One contiguous run of a cell inside one tile of its cube's gather plan:
/// `len` elements at element `src` of the row's block in tile `tile` (of
/// the row layer) go to element `dst` of the cell (CASA order, first axis
/// fastest).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TileRun {
    tile: usize,
    src: usize,
    dst: usize,
    len: usize,
}

/// A cube's precomputed geometry.
#[derive(Debug, Clone, Default)]
struct CubeGeom {
    /// Index into `TsmFile::tile_files` (None: no data file).
    file: Option<usize>,
    /// Empty for a cube whose tile covers the cell (the contiguous fast
    /// path); otherwise the runs of one row's cell, ordered by tile then by
    /// position within the tile (so file offsets increase).
    runs: Vec<TileRun>,
    /// Tiles in one row layer (`prod(ceil(cube / tile))` over cell dims).
    layer_tiles: usize,
    /// Elements of one row's block within a tile (`prod(tile cell dims)`).
    tile_cell_elems: usize,
    /// Whether the cell spans several tiles (or padded ones).
    tiled: bool,
}

impl CubeGeom {
    /// `full_cell`: TiledCellStMan semantics — every row's cell IS a whole
    /// cube (cube index == row), so the gather covers every cube axis
    /// instead of "all but the row axis" (TiledColumnStMan/TiledShapeStMan).
    fn new(cube: &TsmCube, file: Option<usize>, full_cell: bool) -> CubeGeom {
        let nrdim = cube.nrdim as usize;
        let mut g = CubeGeom {
            file,
            ..CubeGeom::default()
        };
        if nrdim < 1 || cube.cube_shape.len() != nrdim || cube.tile_shape.len() != nrdim {
            return g;
        }
        let cut = if full_cell { nrdim } else { nrdim - 1 };
        let cell = &cube.cube_shape[..cut];
        let tile = &cube.tile_shape[..cut];
        if cell == tile || tile.iter().any(|&t| t <= 0) || cell.iter().any(|&c| c < 0) {
            return g;
        }
        g.tiled = true;
        g.runs = tile_runs(cell, tile);
        g.layer_tiles = cell
            .iter()
            .zip(tile)
            .map(|(&c, &t)| (c as usize).div_ceil(t as usize))
            .product();
        g.tile_cell_elems = tile.iter().product::<i64>() as usize;
        g
    }
}

/// The gather plan of one cell of CASA shape `cell` stored in tiles of
/// `tile` (both without the row axis): for every tile of the row layer
/// (first axis fastest), the lines along axis 0 of its in-cube part, merged
/// where they are contiguous both in the tile and in the cell.
fn tile_runs(cell: &[i64], tile: &[i64]) -> Vec<TileRun> {
    let n = cell.len();
    let cell: Vec<usize> = cell.iter().map(|&c| c as usize).collect();
    let tile: Vec<usize> = tile.iter().map(|&t| t as usize).collect();
    let per_dim: Vec<usize> = (0..n).map(|d| cell[d].div_ceil(tile[d])).collect();
    let layer: usize = per_dim.iter().product();
    let mut cstride = vec![1usize; n];
    let mut tstride = vec![1usize; n];
    for d in 1..n {
        cstride[d] = cstride[d - 1] * cell[d - 1];
        tstride[d] = tstride[d - 1] * tile[d - 1];
    }
    let mut runs: Vec<TileRun> = Vec::new();
    let mut tc = vec![0usize; n]; // tile coordinate in the grid
    let mut valid = vec![0usize; n];
    let mut p = vec![0usize; n]; // position inside the tile
    for k in 0..layer {
        let mut rem = k;
        for d in 0..n {
            tc[d] = rem % per_dim[d];
            rem /= per_dim[d];
            valid[d] = tile[d].min(cell[d] - tc[d] * tile[d]);
        }
        p.iter_mut().for_each(|x| *x = 0);
        loop {
            let src: usize = (1..n).map(|d| p[d] * tstride[d]).sum();
            let dst: usize = (0..n).map(|d| (tc[d] * tile[d] + p[d]) * cstride[d]).sum();
            let len = valid[0];
            match runs.last_mut() {
                Some(r) if r.tile == k && r.src + r.len == src && r.dst + r.len == dst => {
                    r.len += len
                }
                _ => runs.push(TileRun {
                    tile: k,
                    src,
                    dst,
                    len,
                }),
            }
            // Odometer over dims 1.. of the tile's valid part.
            let mut d = 1;
            while d < n {
                p[d] += 1;
                if p[d] < valid[d] {
                    break;
                }
                p[d] = 0;
                d += 1;
            }
            if d >= n {
                break;
            }
        }
    }
    runs
}

/// One piece of a multi-tile cell in the tile file (see [`TiledCell`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellSegment {
    /// Offset of the first byte covering the piece.
    pub byte_off: usize,
    /// Bytes covering the piece (Bool: `ceil((skip + len) / 8)`).
    pub nbytes: usize,
    /// Bit offset of the piece within its first byte (Bool only; else 0).
    pub skip: usize,
    /// First element of the cell (CASA order) the piece holds.
    pub dst: usize,
    /// Elements in the piece.
    pub len: usize,
}

/// A stored cell that spans several tiles, borrowed from the mapped tile
/// file: its per-tile [`CellSegment`]s plus gather helpers that assemble
/// the cell in cell order (CASA, first axis fastest).
#[derive(Debug, Clone, Copy)]
pub struct TiledCell<'a> {
    /// The tile file holding the cell.
    pub file_seq: i32,
    /// Elements in the cell.
    pub nelem: usize,
    is_bool: bool,
    /// Bytes per element (1 for Bool, which is bit-packed).
    esize: usize,
    /// Byte offset of the first tile of the row's layer.
    layer_off: usize,
    /// Bytes per tile (bucket).
    bucket: usize,
    /// Element offset of the row's block within each tile.
    row_elem: usize,
    runs: &'a [TileRun],
    data: &'a [u8],
}

impl<'a> TiledCell<'a> {
    fn segment(&self, r: &TileRun) -> CellSegment {
        let tile_off = self.layer_off + r.tile * self.bucket;
        let elem = self.row_elem + r.src;
        if self.is_bool {
            let skip = elem % 8;
            CellSegment {
                byte_off: tile_off + elem / 8,
                nbytes: (skip + r.len).div_ceil(8),
                skip,
                dst: r.dst,
                len: r.len,
            }
        } else {
            CellSegment {
                byte_off: tile_off + elem * self.esize,
                nbytes: r.len * self.esize,
                skip: 0,
                dst: r.dst,
                len: r.len,
            }
        }
    }

    /// Identifies the row layer (and cube) the cell lives in: cells with the
    /// same key have their `j`-th segments in the same tile.
    pub fn layer_key(&self) -> (i32, usize, usize) {
        (self.file_seq, self.layer_off, self.runs.len())
    }

    /// The `j`-th piece (`j < segments().len()`).
    pub fn nth_segment(&self, j: usize) -> CellSegment {
        self.segment(&self.runs[j])
    }

    /// The cell's pieces, in increasing file order.
    pub fn segments(&self) -> impl ExactSizeIterator<Item = CellSegment> + '_ {
        self.runs.iter().map(move |r| self.segment(r))
    }

    /// Byte offset one past the last byte the cell touches.
    fn end(&self) -> usize {
        self.runs.last().map_or(0, |r| {
            let s = self.segment(r);
            s.byte_off + s.nbytes
        })
    }

    /// The cell's element bytes in cell order and in the data file's byte
    /// order (non-Bool types): what a contiguous cell's span would hold.
    pub fn gather_bytes(&self, out: &mut Vec<u8>) {
        out.clear();
        out.resize(self.nelem * self.esize, 0);
        for s in self.segments() {
            let d = s.dst * self.esize;
            out[d..d + s.nbytes].copy_from_slice(&self.data[s.byte_off..s.byte_off + s.nbytes]);
        }
    }

    /// Decode a Bool cell into `out` (`nelem` bools, cell order).
    pub fn gather_bools(&self, out: &mut [bool]) {
        for s in self.segments() {
            decode_bits_into(
                &self.data[s.byte_off..s.byte_off + s.nbytes],
                s.skip,
                &mut out[s.dst..s.dst + s.len],
            );
        }
    }

    /// A Bool cell's bits packed LSB-first from bit 0 of `out` (cell order),
    /// the form a contiguous cell's span holds with `skip == 0`.
    pub fn gather_bits(&self, out: &mut Vec<u8>) {
        out.clear();
        out.resize(self.nelem.div_ceil(8), 0);
        for s in self.segments() {
            or_bits_from(out, s.dst, &self.data[s.byte_off..], s.skip, s.len);
        }
    }
}

/// Where a stored cell lives (see [`TsmFile::cell_place`]).
#[derive(Debug, Clone, Copy)]
pub enum CellPlace<'a> {
    /// The cell is one contiguous run inside one tile (the tile covers the
    /// cell): the fast path.
    Contiguous(CellLocation),
    /// The cell spans several tiles (tiles smaller than the cell).
    Tiled(TiledCell<'a>),
}

impl TsmFile {
    /// Read `<table_dir>/table.f{seq}` (header) plus its tile data files,
    /// and parse the geometry.
    pub fn open(
        table_dir: impl AsRef<std::path::Path>,
        seq_nr: u32,
        table_big_endian: bool,
    ) -> Result<TsmFile, TsmError> {
        let dir = table_dir.as_ref();
        let header_path = dir.join(format!("table.f{seq_nr}"));
        let data = std::fs::read(&header_path)
            .map_err(|e| TsmError::FileIo(crate::datafile::FileIoError::new(&header_path, e)))?;
        let header = parse_header(&data)?;
        // Each cube names its tile file (a shape-stman placeholder cube for
        // not-yet-set cells carries -1; every extensible cube has a file of
        // its own). A column with no cube holding data (every cell unset —
        // casacore skips writing the tile file entirely) opens with no
        // buffer; every read resolves to the column default.
        let mut tile_files: Vec<(i32, crate::datafile::Buffer)> = Vec::new();
        for cube in &header.cubes {
            let f = cube.file_seq_nr;
            if f < 0 || tile_files.iter().any(|(s, _)| *s == f) {
                continue;
            }
            let path = dir.join(format!("table.f{seq_nr}_TSM{f}"));
            let has_rows = header
                .cubes
                .iter()
                .any(|c| c.file_seq_nr == f && c.cube_shape.last().is_some_and(|&r| r > 0));
            let tile_file = match std::fs::File::open(&path) {
                Ok(file) => file,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound && !has_rows => continue,
                Err(e) => {
                    return Err(TsmError::FileIo(crate::datafile::FileIoError::new(
                        &path, e,
                    )))
                }
            };
            let buf = crate::datafile::Buffer::from_file(tile_file)
                .map_err(|e| TsmError::FileIo(crate::datafile::FileIoError::new(&path, e)))?;
            tile_files.push((f, buf));
        }
        Ok(TsmFile::assemble(header, tile_files, table_big_endian))
    }

    fn assemble(
        header: TsmHeader,
        tile_files: Vec<(i32, crate::datafile::Buffer)>,
        big_endian: bool,
    ) -> TsmFile {
        let full_cell = header.root_type == "TiledCellStMan";
        let geoms = header
            .cubes
            .iter()
            .map(|c| {
                let file = tile_files.iter().position(|(s, _)| *s == c.file_seq_nr);
                CubeGeom::new(c, file, full_cell)
            })
            .collect();
        TsmFile {
            header,
            tile_files,
            geoms,
            big_endian,
        }
    }

    /// Assemble a `TsmFile` from an already-parsed header (tests): the
    /// same tile bytes back every file the cubes name.
    #[cfg(test)]
    fn from_header(
        header: TsmHeader,
        tile_data: impl Into<crate::datafile::Buffer>,
        big_endian: bool,
    ) -> TsmFile {
        let buf: crate::datafile::Buffer = tile_data.into();
        let mut seqs: Vec<i32> = header
            .cubes
            .iter()
            .map(|c| c.file_seq_nr)
            .filter(|&f| f >= 0)
            .collect();
        seqs.dedup();
        if seqs.is_empty() {
            seqs.push(0);
        }
        let files = seqs
            .iter()
            .map(|&s| (s, crate::datafile::Buffer::from(buf.as_slice().to_vec())))
            .collect();
        TsmFile::assemble(header, files, big_endian)
    }

    /// Assemble a `TsmFile` with one tile buffer per file sequence number
    /// (tests of multi-file TiledShapeStMan columns).
    #[cfg(test)]
    fn from_header_files(
        header: TsmHeader,
        files: Vec<(i32, Vec<u8>)>,
        big_endian: bool,
    ) -> TsmFile {
        let files = files
            .into_iter()
            .map(|(s, v)| (s, crate::datafile::Buffer::from(v)))
            .collect();
        TsmFile::assemble(header, files, big_endian)
    }

    /// Drop this file's mapped tile pages (used after a bulk read has copied
    /// the cells out, to keep streaming scans resident at ~the working set).
    pub fn drop_data_pages(&self) {
        for (_, b) in &self.tile_files {
            b.drop_pages();
        }
    }

    /// Read the array cell of `desc` (an array column of the hypercolumn)
    /// at `row`, returning the logical shape and values.
    pub fn read_cell(&self, desc: &ColumnDesc, row: u64) -> Result<RecordValue, TsmError> {
        let Some((cube, place)) = self.cell_place(desc, row)? else {
            return self.read_default_cell(desc);
        };
        let nrdim = cube.nrdim as usize;
        // Logical shape = reverse of the on-disk (CASA) cell shape: all but
        // the row axis, or the whole cube for TiledCellStMan.
        let cut = if self.header.root_type == "TiledCellStMan" {
            nrdim
        } else {
            nrdim - 1
        };
        let logical: Vec<u32> = cube.cube_shape[..cut]
            .iter()
            .rev()
            .map(|&d| d as u32)
            .collect();
        let data = match place {
            CellPlace::Contiguous(loc) => {
                let bytes = self.location_bytes(&loc);
                if desc.data_type == DataType::Bool {
                    decode_bits(bytes, loc.skip, loc.nelem)?
                } else {
                    decode_tile_data(bytes, desc.data_type, loc.nelem, self.big_endian)?
                }
            }
            CellPlace::Tiled(tc) => {
                if desc.data_type == DataType::Bool {
                    let mut v = vec![false; tc.nelem];
                    tc.gather_bools(&mut v);
                    ArrayData::Bool(v)
                } else {
                    let mut buf = Vec::new();
                    tc.gather_bytes(&mut buf);
                    decode_tile_data(&buf, desc.data_type, tc.nelem, self.big_endian)?
                }
            }
        };
        Ok(RecordValue::Array(ArrayValue {
            shape: logical,
            data,
        }))
    }

    /// The bytes a [`CellLocation`] from this file covers.
    pub fn location_bytes(&self, loc: &CellLocation) -> &[u8] {
        &self.file_data(loc.file_seq)[loc.byte_off..loc.byte_off + loc.nbytes]
    }

    fn file_data(&self, file_seq: i32) -> &[u8] {
        self.tile_files
            .iter()
            .find(|(s, _)| *s == file_seq)
            .map_or(&[][..], |(_, b)| b.as_slice())
    }

    /// Where the stored cell of `desc` at `row` lives in the tile file,
    /// borrowed from the mapped data (no decode, no allocation), with the
    /// cube holding it. `None` is an unset cell (it reads as the column
    /// default). Bool cells are bit-packed: the cell starts `skip` bits into
    /// `bytes`; every other type is byte-aligned (`skip == 0`).  A cell that
    /// spans several tiles has no single span: this returns
    /// [`TsmError::TileTooSmall`] for it (use [`TsmFile::cell_place`]).
    pub fn cell_span(
        &self,
        desc: &ColumnDesc,
        row: u64,
    ) -> Result<Option<(&TsmCube, CellSpan<'_>)>, TsmError> {
        Ok(self.cell_location(desc, row)?.map(|(cube, loc)| {
            (
                cube,
                CellSpan {
                    bytes: self.location_bytes(&loc),
                    skip: loc.skip,
                    nelem: loc.nelem,
                },
            )
        }))
    }

    /// The tile-file sequence number (`table.f{seq}_TSM{n}`) holding this
    /// column's data, if any cube stores data (the first such file; a
    /// multi-cube TiledShapeStMan column has one file per cube, see
    /// [`CellLocation::file_seq`]).
    pub fn tile_file_seq(&self) -> Option<i32> {
        self.header
            .cubes
            .iter()
            .map(|c| c.file_seq_nr)
            .find(|&f| f >= 0)
    }

    /// Where the stored cell of `desc` at `row` sits in the tile file (see
    /// [`TsmFile::cell_span`]) as byte offsets, so a writer can patch the
    /// cell in place at exactly the position the reader reads it from.
    /// A cell spanning several tiles returns [`TsmError::TileTooSmall`]
    /// (use [`TsmFile::cell_place`]).
    pub fn cell_location(
        &self,
        desc: &ColumnDesc,
        row: u64,
    ) -> Result<Option<(&TsmCube, CellLocation)>, TsmError> {
        match self.cell_place(desc, row)? {
            None => Ok(None),
            Some((cube, CellPlace::Contiguous(loc))) => Ok(Some((cube, loc))),
            Some((cube, CellPlace::Tiled(_))) => Err(TsmError::TileTooSmall {
                cube_shape: cube.cube_shape.clone(),
                tile_shape: cube.tile_shape.clone(),
            }),
        }
    }

    /// The cube index holding `row` and the row's index within that cube,
    /// or `None` for an unset cell.
    fn locate(&self, row: u64) -> Result<Option<(usize, u64)>, TsmError> {
        if row >= self.header.nrrow {
            return Err(TsmError::RowOutOfRange {
                row,
                nrow: self.header.nrrow,
            });
        }
        // TiledCellStMan: every row's cell is its own cube, so the row IS
        // the cube index (position 0 within it).  A row with no cube is an
        // unset cell.
        if self.header.root_type == "TiledCellStMan" {
            return Ok(match self.header.cubes.get(row as usize) {
                Some(c) if !c.cube_shape.is_empty() => Some((row as usize, 0)),
                _ => None,
            });
        }
        // The cube holding `row`. With a row map (TiledShapeStMan) the
        // header's maps are authoritative: a row they do not mention is an
        // unset cell. Without one (TiledColumnStMan) cubes cover consecutive
        // row ranges in creation order, so the ranges partition the rows.
        if !self.header.row_map.is_empty() {
            // Interval maps: entry i covers the rows down to the previous
            // entry, with the cell position counting back from pos_map[i]
            // ("rowMap gives the last row number for which the cubeMap
            // applies"; cube number 0 is the placeholder for "no value").
            let i = self.header.row_map.partition_point(|&r| u64::from(r) < row);
            return Ok(match self.header.row_map.get(i).copied() {
                Some(last_row) if u64::from(last_row) >= row => {
                    let back = u64::from(last_row) - row;
                    let pos = i64::from(self.header.pos_map[i]) - back as i64;
                    if pos < 0 {
                        return Ok(None);
                    }
                    let ci = self.header.cube_map[i] as usize;
                    self.header.cubes.get(ci).and_then(|cube| {
                        // A placeholder cube (casacore writes one for
                        // not-yet-set cells, with no shape) means the
                        // row is unset.
                        let rows = *cube.cube_shape.last().unwrap_or(&0);
                        (rows > 0).then_some((ci, pos as u64))
                    })
                }
                _ => None,
            });
        }
        let mut covered: u64 = 0;
        for (ci, c) in self.header.cubes.iter().enumerate() {
            let rows = *c.cube_shape.last().unwrap_or(&0);
            if rows <= 0 {
                continue;
            }
            if row < covered + rows as u64 {
                return Ok(Some((ci, row - covered)));
            }
            covered += rows as u64;
        }
        Ok(None)
    }

    /// Where the stored cell of `desc` at `row` lives: one contiguous span
    /// when the tile covers the cell, else the per-tile segments of a cell
    /// that spans several tiles.  `None` is an unset cell.
    pub fn cell_place(
        &self,
        desc: &ColumnDesc,
        row: u64,
    ) -> Result<Option<(&TsmCube, CellPlace<'_>)>, TsmError> {
        let Some((ci, row_in_cube)) = self.locate(row)? else {
            return Ok(None);
        };
        let cube = &self.header.cubes[ci];
        let geom = &self.geoms[ci];
        let nrdim = cube.nrdim as usize;
        let bad = || TsmError::BadGeometry {
            cube_shape: cube.cube_shape.clone(),
            tile_shape: cube.tile_shape.clone(),
        };
        if nrdim < 1 || cube.cube_shape.len() != nrdim || cube.tile_shape.len() != nrdim {
            return Err(bad());
        }
        // The per-row cell spans all but the (extensible) row dimension —
        // except in TiledCellStMan, where it is the whole cube.
        let cut = if self.header.root_type == "TiledCellStMan" {
            nrdim
        } else {
            nrdim - 1
        };
        let cell_size: i64 = cube.cube_shape[..cut].iter().product::<i64>();
        let row_tiles: i64 = cube.tile_shape[nrdim - 1];
        if row_tiles <= 0 || cube.tile_shape.iter().any(|&t| t <= 0) {
            return Err(bad());
        }
        let tile_nr = row_in_cube / row_tiles as u64;
        let row_in_tile = (row_in_cube % row_tiles as u64) as i64;
        let tile_size: i64 = cube.tile_shape.iter().product();
        let cell_elems = cell_size as usize;
        // Bool elements are bit-packed in the tile (LSB-first, casacore's
        // `Conversion::bitToBool`); every other element type is byte-based.
        let is_bool = desc.data_type == DataType::Bool;
        let elem_bits: i64 = if is_bool {
            1
        } else {
            elem_size(desc.data_type)? as i64 * 8
        };
        let tile_bits = tile_size * elem_bits;
        // Each tile occupies a whole, byte-aligned bucket (casacore
        // `TSMDataColumn::dataLength`: `(nrPixels + 7) / 8` for Bool; the
        // writer pads the final byte of a bit-packed tile), so a tile starts
        // at `tile_nr * bucket_bytes` — never at a raw bit count, which
        // drifts when the per-tile bit count is not a whole number of bytes
        // (an MS FLAG tile: 158 bools x 26214 rows/tile = 4141812 bits, 4
        // bits over a byte boundary).
        let bucket_bytes = (tile_bits as usize).div_ceil(8);
        let data = geom
            .file
            .map_or(&[][..], |i| self.tile_files[i].1.as_slice());
        let file_seq = cube.file_seq_nr;
        if geom.tiled {
            // Tiles are numbered first-axis-fastest over the tile grid, the
            // row axis last: row layer `tile_nr` starts at tile
            // `tile_nr * layer_tiles`.
            let tc = TiledCell {
                file_seq,
                nelem: cell_elems,
                is_bool,
                esize: if is_bool {
                    1
                } else {
                    elem_size(desc.data_type)?
                },
                layer_off: cube.file_offset as usize
                    + tile_nr as usize * geom.layer_tiles * bucket_bytes,
                bucket: bucket_bytes,
                row_elem: row_in_tile as usize * geom.tile_cell_elems,
                runs: &geom.runs,
                data,
            };
            if tc.end() > data.len() {
                return Err(TsmError::RowOutOfRange {
                    row,
                    nrow: self.header.nrrow,
                });
            }
            return Ok(Some((cube, CellPlace::Tiled(tc))));
        }
        let cell_bits = if is_bool {
            // from the cube: the desc's shape may be absent (variable-shape)
            cell_elems
        } else {
            cell_elems * elem_size(desc.data_type)? * 8
        };
        let bit_off = row_in_tile * cell_bits as i64;
        let byte_off =
            cube.file_offset as usize + tile_nr as usize * bucket_bytes + (bit_off / 8) as usize;
        let skip = (bit_off % 8) as usize;
        let nbytes = (skip + cell_bits).div_ceil(8);
        let end = byte_off + nbytes;
        if end > data.len() {
            return Err(TsmError::RowOutOfRange {
                row,
                nrow: self.header.nrrow,
            });
        }
        Ok(Some((
            cube,
            CellPlace::Contiguous(CellLocation {
                byte_off,
                nbytes,
                skip,
                nelem: cell_elems,
                file_seq,
            }),
        )))
    }

    /// An unset cell (no cube covers its row): the column's default, zeros
    /// in the declared shape, like casacore's storage managers.
    fn read_default_cell(&self, desc: &ColumnDesc) -> Result<RecordValue, TsmError> {
        let casa_shape = desc.shape.as_deref().ok_or(TsmError::RowOutOfRange {
            row: self.header.nrrow,
            nrow: self.header.nrrow,
        })?;
        let cell_elems: usize = casa_shape.iter().product::<i64>().max(0) as usize;
        let nbytes = if desc.data_type == DataType::Bool {
            cell_elems.div_ceil(8)
        } else {
            cell_elems * elem_size(desc.data_type)?
        };
        let zeros = vec![0u8; nbytes];
        let logical: Vec<u32> = casa_shape.iter().rev().map(|&d| d as u32).collect();
        Ok(RecordValue::Array(ArrayValue {
            shape: logical,
            data: decode_tile_data(&zeros, desc.data_type, cell_elems, self.big_endian)?,
        }))
    }
}

/// Row -> entry index into the header's row/cube/pos maps (-1 = unset
/// cell); empty when the header has no maps (TiledColumnStMan).
/// 256-entry table: for each packed byte value, the eight expanded bool
/// values (LSB-first). Mirrors casacore `Conversion::conv_tab` so a single
/// lookup + 8-byte copy (a Vec<bool> element is one byte) decodes one input
/// byte into eight Bools.
const BOOL_LUT: [[bool; 8]; 256] = build_bool_lut();

const fn build_bool_lut() -> [[bool; 8]; 256] {
    let mut tab = [[false; 8]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut bit = 0usize;
        while bit < 8 {
            tab[b][bit] = (b >> bit) & 1 != 0;
            bit += 1;
        }
        b += 1;
    }
    tab
}

/// Decode `nbits` bits starting at bit `skip` of `bytes` (LSB-first within
/// each byte, casacore `Conversion::bitToBool`) to one bool per bit.
fn decode_bits(bytes: &[u8], skip: usize, nbits: usize) -> Result<ArrayData, TsmError> {
    let mut v = vec![false; nbits];
    decode_bits_into(bytes, skip, &mut v);
    Ok(ArrayData::Bool(v))
}

/// Decode `dst.len()` bits starting at bit `skip` of `bytes` (LSB-first)
/// into `dst` — the allocation-free core of [`decode_bits`], also used to
/// fill a caller's numpy bool buffer directly.
pub fn decode_bits_into(bytes: &[u8], skip: usize, dst: &mut [bool]) {
    let nbits = dst.len();
    let base = skip / 8;
    let shift = skip % 8;
    let (full, tail) = dst.as_chunks_mut::<8>();
    if shift == 0 {
        // Byte-aligned fast path: one 256-entry table lookup writes eight
        // 0/1 bytes, matching casacore `bitToBool`'s conv_tab loop.
        for (k, out) in full.iter_mut().enumerate() {
            *out = BOOL_LUT[bytes[base + k] as usize];
        }
    } else {
        // Unaligned: splice each output byte from two input bytes, then the
        // same table lookup (the tail below reads at most one past).
        for (k, out) in full.iter_mut().enumerate() {
            let lo = bytes[base + k] >> shift;
            let hi = bytes.get(base + k + 1).map_or(0, |&h| h << (8 - shift));
            *out = BOOL_LUT[(lo | hi) as usize];
        }
    }
    let done = full.len() * 8;
    for (i, out) in tail.iter_mut().enumerate() {
        let bit = skip + done + i;
        *out = bytes[bit / 8] & (1 << (bit % 8)) != 0;
    }
    debug_assert_eq!(done + tail.len(), nbits);
}

/// Pack bools into bytes, LSB-first (the inverse of [`decode_bits`]).
/// Full bytes use the SWAR gather that mirrors casacore `boolToBit`'s SSE
/// `_mm_cmpeq_epi8` + `_mm_movemask_epi8` (0/1 bytes, one multiply packs 8
/// bits), branch-free and bit-order-correct on any endian host.
fn encode_bits(values: &[bool]) -> Vec<u8> {
    let n = values.len();
    let mut out = vec![0u8; n.div_ceil(8)];
    let mut i = 0usize;
    while i + 8 <= n {
        let mut word = 0u64;
        for j in 0..8 {
            word |= (values[i + j] as u64) << (8 * j);
        }
        out[i >> 3] =
            ((word & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080) >> 56) as u8;
        i += 8;
    }
    let mut acc = 0u8;
    for (k, b) in values[i..].iter().enumerate() {
        if *b {
            acc |= 1 << k;
        }
    }
    if i < n {
        out[i >> 3] = acc;
    }
    out
}

/// Bytes of one tiled element in the tile file.
pub fn elem_size(dt: DataType) -> Result<usize, TsmError> {
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
    if dt == DataType::Bool {
        // Tile Bools are bit-packed (LSB-first); `slice` is byte-aligned.
        return decode_bits(slice, 0, nelem);
    }
    let need = nelem * elem_size(dt)?;
    if slice.len() < need {
        return Err(TsmError::AipsIo(AipsIoError::Truncated {
            needed: need,
            offset: 0,
            len: slice.len(),
        }));
    }
    let slice = &slice[..need];
    // One pass over fixed-size chunks (vectorises to a copy / byte swap)
    // instead of a fallible reader call per element.
    macro_rules! num {
        ($ty:ty, $n:literal) => {
            slice
                .as_chunks::<$n>()
                .0
                .iter()
                .map(|c| {
                    if big_endian {
                        <$ty>::from_be_bytes(*c)
                    } else {
                        <$ty>::from_le_bytes(*c)
                    }
                })
                .collect()
        };
    }
    Ok(match dt {
        DataType::Char | DataType::UChar => ArrayData::UChar(slice.to_vec()),
        DataType::Short => ArrayData::Short(num!(i16, 2)),
        DataType::UShort => ArrayData::UShort(num!(u16, 2)),
        DataType::Int => ArrayData::Int(num!(i32, 4)),
        DataType::UInt => ArrayData::UInt(num!(u32, 4)),
        DataType::Int64 => ArrayData::Int64(num!(i64, 8)),
        DataType::Float => ArrayData::Float(num!(f32, 4)),
        DataType::Double => ArrayData::Double(num!(f64, 8)),
        DataType::Complex => {
            let v: Vec<f32> = num!(f32, 4);
            ArrayData::Complex(v.as_chunks::<2>().0.iter().map(|c| (c[0], c[1])).collect())
        }
        DataType::DComplex => {
            let v: Vec<f64> = num!(f64, 8);
            ArrayData::DComplex(v.as_chunks::<2>().0.iter().map(|c| (c[0], c[1])).collect())
        }
        other => return Err(TsmError::UnsupportedType(other)),
    })
}

/// Parse the TSM header file (always canonical big endian). Both
/// `TiledColumnStMan` and `TiledShapeStMan` write the same layout: their
/// root object followed by one subclass IPosition (fixed cell shape vs
/// default tile shape), then the shared `TiledStMan` payload.
pub fn parse_header(data: &[u8]) -> Result<TsmHeader, TsmError> {
    let mut r = Reader::new(data);
    let obj = r.read_object_start(true)?;
    if !matches!(
        obj.type_name.as_str(),
        "TiledColumnStMan" | "TiledShapeStMan" | "TiledCellStMan"
    ) {
        return Err(TsmError::UnexpectedType {
            expected: "TiledColumnStMan".into(),
            found: obj.type_name,
        });
    }
    let root_type = obj.type_name;
    // Subclass payload: only TiledColumnStMan writes one (the fixed cell
    // shape as an IPosition). TiledShapeStMan's root carries nothing — its
    // default tile shape travels in the (empty) table.dat blob or in a
    // placeholder cube, so it is reconstructed from the cubes below.
    let mut subclass_shape = if root_type == "TiledColumnStMan" || root_type == "TiledCellStMan" {
        r.read_iposition()?
    } else {
        Vec::new()
    };
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
            // TSMFile::putObject (casacore TSMFile.cc): its own object
            // version determines the length width -- version 1 stores the
            // length as u32, version 2 (files >= 2 GiB) as u64. The outer
            // TiledStMan header version must NOT be used here: a v2 header
            // can still hold a >2 GiB tile file, and misreading the length
            // as u32 misaligns every field that follows (nrcube, cubes, and
            // the TiledShapeStMan maps), producing absurd allocations.
            let fver = r.read_u32()?;
            let sequence_nr = r.read_u32()?;
            let length = if fver >= 2 {
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
    // TiledShapeStMan closes with its default tile shape and the row ->
    // (cube, position) maps; TiledColumnStMan ends with the base class.
    let (row_map, cube_map, pos_map) = if root_type == "TiledShapeStMan" {
        let default_tile = r.read_iposition()?;
        subclass_shape = default_tile;
        let nr_used = r.read_u32()?;
        (
            read_block(&mut r, nr_used)?,
            read_block(&mut r, nr_used)?,
            read_block(&mut r, nr_used)?,
        )
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
    Ok(TsmHeader {
        root_type,
        subclass_shape,
        version,
        seq_nr,
        nrrow,
        data_types,
        hypercolumn_name,
        nrdim,
        files,
        cubes,
        row_map,
        cube_map,
        pos_map,
    })
}

/// One casacore `putBlock` payload (`Block` object: version + count +
/// values) with the count supplied by the caller.
fn read_block(r: &mut Reader, count: u32) -> Result<Vec<u32>, TsmError> {
    let obj = r.read_object_start(false)?;
    if obj.type_name != "Block" {
        return Err(TsmError::UnexpectedType {
            expected: "Block".into(),
            found: obj.type_name,
        });
    }
    let stored = r.read_u32()?;
    let n = stored.min(count) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(r.read_u32()?);
    }
    Ok(out)
}

/// Serialize one casacore `putBlock` payload.
fn write_block(w: &mut crate::aipsio::Writer, values: &[u32]) {
    w.put_object_start("Block", 1);
    w.put_u32(values.len() as u32);
    for v in values {
        w.put_u32(*v);
    }
    w.put_object_end();
}
/// Default tile bucket size casacore uses (~512 KiB tiles).
const DEFAULT_TILE_BYTES: usize = 524288;

/// Serialize a tiled storage manager for a single fixed-shape array column:
/// the header file (canonical big-endian AipsIO) and the tile data file.
/// `stman_type` is the manager name the header carries (`TiledColumnStMan`
/// or `TiledShapeStMan` — casacore dispatches on it when re-opening); the
/// subclass payload is the default tile shape for `TiledShapeStMan`, an
/// empty IPosition for `TiledColumnStMan`. `cell_shape` is the fixed
/// per-row cell shape in CASA (reversed logical) dim order; `cells[i]`
/// holds the encoded bytes of that row's cell.  Returns
/// `(table.f{seq} bytes, table.f{seq}_TSM{file_seq} tile bytes, file_seq)` —
/// Tile geometry shared by both TSM writers: rows per (512 KiB) tile, tile
/// count, and the bucket (tile) size in bytes for `nrow` rows of
/// `cell_shape` elements.
pub(crate) struct TsmLayout {
    pub(crate) rows_per_tile: u64,
    pub(crate) n_tiles: u64,
    pub(crate) bucket_size: usize,
}

pub(crate) fn tsm_layout(
    cell_shape: &[i64],
    data_type: DataType,
    nrow: u64,
) -> Result<TsmLayout, TsmError> {
    let elem_size = elem_size(data_type)?;
    let is_bool = data_type == DataType::Bool;
    let cell_elems: i64 = cell_shape.iter().product::<i64>();
    let cell_bytes = if is_bool {
        (cell_elems as usize).div_ceil(8)
    } else {
        cell_elems as usize * elem_size
    };
    let rows_per_tile = (DEFAULT_TILE_BYTES / cell_bytes.max(1)).max(1) as u64;
    let n_tiles = nrow.div_ceil(rows_per_tile).max(1);
    let tile_elems = cell_elems * rows_per_tile as i64;
    let bucket_size = if is_bool {
        (tile_elems as usize).div_ceil(8)
    } else {
        tile_elems as usize * elem_size
    };
    Ok(TsmLayout {
        rows_per_tile,
        n_tiles,
        bucket_size,
    })
}

/// Place `src`'s bits (LSB-first) into `out` starting at bit `bit`,
/// OR-ing over any existing bits.  Full bytes pack eight source bools per
/// SWAR step (the portable form of casacore's `_mm_cmpeq_epi8` +
/// `_mm_movemask_epi8`), straddling destination bytes when unaligned; only
/// the trailing <8 bits fall back to per-bit stores.
fn or_bits_at(out: &mut [u8], src: &[bool], bit: usize) {
    let mut i = 0usize;
    while i + 8 <= src.len() {
        let mut w = 0u64;
        for j in 0..8 {
            w |= (src[i + j] as u64) << (8 * j);
        }
        let byte = ((w & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080) >> 56) as u8;
        let o = (bit + i) >> 3;
        let sh = (bit + i) & 7;
        if sh == 0 {
            out[o] |= byte;
        } else {
            out[o] |= byte << sh;
            out[o + 1] |= byte >> (8 - sh);
        }
        i += 8;
    }
    for (k, b) in src[i..].iter().enumerate() {
        let g = bit + i + k;
        out[g >> 3] |= (*b as u8) << (g & 7);
    }
}

/// Same as [`or_bits_at`] but the source is already a byte-aligned bit
/// stream (`src[0]` holds bits 0..8, LSB-first) — the form the general cell
/// writer keeps per row.
pub fn or_bytes_at(out: &mut [u8], src: &[u8], bit: usize) {
    for (i, byte) in src.iter().enumerate() {
        let o = (bit + i * 8) >> 3;
        let sh = (bit + i * 8) & 7;
        if sh == 0 {
            out[o] |= byte;
        } else {
            out[o] |= byte << sh;
            out[o + 1] |= byte >> (8 - sh);
        }
    }
}

/// OR `n` bits of `src`, starting at bit `src_bit` (LSB-first), into `out`
/// starting at bit `out_bit`: the general (both sides unaligned) form of
/// [`or_bytes_at`], eight bits per step.  Bits of `out` outside
/// `out_bit..out_bit + n` are left untouched.
pub fn or_bits_from(out: &mut [u8], out_bit: usize, src: &[u8], src_bit: usize, n: usize) {
    let mut k = 0usize;
    while k < n {
        let take = (n - k).min(8);
        let sb = src_bit + k;
        let (i, sh) = (sb >> 3, sb & 7);
        let mut byte = src[i] >> sh;
        if sh != 0 && sh + take > 8 {
            byte |= src[i + 1] << (8 - sh);
        }
        if take < 8 {
            byte &= (1u8 << take) - 1;
        }
        let ob = out_bit + k;
        let (o, osh) = (ob >> 3, ob & 7);
        out[o] |= byte << osh;
        if osh != 0 && osh + take > 8 {
            out[o + 1] |= byte >> (8 - osh);
        }
        k += take;
    }
}

/// The writes both TSM writers pass to [`tsm_header`].
struct TsmHeaderParams<'a> {
    stman_type: &'a str,
    big_endian: bool,
    seq_nr: u32,
    hypercolumn_name: &'a str,
    data_type: DataType,
    cell_shape: &'a [i64],
    nrow: u64,
    layout: &'a TsmLayout,
    tile_file_len: usize,
    /// The tile shape's cell dimensions when they differ from the cell
    /// (tests of casacore's sub-cell tiling); `None` = tiles cover the cell.
    tile_cell: Option<&'a [i64]>,
}

/// The canonical big-endian AipsIO TSM header, shared by both writers.  For
/// TiledShapeStMan this must reproduce casacore's exact layout or casacore
/// cannot open the table: two file entries (a placeholder file 0, then the
/// real tile file with sequence number 1), a placeholder cube 0 (no shape,
/// file -1) before the real cube, and the singleHypercube row maps (verified
/// against a casacore-written MS FLAG header).
fn tsm_header(p: TsmHeaderParams<'_>) -> Vec<u8> {
    let TsmHeaderParams {
        stman_type,
        big_endian,
        seq_nr,
        hypercolumn_name,
        data_type,
        cell_shape,
        nrow,
        layout,
        tile_file_len,
        tile_cell,
    } = p;
    let mut cube_shape = cell_shape.to_vec();
    cube_shape.push(nrow as i64);
    let mut tile_shape = tile_cell.unwrap_or(cell_shape).to_vec();
    tile_shape.push(layout.rows_per_tile as i64);

    let mut hw = crate::aipsio::Writer::new();
    hw.put_root_object_start(stman_type, 1);
    // Subclass payload: only TiledColumnStMan writes the fixed cell shape
    // IPosition; TiledShapeStMan's root carries nothing (casacore reads the
    // default tile shape from the cubes).
    if stman_type == "TiledColumnStMan" {
        hw.put_object_start("IPosition", 1);
        hw.put_u32(0);
        hw.put_object_end();
    }
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
                                             // Files: casacore's TiledShapeStMan always keeps a placeholder file 0
                                             // and numbers the first real tile file 1 (the tile lives in
                                             // `table.f{seq}_TSM1`); TiledColumnStMan uses a single file 0.
    let is_shape = stman_type == "TiledShapeStMan";
    let real_file_seq: i32 = if is_shape { 1 } else { 0 };
    let nrfile: u32 = if is_shape { 2 } else { 1 };
    hw.put_u32(nrfile);
    if is_shape {
        hw.put_bool(false); // file[0]: placeholder, absent
    }
    hw.put_bool(true);
    // TSMFile::putObject: version 1 stores the length as a u32; a tile file
    // past the u32 range needs version 2 (Int64 length), which casacore and
    // `parse_header` both read.
    let wide = tile_file_len as u64 > u64::from(u32::MAX);
    hw.put_u32(if wide { 2 } else { 1 }); // TSMFile version
    hw.put_u32(real_file_seq as u32); // TSMFile sequence number
    if wide {
        hw.put_u64(tile_file_len as u64); // TSMFile length
    } else {
        hw.put_u32(tile_file_len as u32);
    }
    // Cubes: TiledShapeStMan keeps a placeholder cube 0 (extensible=false,
    // no shape, file -1) and the real cube at index 1; the singleHypercube
    // row maps reference that index.
    let nrcube: u32 = if is_shape { 2 } else { 1 };
    hw.put_u32(nrcube);
    if is_shape {
        // cube[0]: placeholder for not-yet-set cells.
        hw.put_u32(1); // cube version
        put_empty_values_record(&mut hw);
        hw.put_bool(false); // extensible
        hw.put_i32(0); // nrdim
        put_iposition(&mut hw, &[]); // cube_shape
        put_iposition(&mut hw, &[]); // tile_shape
        hw.put_i32(-1); // file seq nr (no file)
        hw.put_u32(0); // file offset
    }
    // cube[1] (or the sole cube for TiledColumnStMan): the real data cube.
    hw.put_u32(1); // cube version
    put_empty_values_record(&mut hw);
    hw.put_bool(true); // extensible
    hw.put_i32(cube_shape.len() as i32);
    put_iposition(&mut hw, &cube_shape);
    put_iposition(&mut hw, &tile_shape);
    hw.put_i32(real_file_seq); // file sequence nr
    hw.put_u32(0); // file offset (cube version 1)
    hw.put_object_end(); // TiledStMan
    if is_shape {
        // TiledShapeStMan closes with the default tile shape and the
        // row-interval maps (rowMap/cubeMap/posMap hold the LAST row of each
        // interval, its cube and the last cell position).  A single real
        // cube covering every row collapses to casacore's singleHypercube:
        // one entry mapping row nrow-1 to cube 1 at position nrow-1.
        put_iposition(&mut hw, &tile_shape);
        if nrow == 0 {
            // casacore's putBlock writes the (empty) Block objects even
            // with no entries, and both readers expect them: leaving them
            // out made an empty table unreadable ("buffer too short").
            hw.put_u32(0);
            write_block(&mut hw, &[]);
            write_block(&mut hw, &[]);
            write_block(&mut hw, &[]);
        } else {
            hw.put_u32(1);
            write_block(&mut hw, &[nrow as u32 - 1]);
            write_block(&mut hw, &[1]);
            write_block(&mut hw, &[nrow as u32 - 1]);
        }
    }
    hw.put_object_end(); // stman_type
    hw.into_bytes()
}

/// The header of a casacure-layout TSM column of `nrow` rows (one real
/// cube, default tiles), the length its tile file must have, and the tile
/// file's sequence number: what growing a column in place rewrites.  The
/// tile data itself is unchanged by growth — tile `t` stays at
/// `t * bucket_size` and new rows land in zeroed (default) bytes.
pub(crate) fn tsm_grown_header(
    stman_type: &str,
    big_endian: bool,
    seq_nr: u32,
    hypercolumn_name: &str,
    data_type: DataType,
    cell_shape: &[i64],
    nrow: u64,
) -> Result<(Vec<u8>, u64, u32), TsmError> {
    let layout = tsm_layout(cell_shape, data_type, nrow)?;
    let tile_file_len = layout.bucket_size * layout.n_tiles as usize;
    let hdr = tsm_header(TsmHeaderParams {
        stman_type,
        big_endian,
        seq_nr,
        hypercolumn_name,
        data_type,
        cell_shape,
        nrow,
        layout: &layout,
        tile_file_len,
        tile_cell: None,
    });
    let file_seq = u32::from(stman_type == "TiledShapeStMan");
    Ok((hdr, tile_file_len as u64, file_seq))
}

/// Whether `header` has exactly the layout casacure writes for a column of
/// `cell_shape` cells (CASA order): one real cube at file offset 0 with the
/// default tile shape, in the tile file [`tsm_grown_header`] names, and
/// (TiledShapeStMan) the single-hypercube row map.  Only such a column can
/// be grown in place by rewriting its header.
pub(crate) fn is_casacure_layout(
    header: &TsmHeader,
    data_type: DataType,
    cell_shape: &[i64],
) -> bool {
    let is_shape = header.root_type == "TiledShapeStMan";
    let Ok(layout) = tsm_layout(cell_shape, data_type, header.nrrow) else {
        return false;
    };
    let real = usize::from(is_shape);
    let file_seq = real as u32;
    if header.cubes.len() != real + 1 || header.files.len() != 1 {
        return false;
    }
    if header.files[0].sequence_nr != file_seq {
        return false;
    }
    let cube = &header.cubes[real];
    let mut cube_shape = cell_shape.to_vec();
    cube_shape.push(header.nrrow as i64);
    let mut tile_shape = cell_shape.to_vec();
    tile_shape.push(layout.rows_per_tile as i64);
    let maps_ok = !is_shape
        || (header.nrrow == 0 && header.row_map.is_empty())
        || (header.row_map == [header.nrrow as u32 - 1]
            && header.cube_map == [1]
            && header.pos_map == [header.nrrow as u32 - 1]);
    cube.file_seq_nr == file_seq as i32
        && cube.file_offset == 0
        && cube.cube_shape == cube_shape
        && cube.tile_shape == tile_shape
        && maps_ok
}

/// Write a whole bool column straight from its row bit-slices: the
/// typed-buffer form of [`write_tsm_file`].  `rows[r]` holds cell `r`'s
/// `cell_elems` bools (LSB-first); rows pack contiguously into the tiles —
/// no per-row byte-aligning, intermediate `Vec<u8>` cells, or set-bit
/// scatter at write time.
pub fn write_tsm_file_bool(
    stman_type: &str,
    big_endian: bool,
    seq_nr: u32,
    hypercolumn_name: &str,
    cell_shape: &[i64],
    rows: &[&[bool]],
) -> Result<(Vec<u8>, Vec<u8>, u32), TsmError> {
    if stman_type != "TiledColumnStMan" && stman_type != "TiledShapeStMan" {
        return Err(TsmError::UnexpectedType {
            expected: "TiledColumnStMan".into(),
            found: stman_type.into(),
        });
    }
    let cell_elems = cell_shape.iter().product::<i64>();
    if cell_elems <= 0 {
        return Err(TsmError::UnsupportedType(DataType::Bool));
    }
    let cell_elems = cell_elems as usize;
    if rows.iter().any(|r| r.len() != cell_elems) {
        return Err(TsmError::UnsupportedType(DataType::Bool));
    }
    let nrow = rows.len() as u64;
    let layout = tsm_layout(cell_shape, DataType::Bool, nrow)?;
    let mut tile_file = vec![0u8; layout.bucket_size * layout.n_tiles as usize];
    for (row, r) in rows.iter().enumerate() {
        let tile = row as u64 / layout.rows_per_tile;
        let in_tile = row as u64 % layout.rows_per_tile;
        let bucket = tile as usize * layout.bucket_size;
        or_bits_at(&mut tile_file[bucket..], r, in_tile as usize * cell_elems);
    }
    let hdr = tsm_header(TsmHeaderParams {
        stman_type,
        big_endian,
        seq_nr,
        hypercolumn_name,
        data_type: DataType::Bool,
        cell_shape,
        nrow,
        layout: &layout,
        tile_file_len: tile_file.len(),
        tile_cell: None,
    });
    let real_file_seq = if stman_type == "TiledShapeStMan" {
        1
    } else {
        0
    };
    Ok((hdr, tile_file, real_file_seq))
}

/// casacore numbers TiledShapeStMan's first real tile file 1.
pub fn write_tsm_file(
    stman_type: &str,
    big_endian: bool,
    seq_nr: u32,
    hypercolumn_name: &str,
    data_type: DataType,
    cell_shape: &[i64],
    cells: &[Vec<u8>],
) -> Result<(Vec<u8>, Vec<u8>, u32), TsmError> {
    if stman_type != "TiledColumnStMan" && stman_type != "TiledShapeStMan" {
        return Err(TsmError::UnexpectedType {
            expected: "TiledColumnStMan".into(),
            found: stman_type.into(),
        });
    }
    let elem_size = elem_size(data_type)?;
    let is_bool = data_type == DataType::Bool;
    let cell_elems: i64 = cell_shape.iter().product::<i64>();
    let cell_bytes = if is_bool {
        (cell_elems as usize).div_ceil(8)
    } else {
        cell_elems as usize * elem_size
    };
    if cells.iter().any(|c| c.len() != cell_bytes) {
        return Err(TsmError::UnsupportedType(data_type));
    }
    let nrow = cells.len() as u64;
    let layout = tsm_layout(cell_shape, data_type, nrow)?;

    // Tile data file (data-file endianness; rows fill the tile grid). Bool
    // cells are bit-packed (LSB-first), so a row's bits start at an
    // arbitrary bit of the bucket: each byte-aligned cell is placed at the
    // row's bit position (word-oriented, not bit-by-bit).
    let mut tile_file = vec![0u8; layout.bucket_size * layout.n_tiles as usize];
    for (row, cell) in cells.iter().enumerate() {
        let tile = row as u64 / layout.rows_per_tile;
        let in_tile = row as u64 % layout.rows_per_tile;
        let bucket = tile as usize * layout.bucket_size;
        if is_bool {
            or_bytes_at(
                &mut tile_file[bucket..],
                cell,
                in_tile as usize * cell_elems as usize,
            );
        } else {
            let off = bucket + in_tile as usize * cell_bytes;
            tile_file[off..off + cell_bytes].copy_from_slice(cell);
        }
    }

    let hdr = tsm_header(TsmHeaderParams {
        stman_type,
        big_endian,
        seq_nr,
        hypercolumn_name,
        data_type,
        cell_shape,
        nrow,
        layout: &layout,
        tile_file_len: tile_file.len(),
        tile_cell: None,
    });
    let real_file_seq = if stman_type == "TiledShapeStMan" {
        1
    } else {
        0
    };
    Ok((hdr, tile_file, real_file_seq as u32))
}

/// The hypercolumn 'values' Record casacore stores in each TSM cube: an
/// empty Record (RecordDesc with no fields) whose record type is 1 —
/// byte-verified against a casacore-written MS FLAG header (the type field
/// is 1, not the 0 a plain empty record would carry).
fn put_empty_values_record(hw: &mut crate::aipsio::Writer) {
    hw.put_object_start("Record", 1);
    hw.put_object_start("RecordDesc", 2);
    hw.put_i32(0); // no fields
    hw.put_object_end();
    hw.put_i32(1); // record type (Fixed)
    hw.put_object_end();
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
        return Ok(encode_bits(v));
    }
    crate::ssm::encode_array_data(big_endian, data)
        .map_err(|_| TsmError::UnsupportedType(data_type))
}

/// Test support: casacore-layout tile files with tiles smaller than the
/// cell, built element by element from the layout rules (independently of
/// the reader's run plan).
#[cfg(test)]
pub(crate) mod subtile_testutil {
    use super::*;

    /// Linear first-axis-fastest index of `pos` in a grid of `shape`.
    fn linear(pos: &[usize], shape: &[usize]) -> usize {
        pos.iter()
            .zip(shape)
            .rev()
            .fold(0usize, |acc, (&p, &s)| acc * s + p)
    }

    /// Where element `k` (CASA linear index within the cell) of `row` lives:
    /// (tile number, element within the tile).
    pub(crate) fn element_slot(cell: &[i64], tile: &[i64], row: usize, k: usize) -> (usize, usize) {
        let n = cell.len();
        let mut pos = Vec::with_capacity(n + 1);
        let mut rem = k;
        for &c in cell {
            pos.push(rem % c as usize);
            rem /= c as usize;
        }
        pos.push(row);
        let tile_u: Vec<usize> = tile.iter().map(|&t| t as usize).collect();
        let tcoord: Vec<usize> = pos.iter().zip(&tile_u).map(|(p, t)| p / t).collect();
        let within: Vec<usize> = pos.iter().zip(&tile_u).map(|(p, t)| p % t).collect();
        let mut grid: Vec<usize> = cell
            .iter()
            .zip(&tile_u)
            .map(|(&c, &t)| (c as usize).div_ceil(t))
            .collect();
        grid.push(usize::MAX); // the row axis is last: its extent never matters
        (linear(&tcoord, &grid), linear(&within, &tile_u))
    }

    /// The pieces [`write_subtiled`] returns.
    pub(crate) struct SubTiled {
        pub header: Vec<u8>,
        pub tile_file: Vec<u8>,
        /// Per bit of `tile_file`: whether it belongs to a valid element
        /// (everything else is padding, filled with `pad`).
        pub valid_bits: Vec<bool>,
        pub file_seq: u32,
    }

    /// Serialise `cells` (one per row, CASA shape `cell`) with tile shape
    /// `tile` (cell dims + rows per tile) in casacore's layout, padding
    /// bytes/bits set from `pad`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_subtiled(
        stman_type: &str,
        big_endian: bool,
        seq_nr: u32,
        hypercolumn_name: &str,
        data_type: DataType,
        cell: &[i64],
        tile: &[i64],
        cells: &[ArrayData],
        pad: u8,
    ) -> SubTiled {
        assert_eq!(tile.len(), cell.len() + 1);
        let nrow = cells.len();
        let is_bool = data_type == DataType::Bool;
        let es = elem_size(data_type).unwrap();
        let tile_elems: usize = tile.iter().product::<i64>() as usize;
        let bucket = if is_bool {
            tile_elems.div_ceil(8)
        } else {
            tile_elems * es
        };
        let layer: usize = cell
            .iter()
            .zip(tile)
            .map(|(&c, &t)| (c as usize).div_ceil(t as usize))
            .product();
        let row_tiles = nrow.div_ceil(tile[cell.len()] as usize).max(1);
        let n_tiles = layer * row_tiles;
        let mut out = vec![pad; bucket * n_tiles];
        let mut valid = vec![false; out.len() * 8];
        let cell_elems: usize = cell.iter().product::<i64>() as usize;
        for (row, data) in cells.iter().enumerate() {
            let enc = tsm_encode_cell(big_endian, data_type, data).unwrap();
            for k in 0..cell_elems {
                let (t, e) = element_slot(cell, tile, row, k);
                if is_bool {
                    let bit = t * bucket * 8 + e;
                    let v = enc[k / 8] >> (k % 8) & 1;
                    out[bit / 8] = (out[bit / 8] & !(1 << (bit % 8))) | (v << (bit % 8));
                    valid[bit] = true;
                } else {
                    let off = t * bucket + e * es;
                    out[off..off + es].copy_from_slice(&enc[k * es..(k + 1) * es]);
                    valid[off * 8..(off + es) * 8]
                        .iter_mut()
                        .for_each(|b| *b = true);
                }
            }
        }
        let layout = TsmLayout {
            rows_per_tile: tile[cell.len()] as u64,
            n_tiles: n_tiles as u64,
            bucket_size: bucket,
        };
        let header = tsm_header(TsmHeaderParams {
            stman_type,
            big_endian,
            seq_nr,
            hypercolumn_name,
            data_type,
            cell_shape: cell,
            nrow: nrow as u64,
            layout: &layout,
            tile_file_len: out.len(),
            tile_cell: Some(&tile[..cell.len()]),
        });
        SubTiled {
            header,
            tile_file: out,
            valid_bits: valid,
            file_seq: u32::from(stman_type == "TiledShapeStMan"),
        }
    }

    /// Assert every padding bit of `file` still holds `pad`'s bit.
    pub(crate) fn assert_padding_intact(file: &[u8], valid_bits: &[bool], pad: u8) {
        for (bit, &v) in valid_bits.iter().enumerate() {
            if !v {
                let got = file[bit / 8] >> (bit % 8) & 1;
                assert_eq!(got, pad >> (bit % 8) & 1, "padding bit {bit} changed");
            }
        }
    }
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
        let (header_bytes, tile_data, _) = write_tsm_file(
            "TiledColumnStMan",
            big_endian,
            0,
            "TiledData_GROUP",
            dt,
            &casa_shape,
            &cells,
        )
        .unwrap();
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

        let tsm = TsmFile::from_header(header, tile_data, big_endian);
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
    fn bool_tiles_are_bit_packed() {
        let data = ArrayData::Bool(vec![true, false, true, true, false, false]);
        let cells = vec![tsm_encode_cell(false, DataType::Bool, &data).unwrap()];
        let (header, tile_data, _) = write_tsm_file(
            "TiledColumnStMan",
            false,
            0,
            "g",
            DataType::Bool,
            &[3, 2],
            &cells,
        )
        .unwrap();
        // The six elements pack into one byte, LSB-first: bits 0, 2, 3 set.
        assert_eq!(&tile_data[..1], &[0b1101]);
        // And it still decodes to the original cell.
        let header = parse_header(&header).unwrap();
        let tsm = TsmFile::from_header(header, tile_data, false);
        let want = RecordValue::Array(ArrayValue {
            shape: vec![2, 3],
            data,
        });
        assert_eq!(tsm.read_cell(&array_desc(DataType::Bool), 0).unwrap(), want);
    }

    /// Bit-packed Bool rows straddle byte boundaries: row 1 of a 3-element
    /// cell starts at bit 3, so its bits come from two bytes.
    #[test]
    fn bool_rows_straddle_byte_boundaries() {
        let mut rows = Vec::new();
        for r in 0..4u8 {
            rows.push(
                tsm_encode_cell(
                    false,
                    DataType::Bool,
                    &ArrayData::Bool(vec![r & 1 != 0, r & 2 != 0, r & 4 != 0]),
                )
                .unwrap(),
            );
        }
        let (header, tile_data, _) = write_tsm_file(
            "TiledShapeStMan",
            false,
            0,
            "g",
            DataType::Bool,
            &[3],
            &rows,
        )
        .unwrap();
        let header = parse_header(&header).unwrap();
        let tsm = TsmFile::from_header(header, tile_data, false);
        let mut desc = array_desc(DataType::Bool);
        desc.shape = Some(vec![3]);
        desc.ndim = 1;
        for r in 0..4u8 {
            assert_eq!(
                tsm.read_cell(&desc, r as u64).unwrap(),
                RecordValue::Array(ArrayValue {
                    shape: vec![3],
                    data: ArrayData::Bool(vec![r & 1 != 0, r & 2 != 0, r & 4 != 0]),
                }),
                "row {r}"
            );
        }
    }

    /// The LUT/SWAR bit conversions must agree with the naive scalar form
    /// on every bit pattern, including non-byte-aligned starts, tails, and
    /// partial bytes (casacore `bitToBool`/`boolToBit` parity).
    #[test]
    fn bit_conversions_match_scalar() {
        fn scalar_decode(bytes: &[u8], skip: usize, nbits: usize) -> Vec<bool> {
            (0..nbits)
                .map(|i| bytes[(skip + i) / 8] & (1 << ((skip + i) % 8)) != 0)
                .collect()
        }
        fn scalar_encode(values: &[bool]) -> Vec<u8> {
            let mut out = vec![0u8; values.len().div_ceil(8)];
            for (i, b) in values.iter().enumerate() {
                if *b {
                    out[i / 8] |= 1 << (i % 8);
                }
            }
            out
        }
        let mut rng = 0x1234_5678_9abc_def0u64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as u8
        };
        for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 64, 65, 1000, 4097] {
            let bytes: Vec<u8> = (0..len.div_ceil(8)).map(|_| next()).collect();
            for skip in (0..16).chain([31, 63]) {
                let nbits = len.saturating_sub(skip);
                let vals = scalar_decode(&bytes, skip, nbits);
                let ArrayData::Bool(got) = decode_bits(&bytes, skip, nbits).unwrap() else {
                    panic!("decode len {len} skip {skip}");
                };
                assert_eq!(got, vals, "decode len {len} skip {skip}");
                // encode() packs from bit 0 (no skip), LSB-first, incl. tails.
                let enc = encode_bits(&vals);
                assert_eq!(enc, scalar_encode(&vals), "encode len {len} skip {skip}");
                let ArrayData::Bool(back) = decode_bits(&enc, 0, nbits).unwrap() else {
                    panic!("re-decode len {len} skip {skip}");
                };
                assert_eq!(back, vals, "round-trip len {len} skip {skip}");
            }
        }
        // Spot-check a fixed byte against casacore's conv_tab values.
        let ArrayData::Bool(b8) = decode_bits(&[0b1010_0101], 0, 8).unwrap() else {
            panic!();
        };
        assert_eq!(
            b8,
            vec![true, false, true, false, false, true, false, true],
            "0xa5 decodes LSB-first"
        );
        assert_eq!(encode_bits(&b8), vec![0b1010_0101]);
    }

    /// The typed-buffer bool writer must produce byte-identical header and
    /// tile files to the general per-cell writer for the same rows, for
    /// both TSM storage managers and bit counts that force unaligned rows
    /// and multi-tile columns.
    #[test]
    fn bool_typed_writer_matches_general_writer() {
        let mut rng = 0xdead_beef_cafe_f00du64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as u8
        };
        for &stman in &["TiledColumnStMan", "TiledShapeStMan"] {
            for &cell_elems in &[3usize, 6, 8, 9, 158, 316] {
                let nrow = 500 + (next() as usize % 400); // up to ~2 tiles
                let rows: Vec<Vec<bool>> = (0..nrow)
                    .map(|_| (0..cell_elems).map(|_| next() & 1 != 0).collect())
                    .collect();
                let shape = vec![cell_elems as i64];
                let cells: Vec<Vec<u8>> = rows.iter().map(|r| encode_bits(r)).collect();
                let general =
                    write_tsm_file(stman, false, 0, "g", DataType::Bool, &shape, &cells).unwrap();
                let typed_rows: Vec<&[bool]> = rows.iter().map(|r| r.as_slice()).collect();
                let typed = write_tsm_file_bool(stman, false, 0, "g", &shape, &typed_rows).unwrap();
                assert_eq!(general.0, typed.0, "header {stman} cell_elems {cell_elems}");
                assert_eq!(general.1, typed.1, "tile {stman} cell_elems {cell_elems}");
                assert_eq!(
                    general.2, typed.2,
                    "file_seq {stman} cell_elems {cell_elems}"
                );
            }
        }
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
        let (header, tile_data, _) =
            write_tsm_file("TiledColumnStMan", false, 0, "g", dt, &[1024], &cells).unwrap();
        let header = parse_header(&header).unwrap();
        let rows_per_tile = header.cubes[0].tile_shape[1];
        assert_eq!(rows_per_tile, 32);
        assert!(rows_per_tile * 2 < 65, "test must span tiles");
        assert_eq!(header.cubes[0].cube_shape, vec![1024, 65]);

        let tsm = TsmFile::from_header(header, tile_data, false);
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
        let (header, tile_data, _) = write_tsm_file(
            "TiledColumnStMan",
            false,
            0,
            "g",
            DataType::Int,
            &[3, 2],
            &cells,
        )
        .unwrap();
        let header = parse_header(&header).unwrap();
        let tsm = TsmFile::from_header(header, tile_data, false);
        assert!(matches!(
            tsm.read_cell(&array_desc(DataType::Int), 1),
            Err(TsmError::RowOutOfRange { row: 1, nrow: 1 })
        ));
    }

    #[test]
    fn rejects_unsupported_tiled_element_type() {
        assert!(matches!(
            elem_size(DataType::String),
            Err(TsmError::UnsupportedType(DataType::String))
        ));
    }

    #[test]
    fn write_rejects_inconsistent_cell_sizes() {
        // One 1-byte and one 2-byte (u16) cell: 4-element Int cells are 16 B.
        let cells = vec![vec![0u8; 8], vec![0u8; 9]];
        assert!(matches!(
            write_tsm_file(
                "TiledColumnStMan",
                false,
                0,
                "g",
                DataType::Int,
                &[3, 2],
                &cells
            ),
            Err(TsmError::UnsupportedType(DataType::Int))
        ));
    }

    #[test]
    fn header_rejects_wrong_root_type() {
        let cells = vec![vec![0u8; 4]];
        let (header, _, _) = write_tsm_file(
            "TiledColumnStMan",
            false,
            0,
            "g",
            DataType::Int,
            &[1],
            &cells,
        )
        .unwrap();
        // Corrupt the root type name "TiledColumnStMan" in place.
        let mut bad = header;
        bad[12] = b'X';
        assert!(matches!(
            parse_header(&bad),
            Err(TsmError::UnexpectedType { .. })
        ));
    }

    /// A deterministic value for every element type: element `k` of `row`.
    fn elem_values(dt: DataType, row: usize, n: usize) -> ArrayData {
        let v = |k: usize| (row * 1000 + k) as i64;
        match dt {
            DataType::Bool => ArrayData::Bool((0..n).map(|k| (row * 7 + k * 3) % 5 < 2).collect()),
            DataType::UChar => ArrayData::UChar((0..n).map(|k| v(k) as u8).collect()),
            DataType::Short => ArrayData::Short((0..n).map(|k| -(v(k) as i16)).collect()),
            DataType::UShort => ArrayData::UShort((0..n).map(|k| v(k) as u16).collect()),
            DataType::Int => ArrayData::Int((0..n).map(|k| -(v(k) as i32)).collect()),
            DataType::UInt => ArrayData::UInt((0..n).map(|k| v(k) as u32).collect()),
            DataType::Int64 => ArrayData::Int64((0..n).map(|k| -v(k) << 20).collect()),
            DataType::Float => ArrayData::Float((0..n).map(|k| v(k) as f32 + 0.5).collect()),
            DataType::Double => ArrayData::Double((0..n).map(|k| v(k) as f64 * 1.25).collect()),
            DataType::Complex => {
                ArrayData::Complex((0..n).map(|k| (v(k) as f32, -(k as f32) - 0.5)).collect())
            }
            DataType::DComplex => {
                ArrayData::DComplex((0..n).map(|k| (v(k) as f64, -(k as f64) - 0.25)).collect())
            }
            other => panic!("{other:?} is not a tiled element type"),
        }
    }

    fn desc_for(dt: DataType, cell: &[i64]) -> ColumnDesc {
        let mut d = array_desc(dt);
        d.shape = Some(cell.to_vec());
        d.ndim = cell.len() as i32;
        d
    }

    const ALL_TILED_TYPES: [DataType; 11] = [
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
    ];

    /// Tiles smaller than the cell (casacore/dask-ms layout): every cell
    /// reads back for every element type, both endians and both storage
    /// managers, with edge padding on both cell axes and a partial last row
    /// tile.  Padding is filled with 0xA5 so a read of it shows up.
    #[test]
    fn sub_cell_tiles_read_back() {
        use super::subtile_testutil::write_subtiled;
        // (cell, tile) pairs in CASA order, the last tile dim = rows/tile.
        let geoms: [(&[i64], &[i64]); 5] = [
            (&[3, 5], &[2, 2, 3]),  // padding on both cell axes
            (&[2, 79], &[2, 8, 4]), // the MS case (8-channel tiles)
            (&[4], &[3, 2]),        // 1-D cell
            (&[3, 5], &[4, 8, 2]),  // tile larger than the cell
            (&[2, 3, 4], &[1, 2, 3, 2]),
        ];
        for (cell, tile) in geoms {
            let nelem: usize = cell.iter().product::<i64>() as usize;
            for stman in ["TiledColumnStMan", "TiledShapeStMan"] {
                for &big in &[false, true] {
                    for dt in ALL_TILED_TYPES {
                        let data: Vec<ArrayData> =
                            (0..7).map(|r| elem_values(dt, r, nelem)).collect();
                        let st = write_subtiled(stman, big, 0, "g", dt, cell, tile, &data, 0xA5);
                        let header = parse_header(&st.header).unwrap();
                        let tsm = TsmFile::from_header(header, st.tile_file, big);
                        let desc = desc_for(dt, cell);
                        let logical: Vec<u32> = cell.iter().rev().map(|&d| d as u32).collect();
                        for (row, d) in data.iter().enumerate() {
                            let got = tsm.read_cell(&desc, row as u64).unwrap();
                            assert_eq!(
                                got,
                                RecordValue::Array(ArrayValue {
                                    shape: logical.clone(),
                                    data: d.clone()
                                }),
                                "{stman} big={big} {dt:?} cell {cell:?} tile {tile:?} row {row}"
                            );
                            // The single-span accessors refuse a multi-tile
                            // cell rather than return part of it.
                            assert!(matches!(
                                tsm.cell_span(&desc, row as u64),
                                Err(TsmError::TileTooSmall { .. })
                            ));
                            // The raw gathers agree with the encoder.
                            let Some((_, CellPlace::Tiled(tc))) =
                                tsm.cell_place(&desc, row as u64).unwrap()
                            else {
                                panic!("expected a multi-tile cell");
                            };
                            let enc = tsm_encode_cell(big, dt, d).unwrap();
                            let mut buf = Vec::new();
                            if dt == DataType::Bool {
                                tc.gather_bits(&mut buf);
                            } else {
                                tc.gather_bytes(&mut buf);
                            }
                            assert_eq!(buf, enc, "{dt:?} raw gather row {row}");
                        }
                    }
                }
            }
        }
    }

    /// The MS geometry's gather plan: [2, 79] cells in [2, 8] tiles merge
    /// to one run per tile (16 contiguous elements, 14 in the edge tile).
    #[test]
    fn sub_cell_runs_merge_per_tile() {
        let runs = tile_runs(&[2, 79], &[2, 8]);
        assert_eq!(runs.len(), 10);
        for (k, r) in runs.iter().enumerate() {
            assert_eq!((r.tile, r.src, r.dst), (k, 0, 16 * k));
            assert_eq!(r.len, if k == 9 { 14 } else { 16 });
        }
        // Axis 0 split: lines along axis 0 of 2 (then 1) elements.
        let runs = tile_runs(&[3, 2], &[2, 2]);
        assert_eq!(
            runs,
            vec![
                TileRun {
                    tile: 0,
                    src: 0,
                    dst: 0,
                    len: 2
                },
                TileRun {
                    tile: 0,
                    src: 2,
                    dst: 3,
                    len: 2
                },
                TileRun {
                    tile: 1,
                    src: 0,
                    dst: 2,
                    len: 1
                },
                TileRun {
                    tile: 1,
                    src: 2,
                    dst: 5,
                    len: 1
                },
            ]
        );
    }

    /// A multi-cube TiledShapeStMan column (one cube per cell shape, each in
    /// its own tile file, rows interleaved through the maps — the layout
    /// casacore writes for alternating shapes), both cubes sub-cell tiled.
    #[test]
    fn multi_file_multi_cube_sub_cell_tiles() {
        use super::subtile_testutil::write_subtiled;
        let (ca, cb): (&[i64], &[i64]) = (&[2, 5], &[3, 7]);
        let tile: &[i64] = &[2, 3, 4];
        for &big in &[false, true] {
            for dt in [DataType::Int, DataType::Bool, DataType::Complex] {
                let na = 10usize;
                let nb = 21usize;
                // Rows 0,2,4 in cube A (positions 0..3), 1,3,5 in cube B.
                let da: Vec<ArrayData> = (0..3).map(|p| elem_values(dt, 2 * p, na)).collect();
                let db: Vec<ArrayData> = (0..3).map(|p| elem_values(dt, 2 * p + 1, nb)).collect();
                let sa = write_subtiled("TiledShapeStMan", big, 0, "g", dt, ca, tile, &da, 0x5A);
                let sb = write_subtiled("TiledShapeStMan", big, 0, "g", dt, cb, tile, &db, 0x5A);
                let ha = parse_header(&sa.header).unwrap();
                let hb = parse_header(&sb.header).unwrap();
                let mut cube_b = hb.cubes[1].clone();
                cube_b.file_seq_nr = 2;
                let mut header = ha.clone();
                header.nrrow = 6;
                header.cubes.push(cube_b);
                header.row_map = (0..6).collect();
                header.cube_map = vec![1, 2, 1, 2, 1, 2];
                header.pos_map = vec![0, 0, 1, 1, 2, 2];
                let tsm = TsmFile::from_header_files(
                    header,
                    vec![(1, sa.tile_file), (2, sb.tile_file)],
                    big,
                );
                let desc = desc_for(dt, &[]);
                for row in 0..6usize {
                    let (want, cell) = if row % 2 == 0 {
                        (da[row / 2].clone(), ca)
                    } else {
                        (db[row / 2].clone(), cb)
                    };
                    let got = tsm.read_cell(&desc, row as u64).unwrap();
                    assert_eq!(
                        got,
                        RecordValue::Array(ArrayValue {
                            shape: cell.iter().rev().map(|&d| d as u32).collect(),
                            data: want
                        }),
                        "big={big} {dt:?} row {row}"
                    );
                    let Some((_, CellPlace::Tiled(tc))) =
                        tsm.cell_place(&desc, row as u64).unwrap()
                    else {
                        panic!("expected a multi-tile cell");
                    };
                    assert_eq!(tc.file_seq, if row % 2 == 0 { 1 } else { 2 });
                }
            }
        }
    }

    /// A tiled cell whose tiles are missing from the file is out of range,
    /// not a panic.
    #[test]
    fn sub_cell_tiles_truncated_file_is_an_error() {
        use super::subtile_testutil::write_subtiled;
        let data: Vec<ArrayData> = (0..5).map(|r| elem_values(DataType::Int, r, 15)).collect();
        let st = write_subtiled(
            "TiledColumnStMan",
            false,
            0,
            "g",
            DataType::Int,
            &[3, 5],
            &[2, 2, 3],
            &data,
            0,
        );
        let header = parse_header(&st.header).unwrap();
        let mut short = st.tile_file.clone();
        // Drop the last tile (12 Ints) of the second row layer.
        short.truncate(short.len() - 48);
        let tsm = TsmFile::from_header(header, short, false);
        let desc = desc_for(DataType::Int, &[3, 5]);
        assert!(tsm.read_cell(&desc, 0).is_ok());
        assert!(matches!(
            tsm.read_cell(&desc, 4),
            Err(TsmError::RowOutOfRange { .. })
        ));
    }

    /// `or_bits_from` places arbitrary bit ranges like a per-bit loop and
    /// leaves the surrounding bits alone.
    #[test]
    fn or_bits_from_matches_per_bit_loop() {
        let src: Vec<u8> = (0..12u32)
            .map(|i| (i.wrapping_mul(0x9d) ^ 0x3c) as u8)
            .collect();
        for src_bit in 0..17 {
            for out_bit in 0..17 {
                for n in 0..40 {
                    let mut got = vec![0u8; 12];
                    or_bits_from(&mut got, out_bit, &src, src_bit, n);
                    let mut want = vec![0u8; 12];
                    for i in 0..n {
                        let b = src[(src_bit + i) / 8] >> ((src_bit + i) % 8) & 1;
                        want[(out_bit + i) / 8] |= b << ((out_bit + i) % 8);
                    }
                    assert_eq!(got, want, "src_bit {src_bit} out_bit {out_bit} n {n}");
                }
            }
        }
    }

    /// casacore writes a placeholder cube at index 0 (no shape, no file)
    /// and interval maps; rows resolve through the interval, and rows
    /// covered by no interval (or by the placeholder) read as defaults.
    #[test]
    fn casacore_placeholder_and_interval_maps() {
        let cube = |rows: i64, offset: u64| TsmCube {
            extensible: true,
            nrdim: 2,
            cube_shape: vec![2, rows],
            tile_shape: vec![2, 1],
            file_seq_nr: 0,
            file_offset: offset,
        };
        let mut placeholder = cube(0, 0);
        placeholder.cube_shape = Vec::new();
        placeholder.tile_shape = Vec::new();
        placeholder.file_seq_nr = -1;
        let header = TsmHeader {
            root_type: "TiledShapeStMan".into(),
            subclass_shape: Vec::new(),
            version: 2,
            seq_nr: 0,
            nrrow: 4,
            data_types: vec![DataType::Short],
            hypercolumn_name: "g".into(),
            nrdim: 2,
            files: Vec::new(),
            // Cube 0 is casacore's placeholder; cube 1 holds rows 0-2
            // (the interval entry covers up to row 2), row 3 is unset.
            cubes: vec![placeholder, cube(3, 0)],
            row_map: vec![2],
            cube_map: vec![1],
            pos_map: vec![2],
        };
        let cell = |r: i16| {
            let mut v = Vec::new();
            v.extend_from_slice(&r.to_le_bytes());
            v.extend_from_slice(&(100 + r).to_le_bytes());
            v
        };
        let tile_data: Vec<u8> = [cell(0), cell(1), cell(2)].concat();
        let tsm = TsmFile::from_header(header, tile_data, false);
        let mut desc = array_desc(DataType::Short);
        desc.shape = Some(vec![2]);
        desc.ndim = 1;
        let want = |r: i16| {
            RecordValue::Array(ArrayValue {
                shape: vec![2],
                data: ArrayData::Short(vec![r, 100 + r]),
            })
        };
        let zero = RecordValue::Array(ArrayValue {
            shape: vec![2],
            data: ArrayData::Short(vec![0, 0]),
        });
        assert_eq!(tsm.read_cell(&desc, 0).unwrap(), want(0));
        assert_eq!(tsm.read_cell(&desc, 1).unwrap(), want(1));
        assert_eq!(tsm.read_cell(&desc, 2).unwrap(), want(2));
        assert_eq!(tsm.read_cell(&desc, 3).unwrap(), zero, "unset row");
    }

    /// A TiledShapeStMan-written file round-trips: the header carries the
    /// shape-stman root and the default tile shape, and cells read back.
    #[test]
    fn tiled_shape_stman_round_trip() {
        let data: Vec<ArrayData> = (0..5)
            .map(|row| ArrayData::Int(vec![row, -row, row + 1, -row - 1, row + 2, -row - 2]))
            .collect();
        let cells: Vec<Vec<u8>> = data
            .iter()
            .map(|d| tsm_encode_cell(false, DataType::Int, d).unwrap())
            .collect();
        let (header_bytes, tile_data, _tsfile_seq) = write_tsm_file(
            "TiledShapeStMan",
            false,
            7,
            "TiledData",
            DataType::Int,
            &[3, 2],
            &cells,
        )
        .unwrap();
        let header = parse_header(&header_bytes).unwrap();
        assert_eq!(header.root_type, "TiledShapeStMan");
        assert_eq!(header.seq_nr, 7);
        // casacore layout: a placeholder cube[0] precedes the real data cube.
        assert_eq!(header.cubes.len(), 2);
        assert_eq!(header.cubes[0].nrdim, 0);
        assert_eq!(
            header.subclass_shape, header.cubes[1].tile_shape,
            "the header carries the default tile shape"
        );
        let tsm = TsmFile::from_header(header, tile_data, false);
        let desc = array_desc(DataType::Int);
        for (row, d) in data.iter().enumerate() {
            assert_eq!(
                tsm.read_cell(&desc, row as u64).unwrap(),
                RecordValue::Array(ArrayValue {
                    shape: vec![2, 3],
                    data: d.clone()
                })
            );
        }
    }

    /// Rows beyond every cube are unset cells and read as the column's
    /// default (zeros in the declared shape), like casacore.
    #[test]
    fn rows_outside_any_cube_read_as_defaults() {
        let cells =
            vec![
                tsm_encode_cell(false, DataType::Float, &ArrayData::Float(vec![1.0, 2.0])).unwrap(),
            ];
        // One cube holding a single row; the header claims 3 rows total.
        let (header_bytes, tile_data, _tsfile_seq) = write_tsm_file(
            "TiledShapeStMan",
            false,
            0,
            "g",
            DataType::Float,
            &[2],
            &cells,
        )
        .unwrap();
        let mut header = parse_header(&header_bytes).unwrap();
        header.nrrow = 3;
        // The real data cube is cube[1] (cube[0] is the placeholder).
        header.cubes[1].cube_shape[1] = 1;
        let tsm = TsmFile::from_header(header, tile_data, false);
        let mut desc = array_desc(DataType::Float);
        desc.shape = Some(vec![2]);
        desc.ndim = 1;
        // Row 0 is stored; rows 1-2 are unset and read as zeros.
        assert_eq!(
            tsm.read_cell(&desc, 0).unwrap(),
            RecordValue::Array(ArrayValue {
                shape: vec![2],
                data: ArrayData::Float(vec![1.0, 2.0])
            })
        );
        for row in 1..3u64 {
            assert_eq!(
                tsm.read_cell(&desc, row).unwrap(),
                RecordValue::Array(ArrayValue {
                    shape: vec![2],
                    data: ArrayData::Float(vec![0.0, 0.0])
                }),
                "row {row}"
            );
        }
    }

    /// A second cube continues the row range: rows map to the cube whose
    /// range contains them (TiledShapeStMan opens a cube per cell shape).
    /// The geometry is hand-built: two 1-row-tile cubes, one bucket each.
    #[test]
    fn multi_cube_rows_map_in_order() {
        let cube = |rows: i64, offset: u64| TsmCube {
            extensible: true,
            nrdim: 2,
            cube_shape: vec![2, rows],
            tile_shape: vec![2, 1],
            file_seq_nr: 0,
            file_offset: offset,
        };
        let header = TsmHeader {
            root_type: "TiledShapeStMan".into(),
            subclass_shape: vec![2, 1],
            version: 2,
            seq_nr: 0,
            nrrow: 3,
            data_types: vec![DataType::Short],
            hypercolumn_name: "g".into(),
            nrdim: 2,
            files: Vec::new(),
            // Cube 0 holds rows 0-1 (one 4-byte bucket each), cube 1
            // continues with row 2.
            cubes: vec![cube(2, 0), cube(1, 8)],
            row_map: Vec::new(),
            cube_map: Vec::new(),
            pos_map: Vec::new(),
        };
        // Row r's cell = shorts [r, 100+r], little-endian, one 4-byte bucket
        // per row (tile = one row).
        let cell = |r: i16| {
            let mut v = Vec::new();
            v.extend_from_slice(&r.to_le_bytes());
            v.extend_from_slice(&(100 + r).to_le_bytes());
            v
        };
        let tile_data: Vec<u8> = [cell(0), cell(1), cell(2)].concat();
        let tsm = TsmFile::from_header(header, tile_data, false);
        let mut desc = array_desc(DataType::Short);
        desc.shape = Some(vec![2]);
        desc.ndim = 1;
        let want = |row: i16| {
            RecordValue::Array(ArrayValue {
                shape: vec![2],
                data: ArrayData::Short(vec![row, 100 + row]),
            })
        };
        assert_eq!(tsm.read_cell(&desc, 0).unwrap(), want(0));
        assert_eq!(tsm.read_cell(&desc, 1).unwrap(), want(1));
        assert_eq!(tsm.read_cell(&desc, 2).unwrap(), want(2));
    }
}
