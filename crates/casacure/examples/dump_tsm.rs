//! Dump the TiledColumnStMan header of a table's `table.f{seq}` file
//! (casacore `TiledStMan::headerFileGet`). Used to inspect the hypercube /
//! tile geometry while developing the TSM reader.
//!
//! ```sh
//! cargo run -p casacure --example dump_tsm -- /tmp/tsmt.tab 0
//! ```

use casacure::aipsio::Writer;
use casacure::record::DataType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use casacure::aipsio::Reader;
    let dir = std::env::args().nth(1).expect("missing table dir");
    let seq: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let data = std::fs::read(format!("{dir}/table.f{seq}"))?;
    // TSM header files are always canonical big-endian AipsIO.
    let mut r = Reader::new(&data);
    let obj = r.read_object_start(true)?;
    println!("root {} version {}", obj.type_name, obj.version);
    // The TiledColumnStMan subclass writes its tile shape as the first
    // payload, then the base TiledStMan object.
    let subclass_shape = r.read_iposition()?;
    println!("TiledColumnStMan tileShape {subclass_shape:?}");
    let base = r.read_object_start(false)?;
    println!("base {} version {}", base.type_name, base.version);
    let version = base.version;
    let _stored_big = if version >= 2 { r.read_bool()? } else { true };
    let seqnr = r.read_u32()?;
    println!("seqnr {seqnr}");
    let nrrow = if version >= 3 {
        r.read_u64()?
    } else {
        u64::from(r.read_u32()?)
    };
    println!("nrrow {nrrow}");
    let ncol = r.read_u32()?;
    let mut dtypes = Vec::new();
    for _ in 0..ncol {
        dtypes.push(DataType::from_i32(r.read_i32()?)?);
    }
    println!("dtypes {dtypes:?}");
    let hcn = r.read_string()?;
    println!("hypercolumnName {hcn:?}");
    let pers = r.read_u32()?;
    println!("persMaxCacheSize {pers}");
    let nrdim = r.read_i32()?;
    println!("nrdim {nrdim}");
    let nrfile = if version >= 3 {
        r.read_u64()?
    } else {
        u64::from(r.read_u32()?)
    };
    println!("nrfile {nrfile}");
    for f in 0..nrfile {
        let exists = r.read_bool()?;
        if exists {
            let fver = r.read_u32()?;
            let fseq = r.read_u32()?;
            let len = if version >= 3 {
                r.read_u64()?
            } else {
                u64::from(r.read_u32()?)
            };
            println!("  file {f} version {fver} seqnr {fseq} length {len}");
        } else {
            println!("  file {f} exists false");
        }
    }
    let nrcube = if version >= 3 {
        r.read_u64()?
    } else {
        u64::from(r.read_u32()?)
    };
    println!("nrCube {nrcube}");
    for k in 0..nrcube {
        let cver = r.read_u32()?;
        // The hypercolumn 'values' Record (usually empty): skip its framed
        // RecordDesc by length.
        let vobj = r.read_object_start(false)?;
        let vpayload = vobj.length as usize - (4 + 4 + vobj.type_name.len() + 4);
        println!(
            "  cube {k} version {cver} values type {} ({vpayload} bytes)",
            vobj.type_name
        );
        r.skip(vpayload)?;
        let extensible = r.read_bool()?;
        let cnrdim = r.read_i32()?;
        let cube_shape = r.read_iposition()?;
        let tile_shape = r.read_iposition()?;
        let fseq = r.read_i32()?;
        let foff = if cver == 1 {
            u64::from(r.read_u32()?)
        } else {
            r.read_u64()?
        };
        println!("    extensible {extensible} nrdim {cnrdim} cubeShape {cube_shape:?} tileShape {tile_shape:?} fileSeqnr {fseq} fileOffset {foff}");
    }
    println!("consumed {} of {}", r.position(), data.len());
    let _ = Writer::new;
    Ok(())
}
