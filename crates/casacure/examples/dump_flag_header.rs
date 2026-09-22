//! Throwaway: dump the parsed TSM header of a table's FLAG group for
//! comparing casacore-written vs casacure-written TSM headers.
//! Usage: dump_flag_header <table_dir> <seq>
use casacure::tsm::TsmFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).unwrap();
    let seq: u32 = std::env::args().nth(2).unwrap().parse()?;
    let t = TsmFile::open(&dir, seq, false)?;
    println!("root_type      = {}", t.header.root_type);
    println!("subclass_shape = {:?}", t.header.subclass_shape);
    println!("version        = {}", t.header.version);
    println!("seq_nr         = {}", t.header.seq_nr);
    println!("nrrow          = {}", t.header.nrrow);
    println!("data_types     = {:?}", t.header.data_types);
    println!("hypercol       = {}", t.header.hypercolumn_name);
    println!("nrdim          = {}", t.header.nrdim);
    println!("files          = {:?}", t.header.files);
    for (i, c) in t.header.cubes.iter().enumerate() {
        println!(
            "cube[{i}] extensible={} nrdim={} cube_shape={:?} tile_shape={:?} file={} off={}",
            c.extensible, c.nrdim, c.cube_shape, c.tile_shape, c.file_seq_nr, c.file_offset
        );
    }
    println!("row_map.len    = {}", t.header.row_map.len());
    println!("cube_map.len   = {}", t.header.cube_map.len());
    println!("row_map = {:?}", t.header.row_map);
    println!("cube_map= {:?}", t.header.cube_map);
    println!("pos_map = {:?}", t.header.pos_map);
    Ok(())
}
