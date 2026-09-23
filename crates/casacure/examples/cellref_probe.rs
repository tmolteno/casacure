fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let dat = casacure::parse_table_dat(&std::fs::read(dir.join("table.dat")).unwrap()).unwrap();
    let file = casacure::StandardStManFile::open(&dir, 0, false).unwrap();
    let dm = &dat.column_set.data_managers[0];
    let casacure::columnset::DataManagerBlob::StandardStMan(spec) = &dm.blob else {
        panic!("not ssm")
    };
    println!(
        "spec: col_offset={:?} col_index_map={:?}",
        spec.column_offset, spec.col_index_map
    );
    let s_idx = dat.desc.columns.iter().position(|c| c.name == "S").unwrap();
    for row in 0..2u64 {
        // the 12-byte variable-string cell: our read_scalar_cell handles it,
        // but for the ARRAY column use the bucket-ref directly.
        let index_nr = spec.col_index_map[s_idx] as usize;
        let offset = spec.column_offset[s_idx];
        let cell = file.cell_bytes(index_nr, offset, row, 12).unwrap();
        // 3 big-endian ints: bucket, offset, len
        let b = i32::from_be_bytes(cell.0[0..4].try_into().unwrap());
        let o = i32::from_be_bytes(cell.0[4..8].try_into().unwrap());
        let l = i32::from_be_bytes(cell.0[8..12].try_into().unwrap());
        println!(
            "S row {row} ref: bucket={b} offset={o} len={l} cell={:?}",
            cell.0
                .iter()
                .map(|x| format!("{x:02x}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        // bucket data start = 512 + bucket*bucket_size; content at +16 (header) + offset?
        let bs = file.header.bucket_size as usize;
        let base = 512 + (b as usize) * bs;
        let data = std::fs::read(dir.join("table.f0")).unwrap();
        let start = base + o as usize;
        println!(
            "   content at abs {}: {}",
            start,
            data[start..(start + l.max(0) as usize).min(data.len())]
                .iter()
                .map(|x| format!("{x:02x}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}
