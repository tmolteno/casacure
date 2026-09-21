//! Dump every SSM index of a table (debug helper for storage layouts).

fn main() {
    let dir = std::env::args().nth(1).expect("usage: dump_ssm_index <msdir>");
    let dat_bytes = std::fs::read(format!("{dir}/table.dat")).unwrap();
    let dat = casacure::table::parse_table_dat(&dat_bytes).unwrap();
    for dm in &dat.column_set.data_managers {
        if dm.type_name != "StandardStMan" {
            continue;
        }
        let path = format!("{dir}/table.f{}", dm.sequence_nr);
        let len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let file = match casacure::ssm::StandardStManFile::open(
            std::path::Path::new(&dir),
            dm.sequence_nr,
            dat.header.big_endian,
        ) {
            Ok(f) => f,
            Err(e) => {
                println!("seq {} ({} bytes): open failed: {e}", dm.sequence_nr, len);
                continue;
            }
        };
        println!(
            "seq {} file {} bytes, bucket_size {}, {} index object(s)",
            dm.sequence_nr,
            len,
            file.header.bucket_size,
            file.indices.len()
        );
        for (i, ix) in file.indices.iter().enumerate() {
            println!(
                "  index {i}: rows_per_bucket {} nr_columns {} nUsed {} last_row {:?} buckets {:?}",
                ix.rows_per_bucket,
                ix.nr_columns,
                ix.last_row.len(),
                &ix.last_row[..ix.last_row.len().min(6)],
                &ix.bucket_number[..ix.bucket_number.len().min(6)],
            );
        }
    }
}
