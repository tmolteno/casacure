//! Dump the ColumnSet bindings and per-file SSM index summary (debug helper).

fn main() {
    let dir = std::env::args().nth(1).expect("usage: dump_ssm <msdir>");
    let dat_bytes = std::fs::read(format!("{dir}/table.dat")).unwrap();
    let dat = casacure::table::parse_table_dat(&dat_bytes).unwrap();
    let dms = &dat.column_set.data_managers;
    for c in &dat.column_set.columns {
        let dm = dms.iter().find(|d| d.sequence_nr == c.data_manager_seq);
        println!(
            "col {:<22} seq {:<3} dm {:?}",
            c.original_name,
            c.data_manager_seq,
            dm.map(|d| d.type_name.as_str())
        );
    }
    for dm in dms {
        if dm.type_name != "StandardStMan" && dm.type_name != "IncrementalStMan" {
            continue;
        }
        let f = format!("{}/table.f{}", dir, dm.sequence_nr);
        let len = std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
        match &dm.blob {
            casacure::columnset::DataManagerBlob::StandardStMan(ssm) => {
                println!(
                    "seq {} {} file len {} col_offset {:?}",
                    dm.sequence_nr,
                    dm.type_name,
                    len,
                    ssm.column_offset
                );
            }
            _ => {
                println!("seq {} {} file len {}", dm.sequence_nr, dm.type_name, len);
            }
        }
    }
}
