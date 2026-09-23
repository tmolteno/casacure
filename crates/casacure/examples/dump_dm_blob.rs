//! Dump the raw per-data-manager spec blobs of a table.dat (debug helper).

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_dm_blob <table.dat>");
    let buf = std::fs::read(&path).unwrap();
    let dat = casacure::table::parse_table_dat(&buf).unwrap();
    for dm in &dat.column_set.data_managers {
        println!(
            "dm {} seq={} blob={}",
            dm.type_name,
            dm.sequence_nr,
            match &dm.blob {
                casacure::columnset::DataManagerBlob::StandardStMan(_) => "<SSM>".to_string(),
                casacure::columnset::DataManagerBlob::Unsupported(b) => {
                    format!("{:02x?}", b.iter().take(96).collect::<Vec<_>>())
                }
            }
        );
    }
}
