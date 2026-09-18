use casacure::ms::default_ms;
fn main() {
    let path = std::path::PathBuf::from(std::env::args().nth(1).expect("path"));
    let extra = r#"{"DATA":{"_c_order":true,"comment":"The DATA column","dataManagerGroup":"StandardStMan","dataManagerType":"StandardStMan","keywords":{"UNIT":"Jy"},"maxlen":0,"ndim":2,"option":0,"valueType":"COMPLEX"}}"#;
    default_ms(&path, Some(extra)).unwrap();
    println!("created {}", path.display());
}
