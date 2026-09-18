fn main() {
    let t = casacure::Table::open("/tmp/bool100.tab", false).unwrap();
    let b = t.colnames().iter().position(|c| c == "B").unwrap();
    let vals = t.getcol(b, 0, 100).unwrap();
    let got: Vec<i32> = vals
        .iter()
        .map(|v| matches!(v, casacure::record::RecordValue::Bool(true)) as i32)
        .collect();
    println!("first 12: {:?}", &got[..12]);
    println!(
        "expected: {:?}",
        (0..12).map(|i| (i % 3 == 0) as i32).collect::<Vec<_>>()
    );
}
