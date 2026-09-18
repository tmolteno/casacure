fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("table dir");
    let t = casacure::Table::open(dir, true).unwrap();
    println!("colnames: {:?}", t.colnames());
    println!("keywords: {}", t.getkeywords());
    println!("colkw0:   {}", t.getcolkeywords(0).unwrap());
}
