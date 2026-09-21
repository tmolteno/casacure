//! Dump a TSM header incl. the TiledShapeStMan row maps (debug helper).

fn main() {
    let dir = std::env::args().nth(1).expect("dir");
    let seq: u32 = std::env::args().nth(2).unwrap().parse().unwrap();
    let data = std::fs::read(format!("{dir}/table.f{seq}")).unwrap();
    let header = casacure::tsm::parse_header(&data).unwrap();
    println!("root {}", header.root_type);
    println!("subclass {:?}", header.subclass_shape);
    for c in &header.cubes {
        println!(
            "cube {:?} tile {:?} seq {} off {}",
            c.cube_shape, c.tile_shape, c.file_seq_nr, c.file_offset
        );
    }
    println!(
        "row_map {:?}\ncube_map {:?}\npos_map {:?}",
        header.row_map, header.cube_map, header.pos_map
    );
}
