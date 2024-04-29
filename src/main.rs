use log::warn;

mod group;
mod layer;
mod test;
mod training_data;

fn main() {
    simple_logger::SimpleLogger::new()
        .env()
        .without_timestamps()
        .init()
        .unwrap();

    warn!("Nothing to see here, move along!");
}
