use log::{info, warn};

mod layer;
mod test;
mod training_data;

use crate::layer::Layer;
use crate::training_data::TrainingData;

fn main() {
    simple_logger::SimpleLogger::new()
        .env()
        .without_timestamps()
        .init()
        .unwrap();

    warn!("Nothing to see here, move along!");
}
