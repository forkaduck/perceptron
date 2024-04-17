use log::info;

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

    fn assert_ask(data: &Layer, input: &[f64], rv: bool) {
        let result = data.output(input);
        info!("{:?} -> {:.6} | {}", input, result.0, result.1);

        assert_eq!(data.output(input).1, rv);
    }

    let training_data = TrainingData::try_from(vec![
        (vec![0.0, 0.0, 0.0], 0.0),
        (vec![0.0, 0.7, 0.0], 1.0),
        (vec![0.0, 0.8, 0.0], 1.0),
        (vec![1.0, 0.0, 1.0], 0.0),
        (vec![1.0, 0.6, 0.0], 1.0),
        (vec![0.0, 0.7, 1.0], 1.0),
        (vec![0.0, 0.0, 0.0], 0.0),
        (vec![1.0, 0.0, 1.0], 0.0),
        (vec![1.0, 1.0, 1.0], 1.0),
        // (vec![0.0, 0.8, 0.0], 1.0),
    ])
    .unwrap();

    let mut eye = Layer::new(training_data.len(), 0.5, false);

    eye.train_optimizer(&training_data, 0.005..0.3).unwrap();

    info!("Without noise.");
    assert_ask(&eye, &[0.0, 0.7, 0.0], true);
    assert_ask(&eye, &[0.0, 0.5, 0.0], true);
    assert_ask(&eye, &[0.0, 0.2, 0.0], false);

    info!("With noise.");
    assert_ask(&eye, &[0.8, 0.7, 0.3], true);
    assert_ask(&eye, &[0.3, 0.5, 1.0], true);
    assert_ask(&eye, &[0.8, 0.2, 0.2], false);
}
