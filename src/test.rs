#[cfg(test)]
mod tests {
    use crate::training_data::TrainingData;
    use crate::Layer;
    use log::info;

    fn setup() {
        match simple_logger::SimpleLogger::new()
            .env()
            .without_timestamps()
            .init()
        {
            Ok(()) => {}
            Err(_) => {}
        };
    }

    fn assert_ask(data: &Layer, input: &[f64], rv: bool) {
        let result = data.output(input);
        info!("{:?} -> {:.6} | {}", input, result.0, result.1);

        assert_eq!(data.output(input).1, rv);
    }

    #[test]
    fn xor_impossibility_weak() {
        setup();

        let training_data = TrainingData::from(vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ]);

        let mut eye = Layer::new(training_data.len(), 0.5, false);

        eye.train(&training_data, 0.1, 10, 0.1).unwrap();

        assert_ask(&eye, &[0.0, 0.0], false);
        assert_ask(&eye, &[0.0, 1.0], false);
        assert_ask(&eye, &[1.0, 0.0], false);
        assert_ask(&eye, &[1.0, 1.0], true);
    }

    #[test]
    fn normal() {
        setup();

        let training_data = TrainingData::from(vec![
            (vec![0.0, 0.0, 0.0], 0.0),
            (vec![0.0, 0.7, 0.0], 1.0),
            (vec![0.0, 0.8, 0.0], 1.0),
            (vec![1.0, 0.0, 1.0], 0.0),
            (vec![1.0, 0.6, 0.0], 1.0),
            (vec![0.0, 0.7, 1.0], 1.0),
            (vec![0.0, 0.0, 0.0], 0.0),
            (vec![1.0, 0.0, 1.0], 0.0),
            (vec![1.0, 1.0, 1.0], 1.0),
        ]);

        let mut eye = Layer::new(training_data.len(), 0.5, false);

        eye.train(&training_data, 0.1, 200, 0.01).unwrap();

        info!("Without noise.");
        assert_ask(&eye, &[0.0, 0.7, 0.0], true);
        assert_ask(&eye, &[0.0, 0.5, 0.0], false);
        assert_ask(&eye, &[0.0, 0.2, 0.0], false);

        info!("With noise.");
        assert_ask(&eye, &[0.8, 0.7, 0.3], true);
        assert_ask(&eye, &[0.3, 0.5, 1.0], false);
        assert_ask(&eye, &[0.8, 0.2, 0.2], false);
    }
}
