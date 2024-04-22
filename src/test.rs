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

    fn assert_show(data: &Layer, input: &[f64], rv: bool) {
        let result = data.output(input);
        info!("{:?} -> {:.6} | {}", input, result.0, result.1);

        assert_eq!(data.output(input).1, rv);
    }

    /// Showcases that a one-dimensional layer can't learn
    /// an XOR Gate.
    #[test]
    fn xor_impossibility_weak() {
        setup();

        let training_data = TrainingData::try_from(vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ])
        .unwrap();

        let mut eye = Layer::new(training_data.len(), 0.5, false);

        eye.train(&training_data, 0.1, 0.1).ok();

        assert_show(&eye, &[0.0, 0.0], false);
        assert_show(&eye, &[0.0, 1.0], false);
        assert_show(&eye, &[1.0, 0.0], false);
        assert_show(&eye, &[1.0, 1.0], true);
    }

    /// Shows how multiple iterations with the same data
    /// and a nearing learning-strength can strengthen pattern recognition.
    #[test]
    fn optimizer_complex_patterns() {
        setup();

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
        ])
        .unwrap();

        let mut eye = Layer::new(training_data.len(), 0.5, false);

        eye.train_optimizer(&training_data, 0.005..0.3, 0.3)
            .unwrap();
        eye.train(&training_data, 0.055, 0.3).unwrap();

        info!("Without noise.");
        assert_show(&eye, &[0.0, 0.7, 0.0], true);
        assert_show(&eye, &[0.0, 0.5, 0.0], true);
        assert_show(&eye, &[0.0, 0.2, 0.0], false);

        info!("With noise.");
        assert_show(&eye, &[0.8, 0.7, 0.3], true);
        assert_show(&eye, &[0.3, 0.5, 1.0], true);
        assert_show(&eye, &[0.8, 0.2, 0.2], false);
    }

    /// Shows the anti-proof of optimizer_complex_patterns.
    #[test]
    fn optimizer_proof() {
        setup();
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
        ])
        .unwrap();

        let mut eye = Layer::new(training_data.len(), 0.5, false);

        eye.train(&training_data, 0.055, 0.3).unwrap();

        info!("Without noise.");
        assert_show(&eye, &[0.0, 0.7, 0.0], false);
        assert_show(&eye, &[0.0, 0.5, 0.0], false);
        assert_show(&eye, &[0.0, 0.2, 0.0], false);

        info!("With noise.");
        assert_show(&eye, &[0.8, 0.7, 0.3], true);
        assert_show(&eye, &[0.3, 0.5, 1.0], true);
        assert_show(&eye, &[0.8, 0.2, 0.2], false);
    }
}
