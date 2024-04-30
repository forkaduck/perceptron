#[cfg(test)]
mod layer_tests {
    use log::info;
    use std::time::Instant;

    use crate::layer::Layer;
    use crate::test_helper::*;
    use crate::training_data::TrainingData;

    /// Showcases that a one-dimensional layer can't learn
    /// an XOR Gate.
    #[test]
    fn xor_impossibility_weak() {
        setup();
        let start = Instant::now();

        let training_data = TrainingData::try_from(vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ])
        .unwrap();

        let mut layer = Layer::new(training_data.input_length(), false);

        layer.train(&training_data, 0.1, 0.1).0.ok();

        assert_show(&layer, &[0.0, 0.0], false);
        assert_show(&layer, &[0.0, 1.0], false);
        assert_show(&layer, &[1.0, 0.0], false);
        assert_show(&layer, &[1.0, 1.0], true);

        timer_end(start);
    }

    /// Shows how multiple iterations with the same data
    /// and a nearing learning-strength can strengthen pattern recognition.
    #[test]
    fn optimizer_complex_patterns() {
        setup();
        let start = Instant::now();

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

        let mut layer = Layer::new(training_data.input_length(), false);

        layer
            .train_optimizer(&training_data, 0.005..0.3, 0.3)
            .0
            .unwrap();
        layer.train(&training_data, 0.055, 0.3).0.unwrap();

        info!("Without noise.");
        assert_show(&layer, &[0.0, 0.7, 0.0], true);
        assert_show(&layer, &[0.0, 0.5, 0.0], true);
        assert_show(&layer, &[0.0, 0.2, 0.0], false);

        info!("With noise.");
        assert_show(&layer, &[0.8, 0.7, 0.3], true);
        assert_show(&layer, &[0.3, 0.5, 1.0], true);
        assert_show(&layer, &[0.8, 0.2, 0.2], false);

        timer_end(start);
    }

    /// Shows the anti-proof of optimizer_complex_patterns.
    #[test]
    fn optimizer_proof() {
        setup();
        let start = Instant::now();

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

        let mut layer = Layer::new(training_data.input_length(), false);

        layer.train(&training_data, 0.055, 0.3).0.unwrap();

        info!("Without noise.");
        assert_show(&layer, &[0.0, 0.7, 0.0], false);
        assert_show(&layer, &[0.0, 0.5, 0.0], false);
        assert_show(&layer, &[0.0, 0.2, 0.0], false);

        info!("With noise.");
        assert_show(&layer, &[0.8, 0.7, 0.3], true);
        assert_show(&layer, &[0.3, 0.5, 1.0], true);
        assert_show(&layer, &[0.8, 0.2, 0.2], false);

        timer_end(start);
    }
}
