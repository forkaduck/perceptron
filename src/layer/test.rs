#[cfg(test)]
mod layer_tests {
    use log::info;
    use std::time::Instant;

    use crate::layer::{Layer, LayerError};
    use crate::test_helper::*;
    use crate::training_data::TrainingData;

    /// Test that random weights initialization works.
    #[test]
    fn random_init() {
        let layer = Layer::new(4, true);

        let mut iter = layer.weights.iter().peekable();

        while let Some(i) = iter.next() {
            match iter.peek() {
                Some(a) => {
                    assert_ne!(i, *a);
                }
                None => (),
            }
        }
    }

    /// Tests impossible error margin with the given data.
    /// Can be somewhat mitigated by using the optimizer.
    #[test]
    fn stabilized_layer() {
        setup();
        let start = Instant::now();

        let training_data = TrainingData::try_from(vec![
            (vec![0.0, 0.0, 1.0], 0.0),
            (vec![0.0, 1.0, 1.0], 0.0),
            (vec![1.0, 0.0, 0.0], 1.0),
            (vec![1.0, 1.0, 0.0], 1.0),
            //
            (vec![0.0, 1.0, 1.0], 1.0),
            (vec![0.0, 1.0, 0.0], 0.0),
            (vec![1.0, 1.0, 0.0], 1.0),
        ])
        .unwrap();
        let mut layer = Layer::new(training_data.input_length(), false);

        assert_eq!(
            layer.train(&training_data, 0.001, 0.01).0,
            Err(LayerError::ErrStabilized)
        );

        assert_show(&layer, &[0.0, 0.0], false);
        assert_show(&layer, &[0.0, 1.0], false);
        assert_show(&layer, &[1.0, 0.0], true);
        assert_show(&layer, &[1.0, 1.0], true);

        timer_end(start);
    }

    /// Don't give the optimizer enough range to find an optimal strength.
    #[test]
    fn bad_range_choice() {
        setup();
        let start = Instant::now();

        let training_data = TrainingData::try_from(vec![
            (vec![0.0, 0.0, 0.0], 0.0),
            (vec![0.0, 1.0, 0.0], 1.0),
            (vec![1.0, 0.0, 1.0], 1.0),
            (vec![1.0, 1.0, 1.0], 1.0),
        ])
        .unwrap();
        let mut layer = Layer::new(training_data.input_length(), false);

        assert_eq!(
            layer.train_optimizer(&training_data, 1.0..2.0, 0.1).0,
            Err(LayerError::ErrRising)
        );

        timer_end(start);
    }

    #[test]
    fn normal_operation() {
        setup();
        let start = Instant::now();

        let training_data = TrainingData::try_from(vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 0.5),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 1.0),
        ])
        .unwrap();

        let mut layer = Layer::new(training_data.input_length(), false);

        // FIX Wrong parameters?
        layer.train(&training_data, 0.1, 0.1).0.ok();

        assert_show(&layer, &[0.0, 0.0], false);
        assert_show(&layer, &[0.0, 1.0], true);
        assert_show(&layer, &[1.0, 0.0], true);
        assert_show(&layer, &[1.0, 1.0], true);

        timer_end(start);
    }

    /// Showcases that a one-dimensional layer can't learn
    /// an XOR Gate.
    #[test]
    fn xor_impossibility() {
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
        assert_show(&layer, &[1.0, 1.0], false);

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
            .train_optimizer(&training_data, 0.001..0.5, 0.5)
            .0
            .unwrap();

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

    /// Shows how learning without optimizing the parameters, leads to worse results,
    /// even if you take the optimal learning strength from the optimizer and learn manually.
    ///
    /// This suggests some kind of important role of iteratively improving learning uptake which
    /// seems to increase retention?
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

        layer.train(&training_data, 0.25, 0.5).0.unwrap();

        info!("Without noise.");
        assert_show(&layer, &[0.0, 0.7, 0.0], true);
        assert_show(&layer, &[0.0, 0.5, 0.0], false);
        assert_show(&layer, &[0.0, 0.2, 0.0], false);

        info!("With noise.");
        assert_show(&layer, &[0.8, 0.7, 0.3], true);
        assert_show(&layer, &[0.3, 0.5, 1.0], true);
        assert_show(&layer, &[0.8, 0.2, 0.2], true);

        timer_end(start);
    }
}
