#[cfg(test)]
mod tests {
    use colored::Colorize;
    use log::info;
    use std::time::Instant;

    use crate::layer::Layer;
    use crate::training_data::TrainingData;

    fn setup() {
        match simple_logger::SimpleLogger::new()
            .env()
            .without_timestamps()
            .init()
        {
            Ok(()) => {}
            Err(_) => {}
        };

        // The test harness doesn't print "\n" after the test name.
        println!("");
    }

    // Helper-function which outputs and checks the layer output.
    fn assert_show(data: &Layer, input: &[f64], rv: bool) {
        let result = data.output(input);
        info!("{:?} -> {:.6} | {}", input, result.0, result.1);

        assert_eq!(data.output(input).1, rv);
    }

    // Helper-function for easier timing because --report-time is to inaccurate (and unstable).
    fn timer_end(start: Instant) {
        println!(
            "Time elapsed: {}s",
            (Instant::now() - start).as_secs_f64().to_string().green()
        );
    }

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

    /// Demonstrates that xor is actually learnable if multiple layers cooperate.
    #[test]
    fn multi_layer_xor() {
        setup();
        let start = Instant::now();

        // Train the exclusive nodes first.
        let first_data = [
            TrainingData::try_from(vec![
                (vec![0.0, 0.0], 0.0),
                (vec![0.2, 0.0], 0.0),
                (vec![0.5, 0.0], 1.0),
                (vec![0.7, 0.0], 1.0),
            ])
            .unwrap(),
            TrainingData::try_from(vec![
                (vec![0.0, 0.0], 0.0),
                (vec![0.0, 0.2], 0.0),
                (vec![0.0, 0.5], 1.0),
                (vec![0.0, 0.7], 1.0),
            ])
            .unwrap(),
        ];

        let first_layer = [
            Layer::new(first_data[0].input_length(), false),
            Layer::new(first_data[0].input_length(), false),
        ];

        for (index, mut i) in first_layer.clone().into_iter().enumerate() {
            i.train_optimizer(&first_data[index], 0.055..0.3, 0.3)
                .0
                .unwrap();
        }

        // Train the or node.
        let second_data = TrainingData::try_from(vec![
            (vec![0.0, 0.0], 0.0),
            (vec![1.0, 0.2], 1.0),
            (vec![0.2, 1.0], 1.0),
            (vec![1.0, 1.0], 1.0),
        ])
        .unwrap();

        let mut second_layer = Layer::new(second_data.input_length(), false);

        second_layer
            .train_optimizer(&second_data, 0.055..0.3, 0.3)
            .0
            .unwrap();

        let test_data = [
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([0.0, 0.0], 0.0),
        ];

        for i in test_data {
            let first_output = [first_layer[0].output(&i.0).0, first_layer[1].output(&i.0).0];

            let result = second_layer.output(&first_output);
            info!("{:?} -> {:.6} | {}", i, result.0, result.1);
            assert_eq!(result.1, i.1 > 0.5);
        }

        timer_end(start);
    }
}
