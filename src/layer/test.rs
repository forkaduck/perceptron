#[cfg(test)]
mod layer_tests {
    use log::{debug, info};
    use std::time::Instant;

    use crate::layer::{Layer, LayerError, LayerInit};
    use crate::test_helper::*;
    use crate::training_data::TrainingData;

    /// Test that random weights initialization works.
    #[test]
    fn random_init() {
        let layer = Layer::new(4, LayerInit::None);

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
        let mut layer = Layer::new(training_data.input_length(), LayerInit::None);

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
        let mut layer = Layer::new(training_data.input_length(), LayerInit::None);

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

        let mut layer = Layer::new(training_data.input_length(), LayerInit::None);

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

        let mut layer = Layer::new(training_data.input_length(), LayerInit::None);

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

        let mut layer = Layer::new(training_data.input_length(), LayerInit::None);

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

        let mut layer = Layer::new(training_data.input_length(), LayerInit::None);

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

    /// Demonstrates that xor is actually learn-able by multiple layers cooperating.
    #[test]
    fn xor_multilayer() {
        setup();
        let start = Instant::now();

        let mut network = vec![
            vec![
                Layer::new(2, LayerInit::None),
                Layer::new(2, LayerInit::None),
            ],
            vec![Layer::new(2, LayerInit::None)],
        ];

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

        for (index, i) in network[0].iter_mut().enumerate() {
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

        network[1][0]
            .train_optimizer(&second_data, 0.055..0.3, 0.3)
            .0
            .unwrap();

        let test_data = [
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![0.0, 0.0], 0.0),
        ];

        for i in test_data {
            let outputs = [network[0][0].output(&i.0).0, network[0][1].output(&i.0).0];

            let result = network[1][0].output(&outputs);

            info!("{:?} -> {:.6}", i, result.0);
            assert_eq!(result.1, i.1 > 0.5);
        }

        timer_end(start);
    }

    /// Tests a new (to me) learning technique, where two networks are randomly
    /// initialized and then the internal pattern of the weights is reinforced based
    /// on output error.
    ///
    /// By keeping the relative weight differences the same, the learned pattern doesn't change
    /// but instead can be reinforced or weakened.
    #[test]
    fn multi_competitive() {
        use colored::Colorize;
        use rand::prelude::*;

        setup();
        let start = Instant::now();

        const WEIGHTS: usize = 2;

        let test_data = [
            (vec![0.0, 0.0, 0.0, 0.0], 0.0),
            (vec![0.0, 1.0, 0.0, 1.0], 1.0),
            (vec![1.0, 0.0, 1.0, 0.0], 1.0),
            (vec![0.0, 0.0, 0.0, 0.0], 0.0),
        ];

        fn print_group(net: &Vec<Vec<Layer>>) {
            for i in 0..net.len() {
                for k in 0..net[i].len() {
                    let mut temp = String::new();

                    for w in &net[i][k].weights {
                        temp += &format!("{:.2} ", w);
                    }

                    info!("Gr:{} {}", i, temp);
                }
            }
        }

        // Normal improvement.
        // let mut first_rng = rand::rngs::StdRng::seed_from_u64(3000000);
        // let mut second_rng = rand::rngs::StdRng::seed_from_u64(10);

        // Large improvement.
        // let mut first_rng = rand::rngs::StdRng::seed_from_u64(3234);
        // let mut second_rng = rand::rngs::StdRng::seed_from_u64(203874);

        // Weird fluctuations.
        // let mut first_rng = rand::rngs::StdRng::seed_from_u64(323239044);
        // let mut second_rng = rand::rngs::StdRng::seed_from_u64(203872934);

        // Completely random weights.
        let mut first_rng = rand::rngs::StdRng::from_entropy();
        let mut second_rng = rand::rngs::StdRng::from_entropy();

        // Implement two almost identical networks, which only differ in their initial weights.
        let mut first = vec![
            vec![
                Layer::new(WEIGHTS, LayerInit::Seed(&mut first_rng)),
                Layer::new(WEIGHTS, LayerInit::Seed(&mut first_rng)),
            ],
            vec![Layer::new(WEIGHTS, LayerInit::Seed(&mut first_rng))],
        ];

        let mut second = vec![
            vec![
                Layer::new(WEIGHTS, LayerInit::Seed(&mut second_rng)),
                Layer::new(WEIGHTS, LayerInit::Seed(&mut second_rng)),
            ],
            vec![Layer::new(WEIGHTS, LayerInit::Seed(&mut second_rng))],
        ];

        print_group(&second);

        for _ in 0..3 {
            let mut first_err = 0.0;
            let mut second_err = 0.0;

            // Get the output from both and check which one is closer.
            for i in &test_data {
                let mut temp_inputs: Vec<f64> = Vec::from(i.0.clone());
                let mut res_first: Vec<f64> = Vec::with_capacity(i.0.len());

                for i in &first {
                    res_first.clear();

                    for (index, k) in i.into_iter().enumerate() {
                        let offset = index * WEIGHTS;
                        res_first.push(k.output(&temp_inputs[offset..offset + WEIGHTS]).0);
                    }
                    temp_inputs = res_first.clone();
                }

                temp_inputs = Vec::from(i.0.clone());
                let mut res_second: Vec<f64> = Vec::with_capacity(i.0.len());

                for i in &second {
                    res_second.clear();

                    for (index, k) in i.into_iter().enumerate() {
                        let offset = index * WEIGHTS;
                        res_second.push(k.output(&temp_inputs[offset..offset + WEIGHTS]).0);
                    }
                    temp_inputs = res_second.clone();
                }

                first_err += i.1 - res_first[0];
                second_err += i.1 - res_second[0];

                info!(
                    "{:?} -> {:.2} | {:.2} + {} {}",
                    i,
                    res_first[0],
                    res_second[0],
                    format!("{:.2}", first_err).red(),
                    format!("{:.2}", second_err).red()
                );
            }

            for i in 0..first.len() {
                for k in 0..first[i].len() {
                    for w in 0..WEIGHTS {
                        const OFFSET: f64 = 0.5;

                        let mut better = &mut first[i][k].weights[w];
                        let mut worse = &mut second[i][k].weights[w];

                        if first_err > second_err {
                            better = &mut second[i][k].weights[w];
                            worse = &mut first[i][k].weights[w];
                        };

                        // Get the bigger of the two, better or worse.
                        let maximum = better.max(*worse);

                        // Calculate a relative gradient based on the maximum.
                        let gradient = (*better - *worse) / maximum;

                        // Work backwards to get the x coordinate on the linear function.
                        // Then add an offset to strengthen or weaken the pattern.
                        let point_last = *worse / gradient + OFFSET;

                        // Insert the newly adjusted point back into the linear function.
                        let weight_diff = point_last * gradient;

                        debug!(
                            "maximum:{:.2} gradient:{:.2} wΔ:{}",
                            maximum,
                            gradient,
                            format!("{:.2}", weight_diff).green()
                        );

                        debug!(
                            "({}, {}, {}) -> better/worse: {:.2}/{:.2}",
                            i, k, w, better, worse
                        );

                        *worse = 1.0 / (OFFSET + (-*worse * weight_diff).exp2());

                        debug!(
                            "({}, {}, {}) -> better/worse: {:.2}/{:.2}\n",
                            i, k, w, better, worse
                        );
                    }
                }
            }
        }

        print_group(&second);
        timer_end(start);
    }
}
