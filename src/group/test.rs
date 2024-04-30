#[cfg(test)]
mod group_tests {
    use log::info;
    use std::time::Instant;

    use crate::group::Group;
    use crate::test_helper::*;
    use crate::training_data::TrainingData;

    // Demonstrates that xor is actually learn-able by multiple layers cooperating.
    #[test]
    fn group_basic() {
        setup();
        let start = Instant::now();

        let mut network = Group::new(2, 2, false);

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

        for (index, mut i) in network.layers[0].clone().into_iter().enumerate() {
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

        network.layers[1][0]
            .train_optimizer(&second_data, 0.055..0.3, 0.3)
            .0
            .unwrap();

        let test_data = [
            ([0.0, 0.0, 0.0, 0.0], 0.0),
            ([0.0, 1.0, 0.0, 1.0], 1.0),
            ([1.0, 0.0, 1.0, 0.0], 1.0),
            ([0.0, 0.0, 0.0, 0.0], 0.0),
        ];

        for i in test_data {
            let result = network.output(&i.0);

            info!("{:?} -> {:.6}", i, result[0]);
            assert_eq!(result[0] > 0.5, i.1 > 0.5);
        }

        timer_end(start);
    }
}
