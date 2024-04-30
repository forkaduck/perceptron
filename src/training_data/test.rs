#[cfg(test)]
mod training_data_tests {
    use crate::training_data::{TrainingData, TrainingDataError};

    #[test]
    fn tryfrom_implementation() {
        // Test well formatted learning data.
        TrainingData::try_from(vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ])
        .unwrap();

        // Test one entry containing more elements than the rest.
        let training_data = TrainingData::try_from(vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ]);

        assert_eq!(training_data, Err(TrainingDataError::LengthMismatch(1, 3)));
    }

    #[test]
    fn add_input_fn_implementation() {
        use rand::prelude::*;

        // Test adding random numbers to the input data.
        let mut training_data =
            TrainingData::try_from(vec![(vec![0.0, 0.0], 0.0), (vec![0.0, 0.0], 0.0)]).unwrap();
        training_data.add_input_fn(|rng| rng.gen::<f64>() / 10.0);

        for i in &training_data.inner {
            for k in &i.input {
                assert_eq!(*k <= 0.1, true);
            }
        }
    }
}
