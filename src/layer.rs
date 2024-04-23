use log::{debug, info};
use rand::prelude::*;
use std::ops::Range;

use crate::training_data::TrainingData;
use colored::Colorize;

#[derive(Debug)]
pub enum LayerError {
    ErrStabilized,
}

pub struct Layer {
    weights: Vec<f64>,
    threshold: f64,
}

impl Layer {
    /// Instantiates a new Layer instance.
    ///
    /// * `size` - How many weights should be initialized.
    /// * `threshold` - The value which decides when a returned value is true/false.
    /// * `random` - An optional parameter to initialize the weights with random numbers.
    pub fn new(size: usize, threshold: f64, random: bool) -> Layer {
        let mut temp = Layer {
            weights: vec![0.5; size],
            threshold,
        };

        if random {
            let mut rng = rand::thread_rng();

            temp.weights.clear();

            //initialize weights
            for _ in 0..size {
                temp.weights.push(rng.gen::<f64>());
            }
        }

        info!("Random Weights: {:?}", temp.weights);

        temp
    }

    /// Calculates a "response" from the "learned" weights.
    ///
    /// * `input` - Input data to respond to.
    pub fn output(&self, input: &[f64]) -> (f64, bool) {
        let mut sum: f64 = 0.0;

        for i in 0..input.len() {
            sum += self.weights[i] * input[i];
        }

        (sum, sum > self.threshold)
    }

    /// Changes the actual weights by the process of iterative learning.
    ///
    /// * `data` - The inputs and outputs the layer should be trained on.
    /// * `learn_strength` - How much the weights change in respect to the learning data.
    /// * `iterations` - How many times the training data should be applied.
    /// * `err_max` - The amount of error considered acceptable.
    pub fn train(
        &mut self,
        data: &TrainingData,
        learn_strength: f64,
        err_max: f64,
    ) -> Result<(), LayerError> {
        let mut err_sum: [f64; 2] = [0.0, f64::MAX];
        let mut counter = 0;

        loop {
            err_sum[0] = 0.0;

            for k in 0..data.inner.len() {
                let err = data.inner[k].output - self.output(&data.inner[k].input).0;
                err_sum[0] += err;

                for y in 0..data.len() {
                    let delta = learn_strength * data.inner[k].input[y] * err;
                    self.weights[y] += delta;
                }
            }
            debug!(
                "LRN: @ {} -> err_sum: {:.4}",
                counter,
                err_sum[0].to_string().red(),
            );

            // Normal exit (err_sum is in range of err_margin)
            if err_sum[0].abs() < err_max {
                return Ok(());
            }

            // Detect if the err_sum is not in range of err_margin and
            // stopped shrinking.
            if (err_sum[0] * 100.0).round() == (err_sum[1] * 100.0).round() {
                return Err(LayerError::ErrStabilized);
            }

            err_sum[1] = err_sum[0];
            counter += 1;
        }
    }

    /// Tries to find the optimal parameters to learn given material.
    pub fn train_optimizer(
        &mut self,
        data: &TrainingData,
        learn_range: Range<f64>,
        err_max: f64,
    ) -> Result<(), LayerError> {
        let mut learn_strength: [f64; 2] = [learn_range.start, learn_range.end];
        let mut err_sum: [f64; 2] = [0.0, f64::MAX];

        loop {
            // Try training with the current learn_strength.
            debug!("Trying learn_strength: {:.6}", learn_strength[0]);

            if let Err(e) = self.train(data, learn_strength[0], err_max) {
                debug!("Learning error: {:?}", e);

                if learn_strength[0] >= learn_range.end {
                    return Err(e);
                }
            }

            // Compare outputs of the layer with the training data.
            err_sum[0] = 0.0;
            for i in &data.inner {
                let result = self.output(&i.input);

                debug!(
                    "Output Test: {:?} -> {:.6} ?= {} | Bo: {} ",
                    &i.input, result.0, i.output, result.1
                );

                err_sum[0] += (result.0 - i.output).abs();
            }
            debug!("Err: {:.6} prev: {:.6}", err_sum[0], err_sum[1]);

            if err_sum[0] > err_sum[1] {
                info!("Found optimum at {:.6}", learn_strength[1]);
                self.train(data, learn_strength[1], err_max)?;
                break Ok(());
            }

            // Add to learn strength by half of previous step.
            {
                let temp = learn_strength[0] + (learn_strength[1] - learn_strength[0]) / 2.0;

                learn_strength[1] = learn_strength[0];
                learn_strength[0] = temp;
            }

            err_sum[1] = err_sum[0];
        }
    }
}
