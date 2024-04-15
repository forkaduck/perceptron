use log::{debug, info};
use rand::prelude::*;
use std::ops::Range;

use crate::training_data::TrainingData;
use colored::Colorize;

#[derive(Debug)]
pub enum LayerError {
    OutOfIterations,
    Oscillating,
}

pub struct Layer {
    weights: Vec<f64>,
    threshold: f64,
}

impl Layer {
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

    pub fn output(&self, input: &[f64]) -> (f64, bool) {
        let mut sum: f64 = 0.0;

        for i in 0..input.len() {
            sum += self.weights[i] * input[i];
        }

        (sum, sum > self.threshold)
    }

    pub fn train(
        &mut self,
        data: &TrainingData,
        learn_strength: f64,
        iterations: i32,
        err_margin: f64,
    ) -> Result<(), LayerError> {
        for i in 0..iterations {
            let mut err_sum = 0.0;

            for k in 0..data.inner.len() {
                let err = data.inner[k].output - self.output(&data.inner[k].input).0;
                err_sum += err;

                for y in 0..data.len() {
                    let delta = learn_strength * data.inner[k].input[y] * err as f64;
                    self.weights[y] += delta;
                }

                debug!(
                    "LRN: @ {} -> err: {:.4} err_sum: {:.4}",
                    i,
                    err.to_string().bold(),
                    err_sum.to_string().red(),
                );
            }

            // Normal exit (Nears err_margin)
            if err_sum < err_margin && err_sum > -err_margin {
                debug!("Done learning!");
                return Ok(());
            }
        }
        Err(LayerError::OutOfIterations)
    }
}
