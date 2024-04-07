use log::{debug, info};
use rand::prelude::*;

use colored::Colorize;

mod test;
mod training_data;

use crate::training_data::TrainingData;

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
    fn new(size: usize, threshold: f64, random: bool) -> Layer {
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

    fn output(&self, input: &[f64]) -> (f64, bool) {
        let mut sum: f64 = 0.0;

        for i in 0..input.len() {
            sum += self.weights[i] * input[i];
            debug!(
                "OUT: {:.4} * {:.4} =+ {:.4}",
                self.weights[i], input[i], sum
            );
        }

        (sum, sum > self.threshold)
    }

    fn train(
        &mut self,
        data: &TrainingData,
        learn_strength: f64,
        iterations: i32,
        err_margin: f64,
    ) -> Result<(), LayerError> {
        let mut rised = false;
        let mut falled = false;
        let mut last_err_sum = 0.0;

        for i in 0..iterations {
            let mut err_sum = 0.0;

            for k in 0..data.inner.len() {
                let err = data.inner[k].output - self.output(&data.inner[k].input).0;
                debug!("LRN: epoch: {} err: {:.4}\n", i, err.to_string().red());

                err_sum += err;

                for y in 0..(data.len() - 1) {
                    let delta = learn_strength * data.inner[k].input[y] * err as f64;
                    self.weights[y] += delta;
                }
            }

            // Weird learning spiral???
            if err_sum > last_err_sum {
                rised = true;
            }

            if err_sum < last_err_sum {
                falled = true;
            }

            if rised && falled {
                return Err(LayerError::Oscillating);
            }

            // Normal exit (Nears err_margin)
            if err_sum < err_margin && err_sum > -err_margin {
                debug!("Done learning!");
                return Ok(());
            }

            last_err_sum = err_sum;
        }
        Err(LayerError::OutOfIterations)
    }

    fn pretty_output(&self, input: &[f64]) {
        let result = self.output(input);
        info!("{:?} -> {:6} or {}", input, result.0, result.1);
    }
}

fn main() {
    simple_logger::SimpleLogger::new()
        .env()
        .without_timestamps()
        .init()
        .unwrap();
}
