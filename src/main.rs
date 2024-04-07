use log::{debug, info};
use rand::prelude::*;

mod training_data;

use crate::training_data::TrainingData;

struct Layer {
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
            debug!("{} * {} =+ {}", self.weights[i], input[i], sum);
        }

        (sum, sum > self.threshold)
    }

    fn train(
        &mut self,
        data: &TrainingData,
        learn_strength: f64,
        iterations: i32,
        err_margin: f64,
    ) {
        for i in 0..iterations {
            let mut err_sum = 0.0;

            for k in 0..data.inner.len() {
                let err = data.inner[k].output - self.output(&data.inner[k].input).0;
                debug!("epoch:{} err:{}", i, err);

                err_sum += err;

                for y in 0..(data.len() - 1) {
                    let delta = learn_strength * data.inner[k].input[y] * err as f64;
                    self.weights[y] += delta;
                }
            }

            debug!("---");

            if err_sum < err_margin && err_sum > -err_margin {
                break;
            }
        }
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

    let training_data = TrainingData::from(vec![
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 1.0),
        (vec![1.0, 0.0], 1.0),
        (vec![1.0, 1.0], 0.0),
    ]);

    let mut eye = Layer::new(training_data.len(), 0.5, false);

    eye.train(&training_data, 0.1, 100, 0.1);

    eye.pretty_output(&[0.0, 0.0]);
    eye.pretty_output(&[0.0, 1.0]);
    eye.pretty_output(&[1.0, 0.0]);
    eye.pretty_output(&[1.0, 1.0]);
}
