use core::cmp;
use log::{debug, info};
use rand::prelude::*;

struct TrainingData<'a> {
    pub input: Vec<Vec<f64>>,
    pub output: &'a [f64],
}

impl TrainingData<'_> {
    pub fn new(input: Vec<Vec<f64>>, output: &[f64]) -> TrainingData {
        assert!(input.len() == output.len());
        TrainingData { input, output }
    }

    pub fn data_amount(&self) -> usize {
        cmp::min(self.input.len(), self.output.len())
    }

    pub fn input_size(&self) -> usize {
        self.input[0].len()
    }
}

struct Vision {
    weights: Vec<f64>,
    threshold: f64,
}

impl Vision {
    fn new(size: usize, threshold: f64) -> Vision {
        let mut temp = Vision {
            weights: Vec::new(),
            threshold,
        };

        let mut rng = rand::thread_rng();

        //initialize weights
        for _ in 0..size {
            temp.weights.push(rng.gen::<f64>());
        }

        info!("Random Weights: {:?}", temp.weights);

        temp
    }

    fn output(&self, input: &[f64]) -> f64 {
        let mut sum: f64 = 0.0;

        for i in 0..input.len() {
            sum += self.weights[i] * input[i];
            debug!("{} * {} =+ {}", self.weights[i], input[i], sum);
        }

        sum
    }

    fn decide(&self, sum: f64) -> bool {
        sum > self.threshold
    }

    fn train(&mut self, data: &TrainingData, lrate: f64, epoch: i32, err_margin: f64) {
        for i in 0..epoch {
            let mut err_sum = 0.0;

            for k in 0..data.data_amount() {
                let err = data.output[k] - self.output(&data.input[k]);
                debug!("epoch:{} err:{}", i, err);

                err_sum += err;

                for y in 0..(data.input_size() - 1) {
                    let delta = lrate * data.input[k][y] * err as f64;
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
        info!("{:?} -> {:6} or {}", input, result, self.decide(result));
    }
}

fn main() {
    simple_logger::SimpleLogger::new()
        .env()
        .without_timestamps()
        .init()
        .unwrap();

    let training_data = TrainingData::new(
        vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.7, 0.0],
            vec![0.0, 0.8, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 0.6, 0.0],
            vec![0.0, 0.7, 1.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ],
        &[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    );

    let mut eye = Vision::new(training_data.data_amount(), 0.5);

    eye.train(&training_data, 0.1, 200, 0.01);

    info!("Without noise.");
    eye.pretty_output(&[0.0, 0.7, 0.0]);
    eye.pretty_output(&[0.0, 0.5, 0.0]);
    eye.pretty_output(&[0.0, 0.2, 0.0]);

    info!("With noise.");
    eye.pretty_output(&[0.8, 0.7, 0.3]);
    eye.pretty_output(&[0.3, 0.5, 1.0]);
    eye.pretty_output(&[0.8, 0.2, 0.2]);
}
