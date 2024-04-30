use rand::prelude::ThreadRng;
use std::convert::TryFrom;

mod test;

#[derive(Debug, PartialEq)]
pub enum TrainingDataError {
    EmptyData,
    LengthMismatch(usize, usize),
}

#[derive(Default, Debug, PartialEq)]
pub struct TrainingDataMember {
    pub input: Vec<f64>,
    pub output: f64,
}

#[repr(C)]
#[derive(Default, Debug, PartialEq)]
pub struct TrainingData {
    pub inner: Vec<TrainingDataMember>,
    input_length: usize,
}

impl TrainingData {
    pub fn input_length(&self) -> usize {
        self.input_length
    }
}

impl TrainingData {
    /// Adds the result of the function passed as an argument
    /// to all training data input.
    ///
    /// * `input_f` - The input function to add. (Optionally there's a random number generator
    /// available to add, for example, noise.)
    pub fn add_input_fn<F>(&mut self, input_f: F)
    where
        F: Fn(&mut ThreadRng) -> f64,
    {
        let mut rng = rand::thread_rng();

        for i in &mut self.inner {
            for k in &mut i.input {
                *k += input_f(&mut rng);
            }
        }
    }
}

/// Implements a nicer form of instantiation for basic, handwritten data.
impl TryFrom<Vec<(Vec<f64>, f64)>> for TrainingData {
    type Error = TrainingDataError;

    fn try_from(input_pretty: Vec<(Vec<f64>, f64)>) -> Result<Self, Self::Error> {
        let mut temp = TrainingData::default();

        temp.input_length = match input_pretty.get(0) {
            Some(a) => a.0.len(),
            None => return Err(TrainingDataError::EmptyData),
        };

        // Copy all input values into the struct.
        for (index, i) in input_pretty.into_iter().enumerate() {
            if i.0.len() != temp.input_length {
                return Err(TrainingDataError::LengthMismatch(index, i.0.len()));
            }

            temp.inner.push(TrainingDataMember {
                input: i.0,
                output: i.1,
            });
        }

        Ok(temp)
    }
}
