use rand::prelude::ThreadRng;
use std::convert::TryFrom;

#[derive(Default, Debug)]
pub struct TrainingDataMember {
    pub input: Vec<f64>,
    pub output: f64,
}

#[repr(C)]
#[derive(Default, Debug)]
pub struct TrainingData {
    pub inner: Vec<TrainingDataMember>,
    length: usize,
}

impl TrainingData {
    pub fn len(&self) -> usize {
        self.length
    }
}

impl TrainingData {
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
    type Error = ();

    fn try_from(value: Vec<(Vec<f64>, f64)>) -> Result<Self, Self::Error> {
        let mut temp = TrainingData::default();

        temp.length = match value.get(0) {
            Some(l) => l.0.len(),
            None => return Err(()),
        };

        // Copy all input values into the struct.
        for i in value {
            temp.inner.push(TrainingDataMember {
                input: i.0,
                output: i.1,
            });
        }

        Ok(temp)
    }
}
