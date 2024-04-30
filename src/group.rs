use log::info;

mod test;

use crate::layer::Layer;

pub struct Group {
    layers: Vec<Vec<Layer>>,
    input_length: usize,
}

impl Group {
    pub fn new(depth: u32, inputs: usize, random: bool) -> Self {
        let mut group = Group {
            layers: Vec::with_capacity(depth as usize),
            input_length: inputs,
        };

        let mut counter: usize = 0;

        for i in (0..depth).rev() {
            let iterations = inputs.pow(i);
            let mut temp: Vec<Layer> = Vec::with_capacity(iterations);

            for _ in 0..iterations {
                temp.push(Layer::new(inputs, random));
                counter += 1;
            }

            group.layers.push(temp);
        }

        info!("Layers allocated: {}", counter);
        group
    }

    pub fn output(&self, input: &[f64]) -> Vec<f64> {
        let mut temp_inputs: Vec<f64> = Vec::from(input);
        let mut temp_outputs: Vec<f64> = Vec::with_capacity(input.len());

        for i in &self.layers {
            temp_outputs.clear();

            for (index, k) in i.into_iter().enumerate() {
                let offset = index * self.input_length;
                temp_outputs.push(k.output(&temp_inputs[offset..offset + self.input_length]).0);
            }
            temp_inputs = temp_outputs.clone();
        }
        temp_inputs
    }
}
