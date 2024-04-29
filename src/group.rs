use log::info;

use crate::layer::Layer;

pub struct Group {
    layers: Vec<Vec<Layer>>,
}

impl Group {
    pub fn new(depth: u32, inputs: usize, random: bool) -> Self {
        let mut group = Group {
            layers: Vec::with_capacity(depth as usize),
        };

        let mut counter: usize = 0;

        for i in 0..depth {
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
}
