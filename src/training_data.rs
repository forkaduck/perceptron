use std::convert::From;

#[derive(Clone, Default)]
pub struct TrainingDataMember {
    pub input: Vec<f64>,
    pub output: f64,
}

#[repr(C)]
#[derive(Default)]
pub struct TrainingData {
    pub inner: Vec<TrainingDataMember>,
    length: usize,
}

impl TrainingData {
    pub fn len(&self) -> usize {
        self.length
    }
}

impl From<Vec<(Vec<f64>, f64)>> for TrainingData {
    fn from(value: Vec<(Vec<f64>, f64)>) -> Self {
        let mut temp = TrainingData::default();

        //TODO Fix later
        temp.length = match value.get(0) {
            Some(l) => l.0.len(),
            None => return TrainingData::default(),
        };

        for i in value {
            temp.inner.push(TrainingDataMember {
                input: i.0,
                output: i.1,
            });
        }

        temp
    }
}
