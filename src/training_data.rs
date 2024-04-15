use std::convert::TryFrom;

#[derive(Default)]
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

/// Implements a nicer form of instantiation for basic, handwritten data.
impl TryFrom<Vec<(Vec<f64>, f64)>> for TrainingData {
    type Error = ();

    fn try_from(value: Vec<(Vec<f64>, f64)>) -> Result<Self, Self::Error> {
        let mut temp = TrainingData::default();

        temp.length = match value.get(0) {
            Some(l) => l.0.len(),
            None => return Err(()),
        };

        for i in &temp.inner {
            if i.input.len() != temp.length {
                return Err(());
            }
        }

        for i in value {
            temp.inner.push(TrainingDataMember {
                input: i.0,
                output: i.1,
            });
        }

        Ok(temp)
    }
}
