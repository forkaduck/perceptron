pub trait ExtendedStatistics<T> {
    fn avg(&self) -> T;
    fn max(&self) -> f64;
    fn min(&self) -> f64;
    fn dup(&mut self, repeat: usize) -> Vec<T>;
}

impl ExtendedStatistics<f64> for Vec<f64> {
    fn avg(&self) -> f64 {
        self.iter().sum::<f64>() / self.len() as f64
    }

    fn max(&self) -> f64 {
        *self.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
    }

    fn min(&self) -> f64 {
        *self.iter().min_by(|a, b| a.total_cmp(b)).unwrap()
    }

    fn dup(&mut self, repeat: usize) -> Vec<f64> {
        let mut temp = Vec::new();

        for _ in 0..repeat {
            temp.extend_from_slice(self);
        }
        temp
    }
}

