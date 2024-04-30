#![allow(dead_code)]
use colored::Colorize;
use log::info;
use std::time::Instant;

use crate::layer::Layer;

/// Sets up logging when its called at the start of a test.
pub fn setup() {
    match simple_logger::SimpleLogger::new()
        .env()
        .without_timestamps()
        .init()
    {
        Ok(()) => {}
        Err(_) => {}
    };

    // The test harness doesn't print "\n" after the test name.
    println!("");
}

/// Helper-function which outputs and checks the layer output.
pub fn assert_show(data: &Layer, input: &[f64], rv: bool) {
    let result = data.output(input);
    info!("{:?} -> {:.6} | {}", input, result.0, result.1);

    assert_eq!(data.output(input).1, rv);
}

/// Helper-function for easier timing because --report-time is to inaccurate (and unstable).
pub fn timer_end(start: Instant) {
    println!(
        "Time elapsed: {}s",
        (Instant::now() - start).as_secs_f64().to_string().green()
    );
}
