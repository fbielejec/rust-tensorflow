mod config;
mod utils;

// #[macro_use]
// extern crate quick_error;
// extern crate polars;

use config::{Config, Load};
use log::{debug, info, error};
// use polars::chunked_array::ChunkedArray;
// use polars::datatypes::{Utf8Type, ArrowDataType};
use polars::prelude::*;
use std::env;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::result::Result;
use std::time::{Instant, Duration as StdDuration};
use utils::print_type_of;

// #[tokio::main]
fn main() -> Result<(), Box<dyn Error>> {

    let config = Config::load();

    if config.processor == "CPU" {
        // ensure CPU is used
        env::set_var ("CUDA_VISIBLE_DEVICES", "");
    }

    // configure logging
    env::set_var("RUST_LOG", &config.log_level);
    env_logger::init();

    info!("Running with config {:?}", config);

    let data = read_csv (&config.data_path)?;
    let scores = calculate_scores (&data);

    debug!("Head of the data:");
    info!("@@@ {:#?}", &scores);

    // data.add_column ();


    info!("Done");
    Ok(())
}



/// calculates the scores as a linear function of predictors
fn calculate_scores (data : &DataFrame) -> Result<DataFrame, PolarsError> {
    let mut scores_column = &(&(&data ["video_slash_normalized_progress"] + &data ["video_slash_liked_qmark_"]) + &data ["video_slash_commented_qmark_"]) / 3f32;
    scores_column.rename("scores");
    DataFrame::new(vec![data ["video_slash_id"].clone (),
                        data ["user_slash_id"].clone (),
                        scores_column])
}

fn read_csv (path : &String) -> Result<DataFrame, PolarsError> {
    let file = File::open(&path).unwrap_or_else(|_| panic!("Failed to open file: {}", path));

    let video_id = Field::new("video_slash_id", DataType::Utf8);
    let user_id = Field::new("user_slash_id", DataType::Utf8);
    let video_progress = Field::new("video_slash_normalized_progress", DataType::Float32);
    let video_is_liked = Field::new("video_slash_liked_qmark_", DataType::Float32);
    let video_is_commented = Field::new("video_slash_commented_qmark_", DataType::Float32);

    let schema : Arc<Schema> = Arc::new (Schema::new(vec![video_id,
                                                          user_id,
                                                          video_progress,
                                                          video_is_liked,
                                                          video_is_commented]));

    CsvReader::new(file)
        .with_schema(schema)
        .has_header(true)
        .finish()
}
