mod config;

#[macro_use]
extern crate quick_error;
extern crate polars;

use config::{Config, Load};
use log::{debug, info, error};
use polars::chunked_array::ChunkedArray;
use polars::datatypes::{Utf8Type, ArrowDataType};
use polars::prelude::*;
use rayon::prelude::*;
use std::env;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::result::Result;
use std::sync::Arc;
use std::time::{Instant, Duration as StdDuration};
use tensorflow::{Graph, Session, SessionOptions, SessionRunArgs, Tensor, SavedModelBundle };
use tokio::{time::delay_for, fs::File as TokioFile, io};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {

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


    info!("@@@ {:?}", data);




    info!("Done");
    Ok(())
}








fn read_csv (path : &String) -> Result<DataFrame, PolarsError> {

    let file = File::open(&path).unwrap_or_else(|_| panic!("Failed to open file: {}", path));

    // enforce schema
    let video_id = Field::new("video_slash_id", ArrowDataType::Utf8, false);
    let user_id = Field::new("user_slash_id", ArrowDataType::Utf8, false);
    let video_progress = Field::new("video_slash_normalized_progress", ArrowDataType::Float32, false);
    let video_is_liked = Field::new("video_slash_liked_qmark_", ArrowDataType::Int16, false);
    let video_is_commented = Field::new("video_slash_commented_qmark_", ArrowDataType::Int16, false);
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
