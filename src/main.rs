mod config;
mod utils;

#[macro_use]
extern crate quick_error;

use config::{Config, Load};
use log::{debug, info};
use polars::chunked_array::ChunkedArray;
use polars::datatypes::Utf8Type;
use polars::prelude::*;
use rayon::prelude::*;
use std::env;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::result::Result;
use std::time::Instant;
use tensorflow::{Graph, Session, SessionOptions, SessionRunArgs, Tensor, SavedModelBundle };
// use utils::pause;

quick_error! {
    #[derive(Debug)]
    enum DataFrameToTensorError {
        PolarsError(err: polars::error::PolarsError) {
            from()
        }
        Status(err: tensorflow::Status) {
            display("Error when transforming DataFrame to Tensorflow representation: {}", err)
        }
    }
}

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
    let scores = calculate_scores (&data)?;
    // TODO: subsample during development
    // let scores = scores.head (Some (1000));

    debug!("Head of the data:");
    debug!("{:#?}", &scores);

    let tik = Instant::now();
    let (scores, user_ids, video_ids) = pivot_data (&scores)?;
    info!("Pivoting data time elapsed: {:.2?}", tik.elapsed());

    debug!("Data in the long format:");
    debug!("{:?}", &scores);

    // pause ();

    let tik = Instant::now();
    let tensor = df_to_tensor (&scores)?;
    info!("DataFrame to Tensor time elapsed: {:.2?}", tik.elapsed());
    info!("Tensor shape {}", tensor.shape ());

    let tik = Instant::now();
    model_run (&config, &tensor, &user_ids, &video_ids)?;
    info!("Total run model time elapsed: {:.2?}", tik.elapsed());

    info!("Done");
    Ok(())
}

fn model_load () -> Result <(Graph, Session), tensorflow::Status> {

    let options = &SessionOptions::new();
    let tags = &["serve"];
    let mut graph = Graph::new();
    let export_dir = Path::new ("./saved_model");

    let model = SavedModelBundle::load (
        options,
        tags,
        &mut graph,
        export_dir
    )?;

    Ok ((graph, model.session))
}

fn model_predict (scores: &Tensor<f32>, user_index: &Tensor::<i32>, graph : &Graph, session : &Session)
                  -> Result <Tensor::<f32>, tensorflow::Status> {

    let mut args = SessionRunArgs::new();
    args.add_feed(&graph.operation_by_name_required("serving_default_scores").unwrap (), 0, &scores);
    args.add_feed(&graph.operation_by_name_required("serving_default_user_index").unwrap (), 0, &user_index);

    let output = args.request_fetch(&graph.operation_by_name_required("StatefulPartitionedCall").unwrap (), 0);

    session.run (&mut args).unwrap ();
    args.fetch(output)
}

fn model_run (config : &Config, scores: &Tensor<f32>, user_ids: &ChunkedArray<Utf8Type>, video_ids: &ChunkedArray<Utf8Type>)
              -> Result<(), Box<dyn Error>> {

    let Config { batch_size, .. } = config;

    let (graph, session) = model_load ()?;

    // SIMD over all user indices in batches of batch_size
    (0..user_ids.len ())
        .collect::<Vec<usize>>()
        .par_chunks(batch_size.parse::<usize> ()?)
        .for_each (| u_indices | {
            debug!("Parallel chunk {:#?}", u_indices);

            u_indices
                .iter ()
                .for_each(| u_index | {

                    let user_index = Tensor::<i32>::new(&[1]).with_values (&[*u_index as i32]).unwrap ();

                    let tik = Instant::now();
                    let result: Tensor::<f32> = model_predict (&scores, &user_index, &graph, &session).unwrap ();
                    let tok = tik.elapsed();

                    let mut recommendations : Vec<String> = Vec::new();
                    result.chunks(2)
                        .take (100) // first 100 results
                        .for_each (|pair| {
                            let video_id = video_ids.get (pair [0] as usize).unwrap ().to_string ();
                            let recommendation_score = pair [1].to_string ();
                            recommendations.push(recommendation_score); recommendations.push (video_id);
                        });

                    let user_id = user_ids.get (*u_index as usize).unwrap ();

                    match recommendations.is_empty () {
                        true => debug!("No recommendations for user/id {:?}, run model time: {:.2?}", &user_id, &tok),
                        false => info!("user/id {:?}, videos: {:?}, run model time: {:.2?}", &user_id, &recommendations, &tok)
                    }

                });
        });

    Ok (())
}

/// from polars df to tensor (columns are video-ids, rows are user-ids)
fn df_to_tensor (df : &DataFrame)
                 -> Result<Tensor<f32>, DataFrameToTensorError> {

    let ( nrow, mut ncol ) = df.shape ();
    ncol -= 1;

    let mut values = vec![0f32; nrow * ncol];
    for (col_idx, series) in df.drop ("user_slash_id")?
        .get_columns()
        .iter()
        .enumerate() {
            series.f32()?
                .into_iter()
                .enumerate()
                .for_each(|(row_idx, opt_v)| {
                    values [ncol * row_idx + col_idx]
                        = match opt_v {
                            Some(v) => v as f32,
                            None => f32::NAN,
                        };
                })
        }

    Tensor::new(&[nrow as u64, ncol as u64]).with_values(&values).map_err(DataFrameToTensorError::Status)
}

/// from long (columnar) to wide (matrix) format
fn pivot_data (data : &DataFrame) -> Result<(DataFrame, ChunkedArray<Utf8Type>, ChunkedArray<Utf8Type>), PolarsError> {
    let scores = data
        .groupby("user_slash_id")?
        .pivot("video_slash_id", "video_slash_scores")
        .first()?;
    let video_ids = scores.get_column_names ().into_iter().skip(1).collect::<ChunkedArray<Utf8Type>>();
    let user_ids = scores ["user_slash_id"].utf8()?;

    Ok ((scores.clone (), user_ids.clone (), video_ids))
}

/// calculates the scores as a linear function of predictors
fn calculate_scores (data : &DataFrame) -> Result<DataFrame, PolarsError> {
    let mut scores_column = &(&(&data ["video_slash_normalized_progress"] + &data ["video_slash_liked_qmark_"]) + &data ["video_slash_commented_qmark_"]) / 3f32;
    scores_column.rename("video_slash_scores");
    DataFrame::new(vec![data ["video_slash_id"].clone (),
                        data ["user_slash_id"].clone (),
                        scores_column])
}

/// reads a csv file into memory, returns a DataFrame
fn read_csv (path : &str) -> Result<DataFrame, PolarsError> {
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
