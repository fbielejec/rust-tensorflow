use std::env;

#[derive(Default, Debug, Clone)]
pub struct Config {
    pub log_level: String,
    pub processor: String , // TODO : enum
    pub batch_size: String, // TODO : uint
    pub data_path: String,
}

pub trait Load {
    // Static method signature; `Self` refers to the implementor type
    fn load() -> Self;
}

impl Load for Config {
    fn load() -> Config {
        Config {
            log_level: get_env_var ("LOG_LEVEL", Some (String::from ("info"))),
            processor: get_env_var ("PROCESSOR", Some (String::from ("CPU"))),
            batch_size: get_env_var ("BATCH_SIZE", Some (String::from ("1000"))),
            data_path: get_env_var ("DATA_PATH", Some (String::from ("resources/e7ff729c-9ac1-44dd-97a0-641f2fcbb1d4.csv"))),
        }
    }
}

fn get_env_var (var : &str, default: Option<String> ) -> String {
    match env::var(var) {
        Ok (v) => v,
        Err (_) => {
            match default {
                None => panic! ("Missing ENV variable: {} not defined in environment", var),
                Some (d) => d
            }
        }
    }
}
