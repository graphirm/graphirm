use crate::ModelAction;
use crate::error::GraphirmError;

pub async fn run(action: ModelAction) -> Result<(), GraphirmError> {
    match action {
        ModelAction::Download => download().await,
    }
}

#[cfg(feature = "local-extraction")]
async fn download() -> Result<(), GraphirmError> {
    println!("Downloading GLiNER2-large-v1 ONNX model (~1.95 GB)...");
    println!("Files will be cached in ~/.cache/huggingface/hub/");
    println!();
    let model_dir = graphirm_agent::knowledge::local_extraction::download_model()
        .await
        .map_err(|e| GraphirmError::Config(e.to_string()))?;
    println!("Download complete.");
    println!();
    println!("Model directory: {}", model_dir.display());
    println!();
    println!("To use the local ONNX extraction backend, set:");
    println!("  export GLINER2_MODEL_DIR=\"{}\"", model_dir.display());
    println!();
    println!("Then restart `graphirm serve`. Extraction will run at");
    println!("~150-200ms per call instead of 25-35s via the LLM API.");
    Ok(())
}

#[cfg(not(feature = "local-extraction"))]
async fn download() -> Result<(), GraphirmError> {
    eprintln!("Error: this binary was not built with local extraction support.");
    eprintln!();
    eprintln!("Rebuild with:");
    eprintln!("  cargo build --release --features local-extraction");
    std::process::exit(1);
}
