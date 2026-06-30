use clap::Parser;

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let outcome = gemma4d_tui::run(gemma4d_tui::Cli::parse()).await?;
    println!("{}", outcome.message);
    for path in outcome.evidence_paths {
        println!("evidence: {}", path.display());
    }
    Ok(())
}
