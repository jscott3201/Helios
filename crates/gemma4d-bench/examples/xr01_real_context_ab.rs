use gemma4d_bench::xr_ab::{self, RunMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = xr_ab::parse_cli_args(std::env::args().skip(1))?;
    let summary = xr_ab::write_xr01_artifacts(&options)?;

    println!("XR01 real-context A/B harness: {}", summary.decision);
    println!("records: {}", summary.records_path);
    println!("summary: {}", summary.summary_path);
    println!("report: {}", summary.report_path);
    println!("blockers: {}", summary.blockers_path);
    println!("decision: {}", summary.decision_path);

    if matches!(options.mode, RunMode::Real | RunMode::Both)
        && summary.decision == "blocked_with_evidence"
    {
        return Err("XR01 real-run path blocked; see blockers.md".into());
    }

    Ok(())
}
