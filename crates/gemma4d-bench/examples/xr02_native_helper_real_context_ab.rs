use gemma4d_bench::xr_ab;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = xr_ab::parse_xr02_cli_args(std::env::args().skip(1))?;
    let summary = xr_ab::write_xr_ab_artifacts(&options)?;

    println!("XR02 native/helper real-context A/B: {}", summary.decision);
    println!("records: {}", summary.records_path);
    println!("summary: {}", summary.summary_path);
    println!("report: {}", summary.report_path);
    println!("blockers: {}", summary.blockers_path);
    println!("decision: {}", summary.decision_path);

    if summary.decision == "blocked_with_evidence" {
        return Err("XR02 benchmark blocked; see blockers.md".into());
    }

    Ok(())
}
