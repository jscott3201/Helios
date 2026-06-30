fn main() {
    println!(
        "{} {}",
        gemma4d_server::CRATE_NAME,
        gemma4d_server::bootstrap_status()
    );
}
