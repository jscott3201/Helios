fn main() {
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    let code = gemma4d_bench::run_cli(std::env::args().skip(1), &mut stdout, &mut stderr);
    std::process::exit(code);
}
