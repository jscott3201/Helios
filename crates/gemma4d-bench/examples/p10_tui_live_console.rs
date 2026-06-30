use std::{
    env, fs,
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use gemma4d_server::http::{ServerConfig, ServerRuntime, serve_listener};
use gemma4d_tui::{provider::HttpProvider, seed_state, write_p10_live_console_walkthrough};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.out_dir)?;

    let server = ServerHarness::spawn()?;
    let server_url = format!("http://{}", server.addr);
    let mut provider = HttpProvider::new(server_url.clone());
    let config_path = PathBuf::from("references/configs/tui.toml");
    let mut state = seed_state(&mut provider, config_path.clone());
    let outcome =
        write_p10_live_console_walkthrough(&mut state, &mut provider, config_path, &args.out_dir)?;
    server.shutdown()?;

    let metrics_path = args.out_dir.join("metrics.json");
    let metrics = fs::read_to_string(&metrics_path)?;
    let metrics_json = serde_json::from_str::<serde_json::Value>(&metrics)?;
    let p95_ok = metrics_json
        .get("render_p95_within_threshold")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    println!("P10 TUI live console: server={server_url}");
    println!("report: {}", args.out_dir.join("tui-report.md").display());
    println!("metrics: {}", metrics_path.display());
    for path in outcome.evidence_paths {
        println!("evidence: {}", path.display());
    }

    if p95_ok {
        Ok(())
    } else {
        Err("P10 render p95 threshold failed".into())
    }
}

#[derive(Debug)]
struct Args {
    out_dir: PathBuf,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut out_dir = PathBuf::from("benchmarks/out/P10-tui-live-console");
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-dir" => {
                    out_dir = args
                        .next()
                        .map(PathBuf::from)
                        .ok_or("--out-dir requires a path")?;
                }
                "-h" | "--help" => {
                    println!(
                        "usage: cargo run -p gemma4d-bench --example p10_tui_live_console -- [--out-dir PATH]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option '{other}'").into()),
            }
        }
        Ok(Self { out_dir })
    }
}

struct ServerHarness {
    addr: std::net::SocketAddr,
    shutdown: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<Result<(), gemma4d_server::http::HttpError>>>,
}

impl ServerHarness {
    fn spawn() -> Result<Self, Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let server_shutdown = Arc::clone(&shutdown);
        let runtime = ServerRuntime::new(ServerConfig::default().with_bind_addr(addr));
        let handle = thread::spawn(move || serve_listener(listener, runtime, server_shutdown));
        Ok(Self {
            addr,
            shutdown,
            handle: Some(handle),
        })
    }

    fn shutdown(mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.addr);
        if let Some(handle) = self.handle.take() {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => return Err(Box::new(error)),
                Err(_) => return Err("server thread panicked".into()),
            }
        }
        Ok(())
    }
}

impl Drop for ServerHarness {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.addr);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
