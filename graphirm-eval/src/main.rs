mod client;
mod harness;
mod task;
mod tasks;

#[tokio::main]
async fn main() {
    let all = tasks::all_tasks();
    println!("Registered tasks: {}", all.len());
    for t in &all {
        println!("  - [{}] {} {:?}", t.id, t.name, t.tags);
    }
}
