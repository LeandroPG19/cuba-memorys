//! ¿Dónde se va el tiempo? El coste de un veredicto contra la longitud de la premisa.
//!
//! `cargo test --test nli_cost -- --ignored --nocapture`

use cuba_memorys::cognitive::nli;

#[tokio::test]
#[ignore = "medición, no aserción"]
async fn cost_by_premise_length() {
    if !nli::available() || !nli::enabled() {
        eprintln!("SKIP: no hay modelo NLI");
        return;
    }
    let claim = "cuba-memorys está escrito en Java";
    let unit = "cuba-memorys es un servidor MCP de memoria escrito en Rust. ";

    // Calienta la sesión: la primera pasada paga la carga de 1,1 GB.
    let _ = nli::entails(unit, claim).await;

    eprintln!("\n chars | frases |    tiempo | veredicto");
    eprintln!("-------|--------|-----------|-----------");
    for n in [1usize, 2, 4, 8, 16, 32] {
        let premise = unit.repeat(n);
        let t0 = std::time::Instant::now();
        let v = nli::entails(&premise, claim).await.expect("veredicto");
        let dt = t0.elapsed();
        eprintln!(
            "{:6} | {:6} | {:>9?} | {}",
            premise.len(),
            n,
            dt,
            v.label.as_verdict()
        );
    }

    // Y el caso que de verdad importa: 10 evidencias en paralelo, como hace `verify`.
    let premise = unit.repeat(6);
    let t0 = std::time::Instant::now();
    let futs: Vec<_> = (0..10).map(|_| nli::entails(&premise, claim)).collect();
    let _ = futures::future::join_all(futs).await;
    eprintln!(
        "\n10 evidencias concurrentes (como verify): {:?}\n",
        t0.elapsed()
    );
}
