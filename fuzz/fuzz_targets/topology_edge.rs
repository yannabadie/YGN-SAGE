#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let parts: Vec<&str> = s.split('|').collect();
        let edge_type = parts.first().copied().unwrap_or("").to_string();
        let gate = parts.get(1).copied().unwrap_or("").to_string();
        let condition = parts.get(2).map(|x| x.to_string());
        let _ = sage_core::topology::topology_graph::TopologyEdge::try_new(
            edge_type, None, gate, condition, 1.0,
        );
    }
});
