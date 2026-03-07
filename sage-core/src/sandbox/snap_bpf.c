/*
 * STUB — NOT FUNCTIONAL
 *
 * This file is a placeholder for a future kernel-level SnapBPF agent.
 * It compiles but does NOT implement any CoW memory logic.
 * The actual SnapBPF implementation is in Rust: sage-core/src/sandbox/ebpf.rs
 * (userspace CoW via Arc<Vec<u8>> snapshots in DashMap).
 */
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

SEC("kprobe/add_to_page_cache_lru")
int snap_mem_hook(struct pt_regs *ctx) {
    char msg[] = "SnapBPF: stub — no functional logic.\n";
    bpf_trace_printk(msg, sizeof(msg));

    return 0;
}

char _license[] SEC("license") = "GPL";
