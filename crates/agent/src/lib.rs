pub mod compact;
pub mod config;
pub mod context;
pub mod coordinator;
pub mod delegate;
pub mod escalation;
pub mod error;
pub mod event;
pub mod hitl;
pub mod knowledge;
pub mod multi;
pub mod session;
pub mod workflow;

pub use compact::{CompactionConfig, CompactionResult, compact_context, is_compacted};
pub use config::{AgentConfig, AgentMode, Permission};
pub use context::{
    ContextConfig, ContextWindow, EdgeWeights, ScoredNode, build_context, build_subagent_context,
    estimate_tokens, estimate_tokens_str, fit_to_budget, score_node, score_recency,
};
pub use coordinator::Coordinator;
pub use delegate::SubagentTool;
pub use error::AgentError;
pub use event::{AgentEvent, EventBus};
pub use hitl::{is_destructive_tool, HitlDecision, HitlGate};
pub use multi::{
    AgentRegistry, LlmFactory, SubagentHandle, collect_subagent_results, spawn_subagent,
    wait_for_dependencies, wait_for_subagents,
};
pub use session::{Session, SessionMetadata, SessionStatus};
pub use workflow::run_agent_loop;

// Glibc < 2.38 compatibility shim for ort's prebuilt binary.
//
// ort 2.0 ships a prebuilt ONNX Runtime that was compiled against glibc 2.38,
// which introduced C23-compliant `__isoc23_strtoll` / `__isoc23_strtol` /
// `__isoc23_strtoull` symbol names. Ubuntu 22.04 (glibc 2.35) does not have
// them. We provide thin wrappers that forward to the identical old-style
// symbols so the linker is satisfied on older hosts.
//
// This module is a no-op on hosts with glibc >= 2.38 (the linker will prefer
// the real dynamic symbol over our static definition because of link ordering).
#[cfg(all(target_os = "linux", feature = "local-extraction"))]
mod glibc_compat {
    use std::ffi::{c_char, c_int};

    unsafe extern "C" {
        fn strtoll(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> i64;
        fn strtoull(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> u64;
        fn strtol(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> i64;
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn __isoc23_strtoll(
        nptr: *const c_char,
        endptr: *mut *mut c_char,
        base: c_int,
    ) -> i64 {
        unsafe { strtoll(nptr, endptr, base) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn __isoc23_strtoull(
        nptr: *const c_char,
        endptr: *mut *mut c_char,
        base: c_int,
    ) -> u64 {
        unsafe { strtoull(nptr, endptr, base) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn __isoc23_strtol(
        nptr: *const c_char,
        endptr: *mut *mut c_char,
        base: c_int,
    ) -> i64 {
        unsafe { strtol(nptr, endptr, base) }
    }
}
