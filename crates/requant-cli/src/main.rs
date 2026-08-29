use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "requant",
    version,
    about = "MoE-aware GGUF requantization + recipe search"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Quantize a GGUF according to a recipe.
    Quantize {
        #[arg(long, short)]
        input: String,
        #[arg(long, short)]
        output: String,
        #[arg(long)]
        recipe: Option<String>,
        #[arg(long)]
        imatrix: Option<String>,
    },
    /// Search for a size/quality-optimal recipe under a budget.
    Search {
        #[arg(long, short)]
        input: String,
        /// Byte budget, e.g. `12G`, `500M`. Optional with --pareto.
        #[arg(long)]
        budget: Option<String>,
        #[arg(long)]
        imatrix: Option<String>,
        #[arg(long)]
        recipe_base: Option<String>,
        #[arg(long, short)]
        output: Option<String>,
        /// Measured sensitivity table from `requant sensitivity` — use real ΔKL instead of the proxy.
        #[arg(long)]
        sensitivity: Option<String>,
        /// Proxy normalization: `rel` (default, matches published numbers) or `abs`
        /// (cross-tensor comparable; the principled choice).
        #[arg(long)]
        proxy_metric: Option<String>,
        /// Candidate ladder: `kquant`, `iquant`, `full`, `blockfloat`, or a comma-separated list of format names.
        #[arg(long)]
        ladder: Option<String>,
        /// Print the full size↔cost Pareto front.
        #[arg(long)]
        pareto: bool,
        /// Quantize with the searched recipe and run a perplexity check vs the source.
        #[arg(long)]
        validate: bool,
        /// Calibration corpus for --validate (passed to llama-perplexity -f).
        #[arg(long)]
        calib: Option<String>,
    },
    /// Measure per-tensor (bits -> ΔKL) sensitivity curves for the search to allocate against.
    Sensitivity {
        #[arg(long, short)]
        input: String,
        /// Calibration corpus.
        #[arg(long)]
        corpus: String,
        /// Where to write the sensitivity table (JSON).
        #[arg(long, short)]
        output: String,
        /// Scratch directory for candidate models.
        #[arg(long)]
        work_dir: Option<String>,
        #[arg(long)]
        imatrix: Option<String>,
        /// `role` (default), `tensor`, or `role-depth:N`.
        #[arg(long)]
        grouping: Option<String>,
        /// Comma-separated precisions to probe.
        #[arg(long)]
        ladder: Option<String>,
        /// Restrict to these role labels (comma-separated).
        #[arg(long)]
        roles: Option<String>,
        /// Refuse to run if the plan needs more than this many candidate models.
        #[arg(long)]
        max_candidates: Option<usize>,
        /// Keep candidate GGUFs after scoring.
        #[arg(long)]
        keep_candidates: bool,
        /// Score from pre-captured logit dumps in this directory instead of running llama.cpp.
        /// Use when no local runtime implements the architecture.
        #[arg(long)]
        logits_dir: Option<String>,
        /// Vocabulary size, when the dumps are bare fp32 rather than RQLG.
        #[arg(long)]
        raw_vocab: Option<usize>,
        /// Extra flag to forward to llama-perplexity (repeatable). Must be identical between the
        /// reference capture and every candidate.
        #[arg(long = "llama-arg")]
        llama_args: Vec<String>,
    },
    /// Import a llama.cpp imatrix file into the content-addressed cache.
    Imatrix {
        #[command(subcommand)]
        sub: ImatrixCmd,
    },
    /// Evaluate a quantized GGUF against a reference via llama-perplexity.
    Eval {
        #[arg(long)]
        quant: String,
        #[arg(long)]
        reference: String,
        #[arg(long)]
        calib: Option<String>,
        /// Report KL divergence instead of perplexity — far more sensitive when comparing recipes.
        #[arg(long)]
        kl: bool,
        #[arg(long = "llama-arg")]
        llama_args: Vec<String>,
    },
    /// Inspect a GGUF: tensor table + MoE detection + role tagging.
    Inspect {
        #[arg(long, short)]
        input: String,
    },
    /// Convert a dense Qwen3.5-family checkpoint (including Qwen3.8) into a trainable Qwen3.5-MoE warm start.
    MoefyQwen38 {
        /// Dense Hugging Face checkpoint directory.
        #[arg(long)]
        input_dir: String,
        /// New checkpoint directory (must not already exist).
        #[arg(long)]
        output_dir: String,
        /// Number of routed experts created in every text layer.
        #[arg(long, default_value_t = 32)]
        experts: usize,
        /// Experts selected per token. The generated router is untrained.
        #[arg(long, default_value_t = 4)]
        top_k: usize,
        /// Maximum output shard size, e.g. 5G or 750M.
        #[arg(long, default_value = "5G")]
        max_shard_size: String,
        /// Validate the conversion plan without writing a checkpoint.
        #[arg(long)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum ImatrixCmd {
    Import {
        #[arg(long)]
        imatrix: String,
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "./.requant")]
        cache_dir: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Quantize {
            input,
            output,
            recipe,
            imatrix,
        } => requant_cli::run_quantize(&input, &output, recipe.as_deref(), imatrix.as_deref()),
        Cmd::Search {
            input,
            budget,
            imatrix,
            recipe_base,
            output,
            sensitivity,
            proxy_metric,
            ladder,
            pareto,
            validate,
            calib,
        } => requant_cli::run_search_opts(&requant_cli::SearchOpts {
            input,
            budget,
            imatrix,
            recipe_base,
            output,
            validate,
            calib,
            sensitivity,
            metric: proxy_metric,
            pareto,
            ladder,
        }),
        Cmd::Sensitivity {
            input,
            corpus,
            output,
            work_dir,
            imatrix,
            grouping,
            ladder,
            roles,
            max_candidates,
            keep_candidates,
            logits_dir,
            raw_vocab,
            llama_args,
        } => requant_cli::run_sensitivity_cmd(&requant_cli::SensitivityOpts {
            input,
            corpus,
            output,
            work_dir,
            imatrix,
            grouping,
            ladder,
            roles,
            max_candidates,
            keep_candidates,
            logits_dir,
            raw_vocab,
            llama_args,
        }),
        Cmd::Imatrix { sub } => match sub {
            ImatrixCmd::Import {
                imatrix,
                model,
                cache_dir,
            } => requant_cli::run_imatrix_import(&imatrix, &model, &cache_dir),
        },
        Cmd::Eval {
            quant,
            reference,
            calib,
            kl,
            llama_args,
        } => requant_cli::run_eval_ex(&quant, &reference, calib.as_deref(), kl, &llama_args),
        Cmd::Inspect { input } => requant_cli::run_inspect(&input),
        Cmd::MoefyQwen38 {
            input_dir,
            output_dir,
            experts,
            top_k,
            max_shard_size,
            dry_run,
        } => requant_cli::run_moefy_qwen38(&requant_cli::MoefyOptions {
            input_dir,
            output_dir,
            experts,
            top_k,
            max_shard_size,
            dry_run,
        }),
    }
}
