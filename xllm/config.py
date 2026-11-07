import math
from dataclasses import dataclass, field, fields
from logging import getLogger
from typing import Optional
from pathlib import Path
from typing import List, Dict
import os
import json

from xllm.configuration import Config
from xllm.distributed import (
    model_parallel_is_initialized,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_context_parallel_rank,
    get_model_parallel_rank
)
from xllm.utils import get_default_training_half


logger = getLogger()


@dataclass
class TokenizerConf(Config):
    # Config for processing different types of data
    type: str = 'llama2'  # Supports llama2/sentencepiece, llama3, or huggingface
    path: Optional[str] = None  # Path to the tokenizer folder
    use_bos: bool = False  # True adds the BOS token at beginning of each sequence
    num_reserved_special_tokens: int = 0  # Reserved tokens used by SentencePiece tokenizer, multiples of 256
    template_dict_json: Optional[str] = None  # Allows different chat templates to be used in runtime for HF tokenizer. The json str should have a template name to template file relative path mapping.

    @property
    def tokenizer_path(self) -> str:
        assert self.path is not None
        return self.path

    def __post_init__(self):
        if self.template_dict_json is not None:
            json.loads(self.template_dict_json)


@dataclass
class DataLoaderConfig(Config):
    buffer_size: int = 512  # The num of tokens in buffer is buffer_size * seq_len
    num_workers: int = 1  # Tokenizer worker threads during buffer refill
    packing_type: str = "simple"  # Sequence packing strategy: simple = concatenation, bestfit = find best sequence to pack into to reduce truncations
    skip_long_docs: bool = False  # True skips overlong non-text docs; otherwise they are split into seq_len chunks
    min_doc_tokens: int = 0  # Skip docs with fewer tokens (incl. BOS/EOS) while filling the buffer; 0 keeps all
    max_consecutive_skips: int = 10000  # Fail a source that repeatedly produces no tokenized sample


@dataclass
class SlurmConf(Config):
    is_slurm_job: bool = False
    global_rank: int = -1  # The rank of current process in entire distributed training job. 0 is the main process
    world_size: int = -1  # The total ranks of the entire distributed training job

    # Slurm parameters for async eval jobs
    partition: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    mem_gb: Optional[int] = None
    time: int = 3 * 24 * 60  # 3 days for default
    timeout_min: int = 3 * 24 * 60  # 3 days for default
    gpus_per_node: int = 8
    cpus_per_task: int = 10
    additional_params: Optional[str] = None

    def set_values(self, is_slurm_job, global_rank, world_size):
        self.is_slurm_job = is_slurm_job
        self.global_rank = global_rank
        self.world_size = world_size

    @property
    def is_master(self) -> bool:
        return self.global_rank == 0

    def __post_init__(self):
        if self.additional_params is not None:
            json.loads(self.additional_params)


ROLES = ("embed", "hidden_matrix", "hidden_vector", "final_norm", "readout")
# Sub-roles: factors on top of their parent role's. hidden_matrix (Muon) by projection, hidden_vector by kind.
SUBROLES = {
    "hidden_matrix": ("attn_in", "attn_out", "ffn_in", "ffn_out"),
    "hidden_vector": ("vec_norm", "vec_conv", "vec_bias"),
}
SUBROLE_PARENT = {sub: parent for parent, subs in SUBROLES.items() for sub in subs}
ALL_ROLES = ROLES + tuple(SUBROLE_PARENT)


@dataclass
class RoleConf(Config):
    """Factors on one role's optimizer hyperparameters (dion_muon). None = 1. mup.* multiplies its own on top."""
    lr: Optional[float] = None  # x optim.lr (AdamW roles) or optim.muon_lr (hidden_matrix)
    eps: Optional[float] = None  # x optim.epsilon; AdamW roles only
    weight_decay: Optional[float] = None  # x optim.weight_decay (Dion decays by lr * weight_decay)

    def factor(self, name: str) -> float:
        value = getattr(self, name)
        return 1.0 if value is None else value


@dataclass
class RolesConf(Config):
    """Per-role factors for dion_muon. Roles, by parameter name: embed = embed.*; readout = output.output.*;
    final_norm = the rest of output.*; hidden_matrix = the Muon matrices in layers.*; hidden_vector = every other
    parameter in layers.* (norm gains, biases, Gekko conv kernels and gates). All unset: the default groups."""
    embed: RoleConf = field(default_factory=RoleConf)
    hidden_matrix: RoleConf = field(default_factory=RoleConf)
    hidden_vector: RoleConf = field(default_factory=RoleConf)
    final_norm: RoleConf = field(default_factory=RoleConf)
    readout: RoleConf = field(default_factory=RoleConf)
    # Sub-roles, multiplied onto the parent's factor. hidden_matrix: attn_in = wq / wk / wv / wr / wg (attention
    # inputs and gates), attn_out = wo, ffn_in = fc1 / fc3, ffn_out = fc2. hidden_vector: vec_norm = norm gains and
    # biases (timenorm, query / key / attn norms, FFN norm), vec_conv = Gekko depthwise conv kernels, vec_bias =
    # projection biases. Setting any sub-role of a parent splits that parent into one group per sub-role.
    attn_in: RoleConf = field(default_factory=RoleConf)
    attn_out: RoleConf = field(default_factory=RoleConf)
    ffn_in: RoleConf = field(default_factory=RoleConf)
    ffn_out: RoleConf = field(default_factory=RoleConf)
    vec_norm: RoleConf = field(default_factory=RoleConf)
    vec_conv: RoleConf = field(default_factory=RoleConf)
    vec_bias: RoleConf = field(default_factory=RoleConf)

    def __post_init__(self):
        for role in ALL_ROLES:
            for name in ("lr", "eps", "weight_decay"):
                value = getattr(getattr(self, role), name)
                if value is not None and not (math.isfinite(value) and value >= 0 and (name != "eps" or value > 0)):
                    raise ValueError(f"optim.roles.{role}.{name} must be finite and >= 0 (eps > 0), got {value}")
        for role in ("hidden_matrix",) + SUBROLES["hidden_matrix"]:
            if getattr(self, role).eps is not None:
                raise ValueError(f"optim.roles.{role}.eps: Muon has no eps")

    def is_set(self) -> bool:
        return any(getattr(getattr(self, role), name) is not None
                   for role in ALL_ROLES for name in ("lr", "eps", "weight_decay"))

    def split(self, parent: str) -> bool:
        """Whether any sub-role of parent is set (parent then gets one group per sub-role)."""
        return any(getattr(getattr(self, sub), name) is not None
                   for sub in SUBROLES[parent] for name in ("lr", "eps", "weight_decay"))


@dataclass
class OptimConf(Config):
    optimizer: str = 'adamw'  # adamw or dion_muon (requires FSDP2)
    lr: float = 1e-3  # AdamW learning rate after warmup
    muon_lr: float = 0.01  # Muon learning rate, before matrix-size adjustment
    muon_momentum: float = 0.95
    muon_nesterov: bool = False
    muon_adjust_lr: str = 'spectral_norm'  # spectral_norm or rms_norm
    muon_orthogonalizer: str = 'newton_schulz5'  # newton_schulz5 or polar_express
    muon_use_gram_newton_schulz: bool = False  # Optional Dion GNS backend; overrides muon_orthogonalizer
    muon_use_triton: bool = False  # Use Dion's accelerated orthogonalization kernels
    warmup: int = 2000  # Num of steps for LR warmup. 0 means no warmup
    weight_decay: float = 0.1  # Decoupled weight decay
    disable_wd_emb: bool = False  # Disable weight decay on the input embedding weights (adamw and dion_muon)
    disable_wd_out: bool = False  # Disable weight decay on the output (readout) projection weights (adamw and dion_muon)
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.98
    clip: float = 1.0  # Global gradient L2 norm clipping threshold, before applying loss scale.
    scheduler: str = "cosine"  # LR scheduler: linear, cosine, constant, or wsd (warmup-stable-decay)
    lr_init_ratio: float = 1e-4  # The initial LR = lr_init_ratio * lr
    lr_end_ratio: float = 0.01  # The ending LR = lr_end_ratio * lr. Also the end value if using a cosine scheduler
    cycles: float = 1.0  # Cycles of half period of cosine scheduler. 1 means decaying from peak to lr_end_ratio
    wsd_decay_duration_steps: Optional[int] = None  # Number of steps in the WSD decay phase. Decay starts at total_steps - wsd_decay_duration_steps.
    wsd_decay_style: str = 'cosine'  # The curve type of decay phase in WSD scheduler: cosine or linear
    roles: RolesConf = field(default_factory=RolesConf)  # Per-role lr / eps / weight decay factors (dion_muon)

    def role_factor(self, role: str, name: str, subrole: Optional[str] = None) -> float:
        """optim.roles.<role>.<name> (x optim.roles.<subrole>.<name>) times the mup.* factor of role (set by
        TrainerConf, not serialized)."""
        mup = getattr(self, "_mup_role_factors", {})
        factor = getattr(self.roles, role).factor(name) * mup.get(role, {}).get(name, 1.0)
        if subrole is not None:
            assert SUBROLE_PARENT[subrole] == role, (subrole, role)
            factor *= getattr(self.roles, subrole).factor(name)
        return factor

    def role_groups(self) -> bool:
        """Whether dion_muon steps one group per role: some role factor is set or mup.* is on."""
        return self.roles.is_set() or bool(getattr(self, "_mup_role_factors", {}))

    def __post_init__(self):
        if self.roles.is_set() and self.optimizer != 'dion_muon':
            raise ValueError('optim.roles.* require optim.optimizer=dion_muon')
        if self.optimizer not in ['adamw', 'dion_muon']:
            raise ValueError(f'Unknown optimizer: {self.optimizer}')
        if self.optimizer == 'dion_muon':
            import math
            for name in ('lr', 'muon_lr', 'epsilon'):
                value = getattr(self, name)
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{name} must be finite and positive')
            if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
                raise ValueError('weight_decay must be finite and nonnegative')
            for name in ('muon_momentum', 'beta1', 'beta2'):
                if not 0 <= getattr(self, name) < 1:
                    raise ValueError(f'{name} must be in [0, 1)')
            if self.muon_adjust_lr not in ['spectral_norm', 'rms_norm']:
                raise ValueError('muon_adjust_lr must be spectral_norm or rms_norm')
            if self.muon_orthogonalizer not in ['newton_schulz5', 'polar_express']:
                raise ValueError('Unknown muon_orthogonalizer')
        assert self.scheduler in ['linear', 'cosine', 'constant', 'wsd']
        assert self.warmup >= 0
        if self.scheduler == 'wsd':
            assert self.wsd_decay_duration_steps is not None, 'decay duration steps is required for WSD.'


@dataclass
class ModelConf(Config):
    arch: str = "transformer"  # Architecture: transformer or gekko
    ddp_backend: str = 'fsdp2'  # Distributed parameter sharding implementation: fsdp1 or fsdp2
    num_layers: int = 8
    full_attention_interval: int = 0  # Gekko: 0 = unchanged; 4 = full/Gekko/Gekko/Gekko, starting with full attention
    model_dim: int = 1024  # Dimensions of token hidden state and residual stream
    # Attention
    chunk_size: int = 2048  # Used for local attention, memory update in Gekko and prefilling.
    num_heads: int = 1  # Number of Query attention heads
    num_kv_heads: Optional[int] = None  # Number of Key/Value heads. None means = num_heads. Uses GQA when this is small.
    head_dim: Optional[int] = None  # Dimension of each Q/K head. None means = model_dim // num_heads
    v_head_dim: Optional[int] = None  # Dimension of each V head in Gekko. None means = head_dim
    qknorm: bool = False  # Whether to apply RMSNorm to Q/K heads in Transformer
    attn_act_func: str = 'softmax'  # softmax or softdelta
    apply_attn_gate: bool = False  # Whether to apply gating to attention output in Transformer
    attn_gate_func: str = 'softplus'  # silu or softplus
    causal_attn_backend: Optional[str] = None  # Attention operator backend: None = regular implementation, or choose between swift/flash/xattn
    sca_backend: str = 'swift'  # Sliding Chunk Attention (SCA) operator backend (swift/flash/xattn).
    apply_bias_term: bool = False  # Whether the Gekko attention output gating projection W_r includes a bias term.
    ffn_hidden_dim: int = 2048  # Hidden dimension in dense FFN. MoE expert dimensions is set by expert_inter_dim.
    # Adaptive Working Memory (AWM)
    awm_orthogonal_update: bool = False  # Whether to apply orthogonal update when updating the memory in AWM
    # MoE & MoVA
    num_values: int = 0  # Num of value projections available for routing to in MoVA. 0 means MoVA off. Requires using MoE as well when using MoVA.
    num_activated_values: int = 0  # Num of activated value projections per token (top-k) in MoVA
    mova_backend: str = 'sequential'  # sequential, nested, torch, or te
    num_experts: int = 0  # Num of experts to be routed in each MoE layer (not counting shared experts). 0 means using dense FFN
    num_activated_experts: int = 0  # Num of activated experts per token (top-k)
    num_shared_experts: int = 0  # Num of shared experts for each token, in addition to routed experts
    num_dense_layers: Optional[int] = None  # Number of initial dense layers. If None, defaults to 0 when MoE is enabled, otherwise to all layers.
    expert_inter_dim: int = 0  # Hidden dimension of the FFN in a single expert.
    moe_router_score_func: str = 'sigmoid'  # sigmoid or softmax
    moe_router_bias: bool = False  # Whether to include the router bias, only affects top-k selection.
    moe_router_bias_update_rate: Optional[float] = None  # Update rate for the router bias during backprop
    moe_router_scaling_factor: Optional[float] = None  # The scaling factor for outputs from selected experts. None means no scaling
    moe_expert_backend: str = 'sequential'  # sequential, nested, torch, or te
    moe_permutation_backend: str = 'torch'  # Token permutation and unpermutation backend: torch or triton
    # Causal conv1d
    causal_conv_width: int = 4  # Width of Q/K/V causal convolution kernel in Gekko. Currently supports 1-4 tokens
    causal_conv_backend: str = 'triton'  # triton or fla
    causal_conv_weight_normalization: bool = True  # Whether to apply softmax normalization for causal convolution weights
    # Initialization
    init_mode: str = 'he'  # he, gaussian, xavier, or none (skipping init)
    init_std: Optional[float] = None  # Init std for gaussian; If None, deriving from input dimensions. he/xavier requires None
    init_embed_std: Optional[float] = None  # Std used to truncate normal init of embedding weights. If None, = 1 / sqrt(model_dim)
    init_logits_std: Optional[float] = None  # Std used to truncate normal init of output projection weights. If None, = 1 / sqrt(model_dim)
    # Input & output
    vocab_size: int = -1  # Defined later by tokenizer
    # Normalization
    timenorm_num_groups: int = 32  # Num of groups of features used in timestep decay normalization in Gekko.
    timenorm_beta1: float = 0.999  # Used in timestep decay normalization.
    timenorm_beta2: float = 0.9999  # Used in timestep decay normalization.
    timenorm_backend: str = 'cub'  # cub or chunkwise
    layernorm_num_groups: int = 1  # Num of groups of features in LayerNorm/RMSNorm.
    memory_efficient_norm: bool = False  # Whether to use norm output for backward to save activation memory. Unsupported by 'cub' timestep norm backend.
    norm_affine: bool = True  # Enable learnable scaling in attention/FFN input norms (plus bias for LayerNorm).
    timenorm_eps: float = 1e-5  # Used in timestep decay normalization.
    layernorm_eps: float = 1e-5
    rmsnorm_eps: float = 1e-6
    apply_rmsnorm: bool = False  # True = RMSNorm, False = LayerNorm
    residual_func: str = "base"  # Residual connection type: base/add means standard addition, ortho for orthogonal residuals.
    residual_heads: Optional[int] = None  # Num of feature groups for orthogonal residuals. Required when using residual_func='ortho'.
    residual_multiplier: float = 1.0  # Multiplier on every attention / FFN branch output before the residual add.
    branch_scale: float = 1.0  # CompleteP depth factor on the branches, (base_num_layers / num_layers)^alpha; set by mup.*, leave at 1
    embed_multiplier: float = 1.0  # Multiplier on the embedding output.
    # Rope base
    rope_base: float = 100000  # RoPE frequency base
    rope_head_dim: Optional[int] = None  # RoPE dimensions per Q/K head. None means full head_dim, 0 disables RoPE.
    # Dropout rates
    dropout: float = 0.0  # Dropout rate for embeddings and for sublayer outputs before residual connections.
    hidden_dropout: float = 0.0  # Dropout rate for attention activations before output projection and for FFN intermediate activations.
    attention_dropout: float = 0.0  # Dropout rate for attention weights after softmax.

    swiglu: bool = False  # Whether to use SwiGLU in dense FFN. Otherwise uses SiLU.
    scale_emb: bool = False  # Whether to scale embedding output by sqrt(model_dim).
    layerwise_ckpt: bool = False  # Whether to enable activation checkpointing for each layer to save memory by recomputing forward during backprop.
    layerwise_ckpt_layers: Optional[int] = None  # With layerwise_ckpt, checkpoint only the first N layers (None = all). Trades memory for less recompute.

    fused_block: bool = False  # Whether to use custom fused forward/backward implementations for training blocks; incompatible with layerwise_ckpt.
    # Recomputation switches
    recompute_q: bool = False  # Whether to recompute Q activations in fused block during backprop to save activation memory.
    recompute_v: bool = False  # Whether to recompute K/V activations in fused block.
    recompute_attention: bool = False  # Whether to recompute attention outputs in fused block. Applies to sliding chunk attention in Gekko.
    recompute_awk: bool = False  # Whether to recompute AWM outputs in fused block in Gekko.
    recompute_fc1_out: bool = False  # Whether to recompute fc1 outputs in dense FFNs or shared experts in fused block.
    recompute_fc3_out: bool = False  # Whether to recompute fc3 outputs in dense SwiGLU FFNs or shared experts in fused block.
    recompute_router: bool = False  # Whether to recompute router scores for MoE/MoVA in fused block.

    recompute_logits: bool = False  # Whether to recompute logits in output layer. Independent of fused_block. True = recompute logits and compute grad_X/grad_W during backward, False = precompute grad_X/grad_W during forward.

    def __post_init__(self):
        assert self.arch in ['transformer', 'gekko']
        if isinstance(self.full_attention_interval, bool) or not isinstance(self.full_attention_interval, int):
            raise ValueError('full_attention_interval must be an integer')
        if self.full_attention_interval:
            if self.arch != 'gekko' or self.full_attention_interval < 2:
                raise ValueError('full_attention_interval requires arch=gekko and an interval >= 2')
            if self.num_experts or self.num_values:
                raise ValueError('Hybrid Gekko currently requires dense FFNs')
            head_dim = self.model_dim // self.num_heads if self.head_dim is None else self.head_dim
            if self.v_head_dim is not None and self.v_head_dim != head_dim:
                raise ValueError('Hybrid full attention requires v_head_dim=head_dim')
            if self.fused_block and self.causal_attn_backend not in ('flash', 'xattn'):
                raise ValueError('Fused hybrid full attention requires flash or xattn')
        assert self.ddp_backend in ['fsdp1', 'fsdp2']

        assert self.causal_attn_backend in [None, "swift", "flash", 'xattn']
        assert self.sca_backend in ["swift", "flash", 'xattn']
        assert self.mova_backend in ['sequential', 'nested', 'torch', 'te']
        assert self.moe_expert_backend in ['sequential', 'nested', 'torch', 'te']
        assert self.moe_permutation_backend in ['torch', 'triton']
        assert self.timenorm_backend in ['cub', 'chunkwise']
        assert self.causal_conv_backend in ['fla', 'triton']

        if self.fused_block:
            assert not self.layerwise_ckpt, 'disable layer-wise checkpointing when using fused block'

        if self.num_kv_heads is not None:
            assert self.num_heads % self.num_kv_heads == 0

        assert self.attn_act_func in ['softmax', 'softdelta']
        assert self.attn_gate_func in ['silu', 'softplus']
        assert self.residual_func in ['base', 'add', 'ortho']
        for name in ('residual_multiplier', 'branch_scale', 'embed_multiplier'):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f'model.{name} must be finite and positive, got {value}')
        if self.residual_multiplier * self.branch_scale != 1.0:
            if self.residual_func not in ['base', 'add']:
                raise ValueError('A residual branch multiplier needs residual_func base or add')
            if self.fused_block:
                raise ValueError('A residual branch multiplier is not implemented in the fused blocks')
            if self.num_experts or self.num_values:
                raise ValueError('A residual branch multiplier is only wired into dense blocks')

        if self.num_values > 0:
            assert 0 < self.num_activated_values <= self.num_values
            assert self.num_experts > 0, "MoVA requires MoE."

        if self.num_experts > 0:
            assert 0 < self.num_activated_experts <= self.num_experts
            assert self.num_dense_layers is None or self.num_dense_layers <= self.num_layers
            assert self.expert_inter_dim > 0
            assert self.moe_router_score_func in ['softmax', 'sigmoid']
            if self.moe_router_bias:
                assert self.moe_router_bias_update_rate > 0.0, f"Invalid MoE router bias update rate: {self.moe_router_bias_update_rate}"
        else:
            assert self.num_dense_layers is None

        assert self.timenorm_num_groups <= 256

        assert 0 <= self.dropout < 1
        assert 0 <= self.attention_dropout < 1
        assert 0 <= self.hidden_dropout < 1

        if self.arch == 'transformer':
            if self.fused_block:
                assert self.causal_attn_backend is not None, 'requiring efficient attention when using fused block'


@dataclass
class ValidConf(Config):
    # Tasks to evaluate
    ppl_file_list: str = ""  # Comma separated list of files to eval PPL
    task_root: str = ""  # Root directory of tasks
    task_list: str = ""  # Comma separated list of tasks

    batch_size: int = 32  # Eval batch size per DP process.
    seq_len: int = 8192  # Input seq_len in tokens for PPL eval.
    n_batches: int = -1  # Max batches per PPL file per DP process used for PPL eval (<= 0 for full evaluation).

    # Decoding parameters
    use_sampling: bool = False  # Whether to sample during generations. False uses argmax greedy decoding.
    temperature: float = 1.0  # Divide logits by this value before sampling. Must be positive when sampling.
    top_k: int = 0  # Restrict sampling to the top-k tokens. 0 disables it.
    top_p: float = 0.0  # Cumulative prob threshold nucleus sampling. 0 disables it. >0 takes precedence over top_k.

    save_eval: bool = False  # Whether to save per-sample results to the eval output dir.

    @property
    def ppl_files(self) -> List[str]:
        paths = [path for path in self.ppl_file_list.split(",") if len(path) > 0]
        return paths

    @property
    def tasks(self) -> List[str]:
        tasks = [task for task in self.task_list.split(",") if len(task) > 0]
        assert len(tasks) == len(set(tasks))
        if "mmlu" in set(tasks):
            from xllm.eval.task.mmlu import MMLUTask
            tasks = [t for t in tasks if t != "mmlu"] + list(MMLUTask.mmlu_tasks.keys())
        
        if "arabic_mmlu" in set(tasks):
            from xllm.eval.task.arabic_mmlu import ArabicMMLUTask
            tasks = [t for t in tasks if t != "arabic_mmlu"] + list(ArabicMMLUTask.arabic_mmlu_tasks.keys())

        if "ruler" in set(tasks):
            from xllm.eval.task.ruler import RulerTask
            tasks = [t for t in tasks if t != "ruler"] + list(
                RulerTask.get_sub_tasks.keys()
            )
        return tasks

    def should_do_eval(self):
        return len(self.ppl_files) > 0 or len(self.tasks) > 0

    def __post_init__(self):
        # decoding params
        assert self.temperature >= 0
        assert self.top_k >= 0
        assert 0 <= self.top_p < 1

        # ppl files
        if len(self.ppl_files) > 0:
            assert len(self.ppl_files) == len(set(self.ppl_files))
            assert all(path.endswith(".jsonl") for path in self.ppl_files)
            assert all(Path(path).is_file() for path in self.ppl_files), self.ppl_files

        # tasks root dir
        if len(self.task_root) > 0:
            assert os.path.isdir(self.task_root), self.task_root

        # task dirs
        for task in self.tasks:
            task_dir = os.path.join(self.task_root, task)
            assert os.path.isdir(task_dir), task_dir


@dataclass
class MuPConf(Config):
    """muP (width) + CompleteP (depth) for dion_muon, relative to a base model (arXiv:2512.22382; Aditya's
    ifm-ai/xllm#30 table, with a real branch multiplier instead of folding). m_N = model_dim / base_model_dim,
    m_L = num_layers / base_num_layers, c = m_L^-alpha. Keep head_dim fixed; scale heads, kv heads and ffn with
    model_dim. Factors, on top of optim.* and optim.roles.*:

        role           lr              eps        weight_decay   init std
        embed          1               1/m_N      1              base_init_embed_std (all widths)
        hidden_matrix  m_L^(alpha-1)   -          1              he / gaussian: 1/sqrt(fan_in), already ~ m_N^-1/2
        hidden_vector  m_L^(alpha-1)   c/m_N      1              -
        final_norm     1               1          1              -
        readout        1/m_N           1          m_N            base_init_unembed_std / m_N
        branches: x c (model.branch_scale); Muon needs spectral_norm, which transfers across width by itself.

    Unset stds default to 1/sqrt(base_model_dim), xLLM's gaussian default at the base width, so a run at the base
    shape is the run without mup.
    """
    base_model_dim: int = 0  # model_dim the hyperparameters were tuned at
    base_num_layers: int = 0  # num_layers the hyperparameters were tuned at
    alpha: float = 1.0  # CompleteP depth exponent in [0.5, 1]; c = (base_num_layers / num_layers)^alpha
    base_init_embed_std: Optional[float] = None  # embedding init std, all widths
    base_init_unembed_std: Optional[float] = None  # readout init std at the base width; x 1/m_N

    def multipliers(self, model: "ModelConf"):
        m_n = model.model_dim / self.base_model_dim
        m_l = model.num_layers / self.base_num_layers
        return m_n, m_l, m_l ** -self.alpha

    def role_factors(self, model: "ModelConf"):
        m_n, m_l, c = self.multipliers(model)
        depth_lr = m_l ** (self.alpha - 1)
        return {
            "embed": {"eps": 1 / m_n},
            "hidden_matrix": {"lr": depth_lr},
            "hidden_vector": {"lr": depth_lr, "eps": c / m_n},
            "final_norm": {},
            "readout": {"lr": 1 / m_n, "weight_decay": m_n},
        }

    def resolve(self, model: "ModelConf", optim: "OptimConf"):
        """Check the run; set model.branch_scale and the embedding / readout init stds; hand optim the factors.
        Assignments only, so a resumed config (holding the resolved values) resolves to the same run."""
        if self.base_model_dim <= 0 or self.base_num_layers <= 0:
            raise ValueError('mup.base_model_dim and mup.base_num_layers must be positive')
        if not 0.5 <= self.alpha <= 1.0:
            raise ValueError(f'mup.alpha must be in [0.5, 1], got {self.alpha}')
        for name, actual, supported in (
            ('optim.optimizer', optim.optimizer, ['dion_muon']),
            ('optim.muon_adjust_lr', optim.muon_adjust_lr, ['spectral_norm']),
            ('model.arch', model.arch, ['transformer', 'gekko']),
            ('model.init_mode', model.init_mode, ['he', 'gaussian']),
            ('model.init_std', model.init_std, [None]),  # hidden init must stay ~ 1/sqrt(fan_in)
            ('model.scale_emb', model.scale_emb, [False]),
            ('model.residual_func', model.residual_func, ['base', 'add']),
            ('model.fused_block', model.fused_block, [False]),
            ('model.num_experts', model.num_experts, [0]),
            ('model.num_values', model.num_values, [0]),
        ):
            if actual not in supported:
                raise ValueError(f'mup requires {name} in {supported}, got {actual!r}')
        m_n, _, c = self.multipliers(model)
        base_std = 1 / math.sqrt(self.base_model_dim)
        embed_std = self.base_init_embed_std if self.base_init_embed_std is not None else base_std
        unembed_std = (self.base_init_unembed_std if self.base_init_unembed_std is not None else base_std) / m_n
        for name, value in (('branch_scale', c), ('init_embed_std', embed_std), ('init_logits_std', unembed_std)):
            current = getattr(model, name)
            default = 1.0 if name == 'branch_scale' else None
            if current != default and not math.isclose(current, value, rel_tol=1e-12):
                raise ValueError(f'model.{name}={current} conflicts with mup, which sets it to {value}')
            setattr(model, name, value)
        optim._mup_role_factors = self.role_factors(model)


@dataclass
class TrainerConf(Config):
    slurm: Optional[SlurmConf]
    model: ModelConf
    optim: OptimConf
    valid: ValidConf
    tokenizer: TokenizerConf
    dataloader: DataLoaderConfig  # Configs for multi-source data loading.
    mup: Optional[MuPConf] = None  # muP width + CompleteP depth transfer (dion_muon), relative to a base model

    # Training data / iterator
    data: str = ""  # Comma seperated data mix. Each mix is formatted as path:weight:json_key:source_format. Weights need not sum to 1.

    # Output dir
    dump_dir: str = ""  # Root dir for training logs and checkpoints. Also searched for checkpoints to resume training.

    # Base model dir
    base_model_dir: Optional[str] = None  # Initializes training when dump_dir has no checkpoints to resume. Useful for a new training stage.
    reload_base_optim_state: bool = False  # Whether to also load optimizer state from the base model checkpoint.

    # Parallelism
    model_parallel_size: int = 1  # Num of model/tensor parallel.
    context_parallel_size: int = 1  # Num of context parallel. Splits sequences along the token dimension.
    fully_sharded_size: Optional[int] = None  # Num of FSDP shard group. If None, = DP.

    # Logging
    disable_workers_print: bool = False  # Whether to disable logging from non-main training processes (workers).
    disable_logging: bool = False  # Whether to disable metric logging via MetricLogger, including JSONL, TensorBoard, and W&B.
    health_metrics: bool = False
    health_log_freq: int = 100
    health_query_key_stats: bool = False
    health_weight_norms: bool = False
    health_gradient_norms: bool = False
    health_logits_stats: bool = False
    health_gekko_stats: bool = False
    health_moe_stats: bool = False
    health_mova_stats: bool = False

    # Training hyperparameters
    batch_size: Optional[int] = None  # Num of sequences per DP rank per step. Set either this or global_batch_size.
    global_batch_size: Optional[int] = None  # Total batch size across DP ranks. Must be divisible by DP; batch_size is derived from this.
    seq_len: int = 8192  # Num of tokens in each sequence. Must be divisible by context_parallel_size * model.chunk_size.
    multi_segments: bool = True  # Whether to segment docs using BOS token. This will prevent attention across document boundaries.
    deterministic: bool = False  # Whether to request deterministic computations when supported.
    attn_stability_control: int = 0 # The high precision level for attention (only supported in xattn backend).

    steps: int = 100000  # Total number of training steps.
    dtype: str = get_default_training_half()  # Training compute dtype: bf16 (default), fp16, or fp32.
    loss_rescaling: bool = False  # Whether to enable dynamic loss scaling and overflow handling. Required for fp16.

    # MoE hyps
    moe_router_load_balancing_type: Optional[str] = None  # Router load balancing aux loss: dot or entropy. None disables it.
    moe_aux_loss_coeff: float = 0.0  # Coeff of router load balancing aux loss in the total training loss.
    moe_z_loss_coeff: float = 0.0  # Coeff for router z-loss.

    # FSDP parameters
    fp32_reduce_scatter: bool = True  # Whether to use fp32 for FSDP gradient reduction. Otherwise uses the training compute dtype.
    reshard_after_forward: bool = True  # Whether to release unsharded parameters after forward to save memory. Needs to gather them again for backward.
    forward_prefetch: bool = False  # Whether to prefetch parameters for better overlapping. Not supported by the current FSDP2 implementation.

    log_freq: int = 10  # Logs training metric summaries every N steps.
    dump_freq: int = 1000  # Saves checkpoint every N steps. Also save at the final step.
    eval_freq: int = 1000  # Evaluates every N steps. <= 0 disables it.
    gc_collect_freq: int = 1000  # Calls gc.collect() every N steps.
    sync_check_freq: int = 10000  # Interval for checking batch consistency within MP/CP groups before sequence splitting; <= 0 disables it.

    keep_n_last_checkpoints: int = 2  # Keeps the latest N checkpoints plus any retained eval checkpoints; -1 keeps all.
    keep_eval_checkpoints: bool = True  # Whether to additionally retain checkpoints used for evals.
    dcp_for_optimizer: bool = False  # Whether to use DCP optimizer state handling and loading. Also selects DCP for synchronous saves.
    async_checkpointing: bool = False  # Whether to save model and optimizer states async to reduce training stalls.
    async_eval_ngpus: int = -1  # Total GPUs for separate async eval jobs; <= 0 uses synchronous eval.

    # Wandb
    log_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None

    # Random seed
    seed: int = 1
    nccl_timeout: int = 3600  # NCCL timeout in seconds.

    cluster_check_level: int = 1  # Cluster check level at start. 0 will only log env. >=1 will run communication benchmarks in addition.

    @property
    def data_parallel_size(self) -> int:
        assert model_parallel_is_initialized()
        total_model_parallel_size = self.context_parallel_size * self.model_parallel_size
        assert self.slurm.world_size % total_model_parallel_size == 0
        size = self.slurm.world_size // total_model_parallel_size
        assert size == get_data_parallel_world_size()
        return size

    @property
    def data_parallel_rank(self) -> int:
        assert model_parallel_is_initialized()
        rank = self.slurm.global_rank // self.model_parallel_size // self.context_parallel_size
        assert rank == get_data_parallel_rank()
        return rank

    @property
    def context_parallel_rank(self) -> int:
        assert model_parallel_is_initialized()
        rid = (self.slurm.global_rank // self.model_parallel_size) % self.context_parallel_size
        assert rid == get_context_parallel_rank()
        return rid

    @property
    def model_parallel_rank(self) -> int:
        assert model_parallel_is_initialized()
        rid = self.slurm.global_rank % self.model_parallel_size
        assert rid == get_model_parallel_rank()
        return rid

    def __post_init__(self):
        # sanity check
        assert self.dtype in ["bf16", "fp16", "fp32"]
        if self.mup is not None:
            self.mup.resolve(self.model, self.optim)
        if self.optim.optimizer == 'dion_muon':
            if self.model.ddp_backend != 'fsdp2':
                raise ValueError('dion_muon requires model.ddp_backend=fsdp2')
            if self.model_parallel_size != 1:
                raise ValueError('dion_muon tensor parallelism is not supported')
            if self.model.num_experts or self.model.num_values:
                raise ValueError('dion_muon currently supports dense FFNs only')
            if not self.dcp_for_optimizer or self.async_checkpointing:
                raise ValueError('dion_muon requires synchronous DCP optimizer checkpoints')
            if self.model.memory_efficient_norm:
                raise ValueError('dion_muon qualification requires memory_efficient_norm=false')
            if self.seq_len % (self.context_parallel_size * self.model.chunk_size):
                raise ValueError('seq_len must be divisible by context_parallel_size * chunk_size')
        if self.dtype == "fp16":
            assert self.loss_rescaling, "loss rescaling needed for fp16!"

        # MoE
        if self.model.num_experts > 0:
            assert self.model.num_experts % self.model_parallel_size == 0, f"{self.model.num_experts}/{self.model_parallel_size}"
        if self.moe_router_load_balancing_type is not None:
            assert self.moe_router_load_balancing_type in ['dot', 'entropy']
            assert self.moe_aux_loss_coeff >= 0.0
        # freq checks
        assert self.dump_freq % self.log_freq == 0
        assert self.eval_freq < 0 or (self.eval_freq % self.log_freq == 0)
        if self.health_metrics:
            assert self.health_log_freq > 0
            assert self.model.arch in ("transformer", "gekko"), "Health metrics require Transformer or Gekko"
            if self.model.arch == "gekko":
                assert self.model.attn_act_func == "softmax", "Gekko health metrics currently require softmax attention"
                assert self.model.num_experts == 0 and self.model.num_values == 0, "Gekko health metrics currently require dense layers"
        if self.health_gekko_stats:
            assert self.health_metrics, "Gekko statistics require health_metrics=true"
            assert self.model.arch == "gekko", "Gekko statistics require model.arch=gekko"
        if self.health_query_key_stats:
            assert self.health_metrics, "Query/key statistics require health_metrics=true"
        if self.health_moe_stats:
            assert self.health_metrics, "MoE statistics require health_metrics=true"
            assert self.model.num_experts > 0, "MoE statistics require model.num_experts > 0"
        if self.health_mova_stats:
            assert self.health_metrics, "MoVA statistics require health_metrics=true"
            assert self.model.num_values > 0, "MoVA statistics require model.num_values > 0"
        if self.health_logits_stats:
            assert self.health_metrics, "Logit statistics require health_metrics=true"
        if self.health_weight_norms or self.health_gradient_norms:
            assert self.health_metrics, "Matrix norms require health_metrics=true"
            # Dense Gekko is enforced by the health_metrics gate above.
            assert self.model.num_experts == 0 and self.model.arch in ("transformer", "gekko"), (
                "Matrix norms currently require a dense Transformer or Gekko"
            )
            assert self.model.swiglu, "Matrix norms require SwiGLU"

        # async eval
        if self.async_eval_ngpus > 0:
            assert self.keep_eval_checkpoints
            assert self.async_eval_ngpus % self.model_parallel_size == 0, f"{self.async_eval_ngpus}/{self.model_parallel_size}"
            assert self.async_eval_ngpus % self.slurm.gpus_per_node == 0, f"{self.async_eval_ngpus}/{self.slurm.gpus_per_node}"
        if self.async_eval_ngpus > 0 or self.keep_eval_checkpoints:
            # be sure that we dump for each eval
            assert self.eval_freq < 0 or (self.eval_freq % self.dump_freq == 0)

        assert self.seq_len % self.model.chunk_size == 0, f"{self.seq_len}/{self.model.chunk_size}"
        assert self.valid.seq_len % self.model.chunk_size == 0, f"{self.valid.seq_len}/{self.model.chunk_size}"

        assert self.model.num_heads % self.model_parallel_size == 0, f"{self.model.num_heads}/{self.model_parallel_size}"
        if self.model.arch == 'transformer':
            assert self.model.layernorm_num_groups == 1 or self.model.layernorm_num_groups % self.model_parallel_size == 0, \
                f"{self.model.layernorm_num_groups}/{self.model_parallel_size}"
        elif self.model.arch in ['gekko']:
            assert self.model.layernorm_num_groups % self.model_parallel_size == 0, \
                f"{self.model.layernorm_num_groups}/{self.model_parallel_size}"
