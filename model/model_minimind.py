import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        ### MSA (Memory Sparse Attention) specific configs (ignored if use_msa = False)
        # Paper: "MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens"
        # arXiv:2603.23516 — router projectors + chunk mean-pool + top-k sparse retrieval
        self.use_msa = kwargs.get("use_msa", False)
        # Only latter-half layers get routing heads; lower layers do independent document processing
        # without sparse retrieval (paper §3.2.1: "empirical analysis reveals initial layers fail
        # to capture high-level semantics necessary for effective retrieval")
        self.msa_start_layer = kwargs.get("msa_start_layer", num_hidden_layers // 2)
        # Chunk size P for mean-pooling φ(·) that compresses K/V/K_R into latent representations
        # (paper eq 1, implementation detail: P=64 used in all experiments)
        self.msa_chunk_size = kwargs.get("msa_chunk_size", 64)
        # Number of top-k documents selected for sparse attention context assembly (paper eq 3-4)
        self.msa_top_k = kwargs.get("msa_top_k", 16)
        # Weight coefficient for the auxiliary contrastive routing loss L_aux (paper eq 5)
        # Main pre-training phase uses L = L_LLM + 0.1 * L_aux
        self.msa_aux_loss_coef = kwargs.get("msa_aux_loss_coef", 0.1)
        # Temperature τ for InfoNCE-style supervised contrastive loss (paper eq 5)
        self.msa_temperature = kwargs.get("msa_temperature", 0.05)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

def chunk_mean_pool(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """φ(·) operator — paper arXiv:2603.23516 eq 1.
    Segment-wise mean pooling that compresses K / V / K_R into latent chunk representations.
    Args:
        x:          Tensor (..., seq_len, d)  — any leading batch/head dimensions allowed
        chunk_size: tokens per chunk P (paper default P = 64)
    Returns:
        Tensor (..., num_chunks, d),  num_chunks = ceil(seq_len / chunk_size)
    """
    *leading, seq_len, d = x.shape
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad > 0:
        x = F.pad(x, (0, 0, 0, pad))
    return x.view(*leading, -1, chunk_size, d).mean(dim=-2)

class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig, layer_id: int = 0):
        super().__init__()
        self.config = config
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
        # ── MSA Router Projectors (paper §3.2.1) ─────────────────────────────
        # Applied only to the latter-half layers (layer_id >= msa_start_layer).
        # Lower layers still process documents independently but skip sparse retrieval.
        self.is_msa_layer = config.use_msa and (layer_id >= config.msa_start_layer)
        if self.is_msa_layer:
            # W_QR  — Router Q Projector: Q_R = x · W_QR  (routing query, generated online)
            self.qr_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            # W_KR  — Router K Projector: K_R = H · W_KR  (routing key, computed offline per doc)
            self.kr_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.qr_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.kr_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _msa_score(self, x: torch.Tensor, kr_bar: torch.Tensor,
                   seq_len: int, bsz: int):
        """Routing-only pass: compute per-document scores and select top-k indices.

        Separated from _msa_forward so that tiered-storage callers can:
          1. Call _msa_score with GPU-resident kr_bar to get top_k_idx cheaply.
          2. Async-fetch only the selected k_bar / v_bar from CPU DRAM.
          3. Call _msa_forward with the pre-fetched tensors.

        Args:
            x:      (bsz, seq_len, hidden_size) — layernorm'd attention input.
            kr_bar: (bsz, N_docs, C, n_kv_heads, head_dim) — GPU-resident routing keys.
            seq_len, bsz: shape scalars.

        Returns:
            top_k_idx:  LongTensor (bsz, top_k)
            doc_scores: Tensor     (bsz, N_docs)  — for aux_loss computation
        """
        N_docs, C = kr_bar.shape[1], kr_bar.shape[2]
        xqr = self.qr_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xqr = self.qr_norm(xqr)
        xqr = F.normalize(xqr, dim=-1)
        kr_flat = F.normalize(kr_bar, dim=-1).view(bsz, N_docs * C, self.n_local_kv_heads, self.head_dim)
        xqr_h = xqr.permute(0, 2, 1, 3)      # (bsz, n_kv, seq_len, head_dim)
        kr_h  = kr_flat.permute(0, 2, 3, 1)   # (bsz, n_kv, head_dim, N_docs*C)
        cos_sim = torch.matmul(xqr_h, kr_h)   # (bsz, n_kv, seq_len, N_docs*C)
        chunk_scores = cos_sim.permute(0, 2, 3, 1).mean(dim=-1).max(dim=1).values  # (bsz, N*C)
        doc_scores   = chunk_scores.view(bsz, N_docs, C).max(dim=-1).values         # (bsz, N_docs)
        top_k = min(self.config.msa_top_k, N_docs)
        _, top_k_idx = torch.topk(doc_scores, k=top_k, dim=-1)   # (bsz, top_k)
        return top_k_idx, doc_scores

    def _msa_forward(self, x: torch.Tensor, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor,
                     memory_bank: dict, seq_len: int, bsz: int) -> torch.Tensor:
        """MSA sparse attention forward (paper §3.2.1, eq 1-4).

        Two calling conventions
        ───────────────────────
        A) Training / simple inference  — memory_bank contains "k_bar", "v_bar", "kr_bar"
           (all on the same device).  Routing + gather happen here internally.

        B) Tiered-storage inference     — memory_bank contains "sel_k", "sel_v" (already
           gathered, on GPU) plus "kr_bar" (GPU).  Routing was done externally by
           _msa_score + _fetch_topk_from_tiered; this method skips the gather step.

        memory_bank optional keys
        ─────────────────────────
          "k_bar":   (bsz, N_docs, C, n_kv_heads, head_dim)   — full content KV (convention A)
          "v_bar":   (bsz, N_docs, C, n_kv_heads, head_dim)
          "kr_bar":  (bsz, N_docs, C, n_kv_heads, head_dim)   — routing keys (both conventions)
          "sel_k":   (bsz, top_k,  C, n_kv_heads, head_dim)   — pre-fetched KV  (convention B)
          "sel_v":   (bsz, top_k,  C, n_kv_heads, head_dim)
          "pos_doc_mask": bool (bsz, N_docs)                   — training only

        Returns
        ───────
          output: (bsz, n_heads, seq_len, head_dim)
        """
        kr_bar = memory_bank["kr_bar"]   # always present
        N_docs, C = kr_bar.shape[1], kr_bar.shape[2]

        if "sel_k" in memory_bank:
            # ── Convention B: KV already fetched by caller (tiered path) ──────
            sel_k = memory_bank["sel_k"]  # (bsz, top_k, C, n_kv, hd)
            sel_v = memory_bank["sel_v"]
            top_k = sel_k.shape[1]
            # Routing scores still needed for aux_loss; re-use cheap _msa_score.
            _, doc_scores = self._msa_score(x, kr_bar, seq_len, bsz)
        else:
            # ── Convention A: route + gather in one shot (training / simple inf) ──
            top_k_idx, doc_scores = self._msa_score(x, kr_bar, seq_len, bsz)
            top_k = top_k_idx.shape[1]
            idx = top_k_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                bsz, top_k, C, self.n_local_kv_heads, self.head_dim
            )
            k_bar = memory_bank["k_bar"]
            v_bar = memory_bank["v_bar"]
            sel_k = torch.gather(k_bar, dim=1, index=idx)
            sel_v = torch.gather(v_bar, dim=1, index=idx)

        # ── Aux routing loss  L_aux (paper §3.3.1, eq 5) ─────────────────────
        # memory_bank["pos_doc_mask"]: bool (bsz, N_docs) — provided only in training.
        if self.training and self.config.msa_aux_loss_coef > 0 and "pos_doc_mask" in memory_bank:
            pos_mask = memory_bank["pos_doc_mask"].to(doc_scores.device)
            tau = self.config.msa_temperature
            log_scores = doc_scores / tau
            log_scores = log_scores - log_scores.detach().max(dim=-1, keepdim=True).values
            exp_scores  = torch.exp(log_scores)
            denom_neg   = (exp_scores * (~pos_mask).float()).sum(dim=-1, keepdim=True)
            pos_log_p   = log_scores - torch.log(exp_scores + denom_neg + 1e-8)
            n_pos = pos_mask.float().sum(dim=-1).clamp(min=1)
            self.msa_aux_loss = (-(pos_log_p * pos_mask.float()).sum(dim=-1) / n_pos).mean() \
                                * self.config.msa_aux_loss_coef
        else:
            self.msa_aux_loss = doc_scores.new_zeros(1).squeeze()

        # ── Context assembly + sparse generation  (eq 3-4) ───────────────────
        mem_ctx_len  = top_k * C
        sel_k_flat   = sel_k.view(bsz, mem_ctx_len, self.n_local_kv_heads, self.head_dim)
        sel_v_flat   = sel_v.view(bsz, mem_ctx_len, self.n_local_kv_heads, self.head_dim)
        local_kv_len = xk.shape[1]      # past_len + seq_len
        past_len     = local_kv_len - seq_len
        k_ctx = torch.cat([sel_k_flat, xk], dim=1)
        v_ctx = torch.cat([sel_v_flat, xv], dim=1)
        mem_mask   = torch.zeros(seq_len, mem_ctx_len, dtype=xq.dtype, device=x.device)
        local_mask = torch.triu(
            torch.full((seq_len, local_kv_len), float('-inf'), dtype=xq.dtype, device=x.device),
            diagonal=past_len + 1,
        )
        attn_mask = torch.cat([mem_mask, local_mask], dim=-1).unsqueeze(0).unsqueeze(0)
        output = F.scaled_dot_product_attention(
            xq.transpose(1, 2),
            repeat_kv(k_ctx, self.n_rep).transpose(1, 2),
            repeat_kv(v_ctx, self.n_rep).transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return output  # (bsz, n_heads, seq_len, head_dim)

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None, memory_bank=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # ── MSA sparse attention path (latter-half layers + memory bank present) ──
        if self.is_msa_layer and memory_bank is not None:
            output = self._msa_forward(x, xq, xk, xv, memory_bank, seq_len, bsz)
        # ── Standard dense attention path (original, unchanged) ──────────────────
        else:
            self.msa_aux_loss = xq.new_zeros(1).squeeze()
            xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
            if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
                output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
            else:
                scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
                if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
                output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config, layer_id=layer_id)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None, memory_bank=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask, memory_bank
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # memory_bank_list: Optional[List[dict|None]] — one entry per layer.
        # MSA layers receive their pre-computed compressed memory bank;
        # non-MSA layers and layers without a bank receive None (dense attention as usual).
        memory_bank_list = kwargs.pop('memory_bank_list', None)
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            memory_bank = memory_bank_list[i] if memory_bank_list is not None else None
            # Global RoPE for query in MSA layers (paper §3.2.2):
            # After sparse retrieval, K_ctx = [K̄_selected ; K_local] where K̄_selected
            # has shape (top_k × C) compressed KV vectors (C = chunks_per_doc after pooling).
            # The query's position IDs must start from (start_pos + top_k * C) so the model
            # perceives active tokens as a logical continuation after all retrieved memory KVs.
            # Using just msa_top_k (doc count) would be wrong when C > 1 (which is always true
            # in practice, e.g. 1024-token docs / chunk_size=64 → C=16, so offset=top_k*C=256).
            if layer.self_attn.is_msa_layer and memory_bank is not None:
                # Support both Convention A (k_bar key) and Convention B (sel_k key).
                _kv_key = "sel_k" if "sel_k" in memory_bank else "k_bar"
                actual_top_k = memory_bank[_kv_key].shape[1]
                C = memory_bank[_kv_key].shape[2]          # chunks per doc (or top_k chunks)
                eff_start = start_pos + actual_top_k * C
                layer_pos_emb = (
                    self.freqs_cos[eff_start:eff_start + seq_length],
                    self.freqs_sin[eff_start:eff_start + seq_length],
                )
            else:
                layer_pos_emb = position_embeddings
            hidden_states, present = layer(
                hidden_states,
                layer_pos_emb,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
                memory_bank=memory_bank,
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        moe_aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        msa_aux_loss = sum([l.self_attn.msa_aux_loss for l in self.layers if l.self_attn.is_msa_layer], hidden_states.new_zeros(1).squeeze())
        aux_loss = moe_aux_loss + msa_aux_loss
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    @torch.no_grad()
    def build_document_memory(self, doc_token_ids_list, device=None):
        """Stage 1 offline document encoding with Parallel RoPE (paper §3.2.2, §3.4.1).

        Each document resets its position IDs to 0 independently (Parallel RoPE),
        decoupling positional encoding from the total number/order of documents in
        the memory bank. This allows training on short contexts (e.g. 64k) while
        extrapolating to 100M-token memory banks at inference without positional
        distribution shift.

        After per-layer computation, K, V, and K_R are chunk-mean-pooled (φ(·), eq 1)
        into compact representations K̄, V̄, K̄_R that are stored in the memory bank.
        K̄_R (routing keys) have NO RoPE — routing scores should reflect semantic
        content, not position. K̄ has parallel-RoPE applied before pooling.

        Args:
            doc_token_ids_list: list of (1, doc_len_i) LongTensors, one per document.
                                Documents may have different lengths.
            device: target device (defaults to the model's first-parameter device).

        Returns:
            memory_bank_list: list[dict | None] of length num_hidden_layers.
                Non-MSA layer entries (layer_id < msa_start_layer): None.
                MSA layer entries:
                    {"k_bar":  (1, N_docs, C, n_kv_heads, head_dim),
                     "v_bar":  (1, N_docs, C, n_kv_heads, head_dim),
                     "kr_bar": (1, N_docs, C, n_kv_heads, head_dim)}
                C = max(ceil(doc_len_i / msa_chunk_size)) across all documents;
                shorter-document chunk tensors are zero-padded along dim C.
        """
        if device is None:
            device = next(self.parameters()).device
        chunk_size  = self.config.msa_chunk_size
        num_layers  = self.config.num_hidden_layers
        all_k  = [[] for _ in range(num_layers)]
        all_v  = [[] for _ in range(num_layers)]
        all_kr = [[] for _ in range(num_layers)]
        for doc_ids in doc_token_ids_list:
            doc_ids = doc_ids.to(device)
            doc_len = doc_ids.shape[1]
            # Parallel RoPE: positions 0..doc_len-1, independent per document (paper §3.2.2)
            doc_cos = self.model.freqs_cos[:doc_len]
            doc_sin = self.model.freqs_sin[:doc_len]
            doc_pos_emb = (doc_cos, doc_sin)
            hidden = self.model.dropout(self.model.embed_tokens(doc_ids))
            for layer_id, layer in enumerate(self.model.layers):
                # Capture layernorm'd attention input BEFORE running the layer.
                # This mirrors what Attention.forward receives for K/V/K_R projections.
                attn_input = layer.input_layernorm(hidden)
                # Run full layer with parallel-RoPE; no memory bank during offline encoding.
                hidden, _ = layer(hidden, doc_pos_emb, past_key_value=None, use_cache=False)
                if layer.self_attn.is_msa_layer:
                    bsz_d, sl, _ = attn_input.shape
                    n_kv = layer.self_attn.n_local_kv_heads
                    hd   = layer.self_attn.head_dim
                    # K: backbone projection → k_norm → parallel RoPE (document-local pos)
                    xk = layer.self_attn.k_proj(attn_input).view(bsz_d, sl, n_kv, hd)
                    xv = layer.self_attn.v_proj(attn_input).view(bsz_d, sl, n_kv, hd)
                    xk = layer.self_attn.k_norm(xk)
                    _, xk = apply_rotary_pos_emb(xk, xk, doc_cos, doc_sin)
                    # K_R: router projection → kr_norm (no RoPE; routing is position-agnostic)
                    xkr = layer.self_attn.kr_proj(attn_input).view(bsz_d, sl, n_kv, hd)
                    xkr = layer.self_attn.kr_norm(xkr)
                    # Chunk mean-pool over seq_len (φ(·), paper eq 1).
                    # chunk_mean_pool treats last two dims as (seq_len, d); we first
                    # permute seq_len to second-to-last: (bsz, seq, n_kv, hd) →
                    # (bsz, n_kv, seq, hd) → pool → (bsz, n_kv, C, hd) → (bsz, C, n_kv, hd)
                    xk_t  = xk.permute(0, 2, 1, 3)
                    xv_t  = xv.permute(0, 2, 1, 3)
                    xkr_t = xkr.permute(0, 2, 1, 3)
                    k_bar  = chunk_mean_pool(xk_t,  chunk_size).permute(0, 2, 1, 3)
                    v_bar  = chunk_mean_pool(xv_t,  chunk_size).permute(0, 2, 1, 3)
                    kr_bar = chunk_mean_pool(xkr_t, chunk_size).permute(0, 2, 1, 3)
                    # Each tensor: (1, C_i, n_kv, hd)
                    all_k[layer_id].append(k_bar)
                    all_v[layer_id].append(v_bar)
                    all_kr[layer_id].append(kr_bar)
        # Stack documents: list of (1, C_i, n_kv, hd) → (1, N_docs, max_C, n_kv, hd)
        # Documents with fewer chunks are zero-padded along the C dimension.
        def _pad_and_stack(tensors):
            max_c  = max(t.shape[1] for t in tensors)
            padded = [F.pad(t, (0, 0, 0, 0, 0, max_c - t.shape[1])) for t in tensors]
            return torch.stack(padded, dim=1)   # (1, N_docs, max_C, n_kv, hd)
        memory_bank_list = []
        for layer_id in range(num_layers):
            if not self.model.layers[layer_id].self_attn.is_msa_layer or not all_k[layer_id]:
                memory_bank_list.append(None)
            else:
                memory_bank_list.append({
                    "k_bar":  _pad_and_stack(all_k[layer_id]),
                    "v_bar":  _pad_and_stack(all_v[layer_id]),
                    "kr_bar": _pad_and_stack(all_kr[layer_id]),
                })
        return memory_bank_list

    @torch.no_grad()
    def build_document_memory_tiered(self, doc_token_ids_list, gpu_device=None):
        """Stage 1 offline encoding with **tiered storage** (paper §3.4.2 Memory Parallel).

        Storage layout
        ──────────────
        GPU (VRAM)  — routing keys K̄_R:  used for every query to score all documents.
                      Must be on GPU to keep retrieval latency low.
        CPU (DRAM)  — content KVs K̄, V̄: loaded only for the top-k selected documents.
                      Async prefetch with pin_memory for fast host→device transfer.

        This decouples memory *capacity* from VRAM limits:
        - 100M-token corpus @ chunk_size=64, 8 KV-heads, head_dim=128, 18 layers, BF16
          ≈ 169 GB total; K̄_R alone ≈ 56 GB (stays on GPU across multiple GPUs),
          K̄+V̄ ≈ 113 GB offloaded to host DRAM.

        Args:
            doc_token_ids_list: list of (1, doc_len_i) LongTensors.
            gpu_device: target GPU device for K̄_R (defaults to model device).

        Returns:
            tiered_memory: list[dict | None] of length num_hidden_layers.
                MSA layer entries:
                    {"kr_bar_gpu": (1, N_docs, C, n_kv_heads, head_dim)  — on gpu_device
                     "k_bar_cpu":  (1, N_docs, C, n_kv_heads, head_dim)  — pinned CPU tensor
                     "v_bar_cpu":  (1, N_docs, C, n_kv_heads, head_dim)  — pinned CPU tensor}
        """
        if gpu_device is None:
            gpu_device = next(self.parameters()).device
        # Build full memory bank on CPU first (avoids GPU OOM during encoding)
        cpu_bank = self.build_document_memory(doc_token_ids_list, device='cpu')
        tiered = []
        for entry in cpu_bank:
            if entry is None:
                tiered.append(None)
                continue
            # Pin k_bar / v_bar in host memory for fast async DMA transfers
            k_cpu = entry["k_bar"].pin_memory() if gpu_device.type != 'cpu' else entry["k_bar"]
            v_cpu = entry["v_bar"].pin_memory() if gpu_device.type != 'cpu' else entry["v_bar"]
            # Move routing keys to GPU — these are read on every forward pass
            kr_gpu = entry["kr_bar"].to(gpu_device, non_blocking=True)
            tiered.append({"kr_bar_gpu": kr_gpu, "k_bar_cpu": k_cpu, "v_bar_cpu": v_cpu})
        return tiered

    @staticmethod
    def _fetch_topk_from_tiered(tiered_memory, top_k_idx, gpu_device):
        """Given a tiered memory bank and already-selected document indices (LongTensor
        (bsz, top_k)), asynchronously prefetch the corresponding K̄/V̄ slices from
        pinned CPU memory to gpu_device and assemble per-layer memory_bank dicts
        ready for `Attention._msa_forward`.

        This is the online Step 2 of the three-stage inference (paper §3.4.1):
        'only the compact K̄, V̄ of selected documents are loaded' after scoring.
        """
        assembled = []
        for entry in tiered_memory:
            if entry is None:
                assembled.append(None)
                continue
            bsz, N_docs, C, n_kv, hd = entry["k_bar_cpu"].shape
            top_k = top_k_idx.shape[1]
            # Gather selected slices from pinned CPU → transfer to GPU (non_blocking)
            idx = top_k_idx.cpu().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                bsz, top_k, C, n_kv, hd)
            sel_k = torch.gather(entry["k_bar_cpu"], dim=1, index=idx).to(
                gpu_device, non_blocking=True)
            sel_v = torch.gather(entry["v_bar_cpu"], dim=1, index=idx).to(
                gpu_device, non_blocking=True)
            # Rebuild a memory_bank dict that _msa_forward expects:
            # k_bar/v_bar now contain only top_k docs (not full N_docs); kr_bar
            # is the full GPU tensor (scoring was already done externally).
            assembled.append({
                "k_bar":  sel_k,                        # (bsz, top_k, C, n_kv, hd) on GPU
                "v_bar":  sel_v,                        # (bsz, top_k, C, n_kv, hd) on GPU
                "kr_bar": entry["kr_bar_gpu"],          # (bsz, N_docs, C, n_kv, hd) on GPU
            })
        return assembled

    @torch.inference_mode()
    def generate_with_memory_interleave(
        self,
        input_ids,
        tiered_memory,
        doc_texts,
        tokenizer,
        eor_token_id,
        doc_token_id_to_idx=None,
        max_retrieve_rounds=4,
        max_new_tokens=512,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        do_sample=True,
        streamer=None,
    ):
        """Memory Interleave generation (paper §3.5).

        Corrected tiered-storage flow per round
        ─────────────────────────────────────────
        0. (one-time) Use the full query hidden-states to call _msa_score with
           GPU-resident kr_bar → get top_k_idx.  (N_docs scoring, cheap)
        1. _fetch_topk_from_tiered: async-load only the top_k selected k_bar/v_bar
           slices from pinned CPU DRAM → GPU.  (~top_k × C × layers × heads × hd)
        2. Autoregressive generation with sel_k / sel_v injected via "sel_k" / "sel_v"
           keys in the memory_bank dict (Convention B of _msa_forward).
        3. On <End-of-Retrieve>: parse emitted doc IDs using doc_token_id_to_idx map,
           append their original texts to input_ids, re-run top_k routing on the
           expanded context, re-fetch KV → next round.

        Args:
            input_ids:           (1, seq_len) LongTensor on GPU — initial query.
            tiered_memory:       output of build_document_memory_tiered().
            doc_texts:           list[str] — one entry per document.
            tokenizer:           object with .encode(str) → list[int].
            eor_token_id:        int — <End-of-Retrieve> special token ID.
            doc_token_id_to_idx: dict[int, int] | None
                Mapping from the token IDs the model emits to document indices in
                doc_texts / tiered_memory.  Prevents collision with ordinary vocabulary
                tokens.  If None, the raw token value is used as the document index
                (safe only when doc IDs are guaranteed not to overlap with vocab tokens,
                e.g. via reserved token IDs at the end of the vocabulary).
            max_retrieve_rounds: int — safety cap on multi-hop iterations.
            max_new_tokens:      int — per-round generation budget.
            temperature / top_p / top_k / do_sample: sampling hyperparams.
            streamer:            optional streaming callback.

        Returns:
            input_ids: (1, total_len) — full generated sequence.
        """
        device    = input_ids.device
        n_docs    = len(doc_texts)
        # doc_token_id_to_idx defaults to identity mapping.
        # Callers should provide an explicit map whose domain is a reserved set of
        # token IDs that do not collide with ordinary vocabulary tokens.
        _id2idx   = doc_token_id_to_idx if doc_token_id_to_idx is not None else {}

        def _resolve_doc_idx(tok_id):
            """Map emitted token → doc index.  Falls back to raw value when no map."""
            if _id2idx:
                return _id2idx.get(tok_id, None)
            # Raw fallback: only accept values in valid range
            return tok_id if 0 <= tok_id < n_docs else None

        def _sample_one(logits_last):
            logits_last = logits_last / temperature
            if top_k > 0:
                logits_last[logits_last < torch.topk(logits_last, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sl, si = torch.sort(logits_last, descending=True)
                mask = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits_last[mask.scatter(1, si, mask)] = -float('inf')
            return (torch.multinomial(torch.softmax(logits_last, dim=-1), 1)
                    if do_sample else torch.argmax(logits_last, dim=-1, keepdim=True))

        def _route_and_fetch(query_ids):
            """Run one forward pass to get query hidden states, call _msa_score on the
            last token's representation using GPU-resident kr_bar, then async-fetch
            the selected k_bar / v_bar from CPU.  Returns active_mb.

            We use a single no-cache forward to obtain the final hidden state of the
            full current context, then extract the first MSA layer's Q_R score to
            decide which documents to retrieve.  This matches the paper's Stage 2:
            'routing query Q_R is matched against cached global routing keys K̄_R'.
            """
            # Cheap scoring pass: forward without memory bank to get hidden states
            with torch.no_grad():
                hidden_states = self.model.dropout(self.model.embed_tokens(query_ids))
                seq_len = query_ids.shape[1]
                pos_emb = (self.model.freqs_cos[:seq_len], self.model.freqs_sin[:seq_len])
                for layer in self.model.layers:
                    hidden_states, _ = layer(hidden_states, pos_emb)
                # hidden_states: (1, seq_len, hidden_size) — use last token for routing
                query_repr = layer.input_layernorm(hidden_states)  # reuse final layer norm

            # Find the first MSA layer's attention module for scoring
            first_msa = next(l.self_attn for l in self.model.layers if l.self_attn.is_msa_layer)
            # Build a composite top_k_idx by taking the union of scores from all MSA layers
            # (simple approach: use the first MSA layer's scorer; paper does per-layer routing)
            first_entry = next(e for e in tiered_memory if e is not None)
            kr_bar_gpu = first_entry["kr_bar_gpu"].to(device)
            bsz, sl, _ = query_repr.shape
            top_k_idx, _ = first_msa._msa_score(query_repr, kr_bar_gpu, sl, bsz)
            # top_k_idx: (1, top_k) — same selection applied to all layers (sufficient
            # for routing; each layer has its own kr_bar but we share the selection)

            # Async-fetch selected k_bar / v_bar from CPU; assemble active_mb (Convention B)
            fetched = self._fetch_topk_from_tiered(tiered_memory, top_k_idx, device)
            # Convert fetched dicts (which have k_bar/v_bar = top_k docs) to Convention-B
            # format by renaming keys to sel_k / sel_v so _msa_forward skips re-routing.
            active = []
            for entry in fetched:
                if entry is None:
                    active.append(None)
                else:
                    active.append({
                        "sel_k":  entry["k_bar"],    # (1, top_k, C, n_kv, hd) on GPU
                        "sel_v":  entry["v_bar"],
                        "kr_bar": entry["kr_bar"],   # full (1, N_docs, C, n_kv, hd) on GPU
                    })
            return active

        if streamer:
            streamer.put(input_ids.cpu())

        # Initial routing + fetch before the first generation round
        active_mb = _route_and_fetch(input_ids)

        for _round in range(max_retrieve_rounds + 1):
            # ── Autoregressive generation for this round ──────────────────────
            past_key_values = None
            round_tokens    = []
            found_eor       = False

            for _ in range(max_new_tokens):
                past_len = past_key_values[0][0].shape[1] if past_key_values else 0
                outputs  = self.forward(
                    input_ids[:, past_len:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    memory_bank_list=active_mb,
                )
                past_key_values = outputs.past_key_values
                next_token = _sample_one(outputs.logits[:, -1, :])
                input_ids  = torch.cat([input_ids, next_token], dim=-1)
                round_tokens.append(next_token.item())

                if streamer:
                    streamer.put(next_token.cpu())

                if next_token.item() == eor_token_id:
                    found_eor = True
                    break

            if not found_eor or _round == max_retrieve_rounds:
                break   # final answer round or safety cap

            # ── Retrieval step ────────────────────────────────────────────────
            # Parse emitted doc-ID tokens (all tokens before EOR in this round).
            # Use doc_token_id_to_idx to resolve token IDs → document indices,
            # avoiding collision with ordinary vocabulary tokens.
            retrieved_doc_ids = []
            for tok in round_tokens[:-1]:   # exclude the EOR token itself
                idx = _resolve_doc_idx(tok)
                if idx is not None and idx not in retrieved_doc_ids:
                    retrieved_doc_ids.append(idx)

            if not retrieved_doc_ids:
                break   # EOR with no valid doc IDs

            # ── Append original document texts ────────────────────────────────
            # Paper §3.4.1 / §3.5: retrieved docs' raw text is appended to the
            # query so the model can extract fine-grained factual details.
            for doc_idx in retrieved_doc_ids:
                doc_tok = torch.tensor(
                    tokenizer.encode(doc_texts[doc_idx]),
                    dtype=torch.long, device=device,
                ).unsqueeze(0)
                input_ids = torch.cat([input_ids, doc_tok], dim=-1)
                if streamer:
                    streamer.put(doc_tok.cpu())

            # ── Re-route on expanded context → fetch new top-k KVs ───────────
            # The appended document text changes the query representation, so we
            # must re-score all documents and potentially load a different top-k.
            # This is the key correction: the old active_mb is discarded.
            active_mb = _route_and_fetch(input_ids)

        if streamer:
            streamer.end()
        return input_ids

    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids