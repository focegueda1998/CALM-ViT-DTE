import torch
from typing import Callable
from functools import partial
from torch.nn import functional as F

class ResidualStateManager():
    def __init__(
        self
    ):
        super().__init__()
        self.mean_q_sum = None
        self.var_q_sum = None
        self.mean_kv_sum = None
        self.var_kv_sum = None
        self.mask_sum = None
    
    def get_mean_var_sums(self, mean_q, var_q, mean_kv, var_kv):
        self.mean_q_sum = mean_q if self.mean_q_sum is None else self.mean_q_sum + mean_q
        self.var_q_sum = var_q if self.var_q_sum is None else self.var_q_sum + var_q
        self.mean_kv_sum = mean_kv if self.mean_kv_sum is None else self.mean_kv_sum + mean_kv
        self.var_kv_sum = var_kv if self.var_kv_sum is None else self.var_kv_sum + var_kv
        return self.mean_q_sum, self.var_q_sum, self.mean_kv_sum, self.var_kv_sum

# Since our tokens are processed as sequences (row-wise, column-wise), we can apply RoPE as a 1D Embedding.
# I'm not sure if 2D would still be applicable in this case since tokens are not grid based. Using 1D should
# be fine since the 2D spatial relationships are implied by the tokenization strategy (?).
class RoPE(torch.nn.Module):
    def __init__(self, seq: int, dim: int, theta: float=10.0, learned: bool=False):
        super().__init__()
        self.seq = seq
        self.dim = dim
        self.theta = theta
        self.learned = learned
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2).float() / self.dim))
        t = torch.arange(self.seq, dtype=torch.float)
        self.freqs = torch.outer(t, inv_freq)
        # According to Axial 2D Rope, the frequencies can be learned enabling finer
        # grained control for PEs over constant ones, but the actual cos / sin embeddings
        # must be regenerated after each forward pass during training. Inference must 
        # cache the learned freqs as buffers to avoid recomputation, which I have not 
        # yet done :P.
        if learned:
            self.freqs = torch.nn.Parameter(self.freqs, requires_grad=True)
        else:
            self.register_buffer("inv_freq", inv_freq)
            emb = torch.cat((self.freqs, self.freqs), dim=-1)
            self.register_buffer("cos_emb", emb.cos(), persistent=False)
            self.register_buffer("sin_emb", emb.sin(), persistent=False)
    
    def rotate_half(self, x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x):       
        if self.learned:
            emb = torch.cat((self.freqs, self.freqs), dim=-1)
            cos = emb.cos().unsqueeze(0).unsqueeze(0)
            sin = emb.sin().unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_emb[:self.seq, :].unsqueeze(0).unsqueeze(0)
            sin = self.sin_emb[:self.seq, :].unsqueeze(0).unsqueeze(0)
        return (x * cos) + (self.rotate_half(x) * sin)

# Multi-Head Latent Distribution Attention
class VMLA_Block(torch.nn.Module):
    def __init__(
        self,
        heads: int,
        dim1: int,
        dim2: int,
        dim3: int,
        mean_var_hidden: int,
        seq_length: int,
        seq_len_reduce:int,
        seq_len_new:int,
        mlp_dim: int,
        force_reduce: bool=True, # Force reduction even if input and output dims / seq lengths are the same (NOT RECOMMENDED; REQUIRES A LOT MORE PARAMETERS)
        static_epsilon_weight: bool=False,
        e2de_ratio: float=1.0, # 1 for full stochasticity, 0 for deterministic, this will be learnable if static_epsilon_weight is False
        dropout:float=0.0,
        use_mlp: bool=True,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.heads = heads
        self.head_dim_content = dim3 // heads // 2
        self.head_dim_rope = dim3 // heads // 2
        self.head_dim = self.head_dim_content + self.head_dim_rope
        #! This is ugly but I don't care :)
        # One-Shot Encoders
        self.e2de_ratio = None
        self.static_epsilon_weight = None
        self.dirty_epsilon_zq = None
        self.dirty_epsilon_zkv = None
        self.t_reduce = seq_len_new != seq_length or force_reduce
        self.reduce = dim1 != dim3 or force_reduce
        if self.reduce and self.t_reduce:
            self.e2de_ratio = e2de_ratio
            self.static_epsilon_weight = static_epsilon_weight
            if not static_epsilon_weight:
                self.e2de_ratio = torch.nn.Parameter(torch.tensor([e2de_ratio]), requires_grad=True)
            # Cheating by expressing epsilon as a learnable parameter
            self.dirty_epsilon_zq = torch.nn.Parameter(torch.randn(1, seq_len_reduce, mean_var_hidden), requires_grad=True)
            self.dirty_epsilon_zkv = torch.nn.Parameter(torch.randn(1, seq_len_reduce, mean_var_hidden), requires_grad=True)
        # Mean and Variance Bottleneck
        self.ln_q = norm_layer(dim1, bias=False)
        self.ln_kv = norm_layer(dim1, bias=False)
        self.t_mean_zq = None
        self.t_var_zq = None
        self.t_mean_zkv = None
        self.t_var_zkv = None
        # Only apply token temporal reduction if seq_len_reduce is different from seq_length (or forced)
        if self.t_reduce:
            self.t_mean_zq = torch.nn.Linear(seq_length, seq_len_reduce, bias=False)
            self.t_var_zq = torch.nn.Linear(seq_length, seq_len_reduce, bias=False)
            self.t_mean_zkv = torch.nn.Linear(seq_length, seq_len_reduce, bias=False)
            self.t_var_zkv = torch.nn.Linear(seq_length, seq_len_reduce, bias=False)
        # Only compute the mean and variance projections if input and output dims differ
        self.mean_zq = None
        self.var_zq = None
        self.mean_zkv = None
        self.var_zkv = None
        if self.reduce:
            self.mean_zq = torch.nn.Sequential(
                torch.nn.Linear(dim1, dim2, bias=False),
                torch.nn.Linear(dim2, mean_var_hidden, bias=False)
            )
            self.var_zq = torch.nn.Sequential(
                torch.nn.Linear(dim1, dim2, bias=False),
                torch.nn.Linear(dim2, mean_var_hidden, bias=False)
            )
            self.mean_zkv = torch.nn.Sequential(
                torch.nn.Linear(dim1, dim2, bias=False),
                torch.nn.Linear(dim2, mean_var_hidden, bias=False)
            )
            self.var_zkv = torch.nn.Sequential(
                torch.nn.Linear(dim1, dim2, bias=False),
                torch.nn.Linear(dim2, mean_var_hidden, bias=False)
            )
        # Decoders
        # Similarly, only apply token temporal upsampling if seq_len_reduce is different from seq_len_new
        self.t_qz_upsample = None
        self.t_kz_upsample = None
        self.t_vz_upsample = None
        self.t_qr_proj = None
        self.t_kr_proj = None
        if self.t_reduce:
            self.t_qz_upsample = torch.nn.Linear(seq_len_reduce, seq_len_new, bias=False)
            self.t_kz_upsample = torch.nn.Linear(seq_len_reduce, seq_len_new, bias=False)
            self.t_vz_upsample = torch.nn.Linear(seq_len_reduce, seq_len_new, bias=False)
            self.t_qr_proj = torch.nn.Linear(seq_len_reduce, seq_len_new, bias=False)
            self.t_kr_proj = torch.nn.Linear(seq_length, seq_len_new, bias=False)
        # Only compute the upsample projections if input and output dims differ
        self.qz_upsample = None
        self.kz_upsample = None
        self.vz_upsample = None
        if self.reduce:
            self.qz_upsample = torch.nn.Linear(mean_var_hidden, dim2, bias=False)
            self.kz_upsample = torch.nn.Linear(mean_var_hidden, dim2, bias=False)
            self.vz_upsample = torch.nn.Linear(mean_var_hidden, dim2, bias=False)
        # Projections
        self.q_proj = torch.nn.Linear(dim3 if dim1 == dim3 and not force_reduce else dim2, self.heads * self.head_dim_content, bias=False)
        self.k_proj = torch.nn.Linear(dim3 if dim1 == dim3 and not force_reduce else dim2, self.heads * self.head_dim_content, bias=False)
        self.v_proj = torch.nn.Linear(dim3 if dim1 == dim3 and not force_reduce else dim2, dim3, bias=False)
        # Decoupled RoPE Projections as per Multi-Head Latent Attention Paper
        self.qr_proj_0 = None
        if not self.reduce and not self.t_reduce:
            # Decoupled RoPE projections for Q still needs to mimic the cache mechanism even if no dimension reduction is applied
            self.qr_proj_0 = torch.nn.Linear(dim1, dim3, bias=False)
        self.qr_proj = torch.nn.Linear(dim3 if dim1 == dim3 and not force_reduce else mean_var_hidden, self.head_dim_rope * self.heads, bias=False)
        self.kr_proj = torch.nn.Linear(dim1, self.head_dim_rope * self.heads, bias=False)
        self.linear_mask = torch.nn.Sequential(
            torch.nn.Linear(seq_len_new, seq_len_new * 2, bias=False),
            torch.nn.Linear(seq_len_new * 2, seq_len_new, bias=False)
        )
        self.input_t_proj = None
        self.input_proj = None
        # Dimension Adjustments, Do not force reduction here, we won't modify seq_length or dim1 unless necessary
        # This isn't optimal but closer to how residual connections are supposed to work.
        if seq_len_new != seq_length:
            self.input_t_proj = torch.nn.Linear(seq_length, seq_len_new, bias=False)
        if dim1 != dim3:
            self.input_proj = torch.nn.Linear(dim1, dim3, bias=False)
        # Attention
        self.rope_q = RoPE(seq_len_new, self.head_dim_rope, learned=True)
        self.rope_k = RoPE(seq_len_new, self.head_dim_rope, learned=True)
        self.out_proj = torch.nn.Linear(dim3, dim3, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.ln_2 = norm_layer(dim3, bias=False)
        self.mlp = None
        if use_mlp:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dim3, mlp_dim, bias=False),
                torch.nn.GELU(approximate='none'),
                torch.nn.Dropout(dropout, inplace=False),
                torch.nn.Linear(mlp_dim, dim3, bias=False),
            )

    def forward(self, input_q, input_kv=None, state_manager=None, mask=False):
        # One-Shot Encoders
        residual = input_q
        input_kv = input_q if input_kv is None else input_kv
        xq = input_q
        xkv = input_kv
        xq = self.ln_q(xq)
        xkv = self.ln_kv(xkv)
        qz = xq
        kz = xkv
        vz = xkv
        qr = input_q
        kr = input_kv
        if self.t_reduce and self.reduce:
            xq = xq.permute(0, 2, 1)
            mean_zq = self.t_mean_zq(xq)
            var_zq = self.t_var_zq(xq)
            mean_zq = mean_zq.permute(0, 2, 1)
            var_zq = var_zq.permute(0, 2, 1)
            mean_zkv = self.t_mean_zkv(xkv.permute(0, 2, 1))
            var_zkv = self.t_var_zkv(xkv.permute(0, 2, 1))
            mean_zkv = mean_zkv.permute(0, 2, 1)
            var_zkv = var_zkv.permute(0, 2, 1)
            mean_zq = self.mean_zq(mean_zq)
            var_zq = self.var_zq(var_zq)
            mean_zkv = self.mean_zkv(mean_zkv)
            var_zkv = self.var_zkv(var_zkv)
            # Get residual states from state manager
            if state_manager is not None:
                mean_zq, var_zq, mean_zkv, var_zkv = state_manager.get_mean_var_sums(mean_zq, var_zq, mean_zkv, var_zkv)
            # Compute samples
            zq = mean_zq + ((self.e2de_ratio * torch.randn_like(var_zq)) + ((1 - self.e2de_ratio) * self.dirty_epsilon_zq)) * torch.exp(0.5 * var_zq)
            zkv = mean_zkv + ((self.e2de_ratio * torch.randn_like(var_zkv)) + ((1 - self.e2de_ratio) * self.dirty_epsilon_zkv)) * torch.exp(0.5 * var_zkv)
            qr = zq
            qz = zq.permute(0, 2, 1)
            qz = self.t_qz_upsample(qz)
            qz = qz.permute(0, 2, 1)
            zkv = zkv.permute(0, 2, 1)
            kz = self.t_kz_upsample(zkv)
            kz = kz.permute(0, 2, 1)
            vz = self.t_vz_upsample(zkv)
            vz = vz.permute(0, 2, 1)
            qr = qr.permute(0, 2, 1)
            qr = self.t_qr_proj(qr)
            qr = qr.permute(0, 2, 1)
            kr = kr.permute(0, 2, 1)
            kr = self.t_kr_proj(kr)
            kr = kr.permute(0, 2, 1)
            qz = self.qz_upsample(qz)
            kz = self.kz_upsample(kz)
            vz = self.vz_upsample(vz)
        qz = self.q_proj(qz)
        kz = self.k_proj(kz)
        vz = self.v_proj(vz)
        if self.qr_proj_0 is not None:
            qr = self.qr_proj_0(qr)
        qr = self.qr_proj(qr)
        kr = self.kr_proj(kr)
        mask_mat = self.linear_mask(qz @ kz.permute(0, 2, 1)) if mask else None
        mask_mat = mask_mat.unsqueeze(1)
        batch_size = qz.shape[0]
        seq_len_q = qz.shape[1]
        seq_len_kv = kz.shape[1]

        q = qz.view(batch_size, seq_len_q, self.heads, self.head_dim_content).transpose(1, 2)
        k = kz.view(batch_size, seq_len_kv, self.heads, self.head_dim_content).transpose(1, 2)
        v = vz.view(batch_size, seq_len_kv, self.heads, self.head_dim).transpose(1, 2)
        qr = qr.view(batch_size, seq_len_q, self.heads, self.head_dim_rope).transpose(1, 2)
        kr = kr.view(batch_size, seq_len_kv, self.heads, self.head_dim_rope).transpose(1, 2)
        q = torch.cat((q, self.rope_q(qr)), dim=-1)
        k = torch.cat((k, self.rope_k(kr)), dim=-1)
        # Attention
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask_mat if mask else None,
            dropout_p=0.0,
            is_causal=False
        )
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.heads * self.head_dim)
        x = self.out_proj(x)
        x = self.dropout(x)
        if residual.shape != x.shape:
            if self.input_t_proj is not None:
                residual = residual.permute(0, 2, 1)
                residual = self.input_t_proj(residual)
                residual = residual.permute(0, 2, 1)
            if self.input_proj is not None:
                residual = self.input_proj(residual)
        x += residual
        if self.mlp is not None:
            y = self.ln_2(x)
            y = self.mlp(y)
        return x + y if self.mlp is not None else self.ln_2(x)

class Block(torch.nn.Module):
    def __init__(
        self,
        heads: int,
        dim1: int,
        dim2: int,
        dim_step: int,
        mean_var_hidden: int,
        seq_length: int,
        seq_len_step: int,
        is_first_block: bool,
        seq_len_reduce:int,
        force_reduce: bool=False,
        out_features_override: int = None,
    ):
        super().__init__()
        self.is_first_block = is_first_block
        self.encoder = VMLA_Block(
            heads=heads,
            dim1=dim1,
            dim2=dim2,
            dim3=dim1,
            mean_var_hidden=mean_var_hidden,
            seq_length=seq_length,
            seq_len_reduce=seq_len_reduce,
            seq_len_new=seq_length,
            mlp_dim = dim1 * 2,
            force_reduce=force_reduce,
            use_mlp=True
        )
        self.decoder = VMLA_Block(
            heads=heads,
            dim1=dim1,
            dim2=dim2,
            dim3=dim1,
            mean_var_hidden=mean_var_hidden,
            seq_length=seq_length,
            seq_len_reduce=seq_len_reduce,
            seq_len_new=seq_length,
            mlp_dim=dim1 * 2,
            force_reduce=force_reduce,
            use_mlp=True
        )
        self.cross = VMLA_Block(
            heads=heads,
            dim1=dim1,
            dim2=dim2,
            dim3=(dim1 + (dim_step * 3)) if out_features_override is None else out_features_override, # Need to adjust output features if this is the last block in the encoder/decoder stack
            mean_var_hidden=mean_var_hidden,
            seq_length=seq_length,
            seq_len_reduce=seq_len_reduce,
            seq_len_new=seq_length + (seq_len_step * 3),
            mlp_dim=(dim1 + (dim_step * 3)) * 2,
            force_reduce=force_reduce,
            use_mlp=True
        )

    def forward(self, x, esm=None, dsm=None, csm=None, mask=True, embeddings=None):
        xq = x
        if self.is_first_block:
            xq = xq.permute(0, 2, 3, 1)
            xq = xq.reshape(xq.shape[0], xq.shape[1], xq.shape[2] * xq.shape[3])
            if embeddings is not None:
                xq = xq + embeddings[0]
        xq = self.encoder(xq, state_manager=esm, mask=mask)
        xkv = xq
        if self.is_first_block and embeddings is not None:
            xkv = xkv + embeddings[1]
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[1], 3).permute(0, 2, 1, 3)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[2] * xkv.shape[3])
        xkv = self.decoder(xkv, state_manager=dsm, mask=mask)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[1], 3).permute(0, 2, 1, 3)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[2] * xkv.shape[3])
        x = self.cross(xq, xkv, state_manager=csm, mask=mask)
        return x



# 8 total Encoder-Decoder Blocks, 24 attention layers, For Auto-Regressive Generation
# Defaults are for 256x256x3 images.
class EncoderDecoder_8(torch.nn.Module):
    def __init__(
        self,
        heads: int=12,
        dim1: int=768,
        dim2: int=256,
        dim_step: int=48,
        mean_var_hidden: int=192,
        seq_length: int=256,
        seq_len_step: int=16,
        seq_len_reduce:int=128,
        out_features_override: int = None, #Not used here, but useful for last block in encoder/decoder stacks
        force_reduce: bool=False,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.force_reduce = force_reduce
        # Encoder-Blocks
        self.encoder_blocks = torch.nn.ModuleList()
        for i in range(3):
            self.encoder_blocks.append(
                Block(
                    heads=heads,
                    dim1=dim1,
                    dim2=dim2,
                    dim_step=-dim_step,
                    mean_var_hidden=mean_var_hidden,
                    is_first_block=(i == 0),
                    seq_length=seq_length,
                    seq_len_step=-seq_len_step,
                    seq_len_reduce=seq_len_reduce,
                    out_features_override=None, # Not used here, but useful for last block in encoder/decoder stacks
                    force_reduce=force_reduce
                )
            )
            dim1 -= (dim_step * 3)
            seq_length -= (seq_len_step * 3)
        self.block_bottle_neck_1 = Block(
            heads=heads,
            dim1=dim1,
            dim2=dim2,
            dim_step=0,
            mean_var_hidden=mean_var_hidden,
            is_first_block=False,
            seq_length=seq_length,
            seq_len_step=0,
            seq_len_reduce=seq_len_reduce,
            out_features_override=None,
            force_reduce=force_reduce
        )
        # Decoder-Blocks
        self.block_bottle_neck_2 = Block(
            heads=heads,
            dim1=dim1,
            dim2=dim2,
            dim_step=0,
            mean_var_hidden=mean_var_hidden,
            is_first_block=False,
            seq_length=seq_length,
            seq_len_step=0,
            seq_len_reduce=seq_len_reduce,
            out_features_override=None,
            force_reduce=force_reduce
        )
        self.decoder_blocks = torch.nn.ModuleList()
        for i in range(3):
            self.decoder_blocks.append(
                Block(
                    heads=heads,
                    dim1=dim1,
                    dim2=dim2,
                    dim_step=dim_step,
                    mean_var_hidden=mean_var_hidden,
                    is_first_block=False,
                    seq_length=seq_length,
                    seq_len_step=seq_len_step,
                    seq_len_reduce=seq_len_reduce,
                    out_features_override=out_features_override if i == 2 else None,
                    force_reduce=force_reduce
                )
            )
            dim1 += (dim_step * 3)
            seq_length += (seq_len_step * 3)

    def forward(self, x, embeddings=None):
        esm = ResidualStateManager() if self.force_reduce else None # The encoder residual state manager will never be used if the reduction mechanism is not forced
        dsm = ResidualStateManager() if self.force_reduce else None # Same for the decoder
        csm = ResidualStateManager() # the cross residual state manager will always be used
        skip_1 = None
        skip_2 = None
        skip_bn_1 = None
        skip_bn_2 = None
        for i, block in enumerate(self.encoder_blocks):
            x = block(x, esm=esm, dsm=dsm, csm=csm, mask=True, embeddings=embeddings if embeddings is not None else None)
            if i == 0:
                skip_1 = x
            elif i == 1:
                skip_2 = x
            else:
                skip_bn_1 = x
        x = self.block_bottle_neck_1(x, esm=esm, dsm=dsm, csm=csm, mask=True) + skip_bn_1
        skip_bn_2 = x
        x = self.block_bottle_neck_2(x, esm=esm, dsm=dsm, csm=csm, mask=True) + skip_bn_1 + skip_bn_2
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, esm=esm, dsm=dsm, csm=csm, mask=True, embeddings=embeddings if embeddings is not None else None)
            if i == 0:
                x += skip_2
            elif i == 1:
                x += skip_1
        return x

# 8 Total Encoder-Decoder Blocks, 24 attention layers, For Classification, Defaults are for 224x224x3 images.
# Encoder Only.
class Encoder_8(torch.nn.Module):
    def __init__(
        self,
        heads: int=12,
        dim1: int=672,
        dim2: int=224,
        dim_step: int=24,
        mean_var_hidden: int=192,
        seq_length: int=224,
        seq_len_step: int=8,
        seq_len_reduce:int=96,
        out_features_override: int = None,
        force_reduce: bool=False,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.encoder_blocks = torch.nn.ModuleList()
        self.force_reduce = force_reduce
        for i in range(8):
            step = i % 2 == 0
            self.encoder_blocks.append(
                Block(
                    heads=heads,
                    dim1=dim1,
                    dim2=dim2,
                    dim_step=-dim_step if step else 0,
                    mean_var_hidden=mean_var_hidden,
                    is_first_block=(i == 0),
                    seq_length=seq_length,
                    seq_len_step=-seq_len_step if step else 0,
                    seq_len_reduce=seq_len_reduce,
                    out_features_override=None,
                    force_reduce=force_reduce
                )
            )
            dim1 -= (dim_step * 3) if step else 0
            seq_length -= (seq_len_step * 3) if step else 0
    
    def forward(self, x, embeddings=None):
        esm = ResidualStateManager() if self.force_reduce else None # The encoder residual state manager will never be used if the reduction mechanism is not forced
        dsm = ResidualStateManager() if self.force_reduce else None # Same for the decoder
        csm = ResidualStateManager() # the cross residual state manager will always be used
        for i, block in enumerate(self.encoder_blocks):
            x = block(x, esm=esm, dsm=dsm, csm=csm, mask=True, embeddings=embeddings if embeddings is not None else None)
        return x