import torch
from typing import Callable
from functools import partial
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as sn

class ResidualStateManager():
    def __init__(
        self,
        smooth_factor: float = 2.0, # Ignored unless using exponential moving average
        momentum: float = 0.9, # Ignored unless using static momentum
        mode: str = "ema" # modes for "sma" (simple moving average), "ema" (exponential moving average), "lp" (later priority), and "sum" for raw summation. Any other value defaults to static momentum.
    ):
        super().__init__()
        self.zq_sum = None
        self.zkv_sum = None
        self.tot_kl_loss = 0.0
        self.count = 0
        self.smooth_factor = smooth_factor
        self.mode = mode
        self.momentum = momentum

    def get_sums(self, zq, zkv, mean_q, var_q, mean_kv, var_kv):
        kl_q = -0.5 * torch.mean(1 + 2 * torch.log(var_q) - mean_q.pow(2) - var_q.pow(2))
        kl_kv = -0.5 * torch.mean(1 + 2 * torch.log(var_kv) - mean_kv.pow(2) - var_kv.pow(2))
        self.tot_kl_loss = kl_q + kl_kv + self.tot_kl_loss
        if self.zq_sum is None:
            self.zq_sum = zq
            self.zkv_sum = zkv 
            self.count = 1
        elif self.mode != "sum" and self.mode != "sma":
            # Moving average instead of sum
            self.count += 1
            if self.mode == "ema": # Exponential Moving Average, early layers are weighted more
                self.momentum = self.smooth_factor / (self.count + 1)
            elif self.mode == "lp": # Later Priority, later layers are weighted more
                self.momentum = self.count / (self.count + 1)
            self.zq_sum = (self.momentum * zq) + ((1 - self.momentum) * self.zq_sum)
            self.zkv_sum = (self.momentum * zkv) + ((1 - self.momentum) * self.zkv_sum)
        else:
            # Use out-of-place operations to avoid modifying views inplace
            self.count += 1
            self.zq_sum = self.zq_sum + zq
            self.zkv_sum = self.zkv_sum + zkv
            if self.mode == "sma":
                return self.zq_sum / self.count, self.zkv_sum / self.count
        return self.zq_sum, self.zkv_sum

    def get_kl_loss(self):
        return self.tot_kl_loss / self.count if self.count > 0 else 0.0

# Since our tokens are processed as sequences (row-wise, column-wise), we can apply RoPE as a 1D Embedding.
# I'm not sure if 2D would still be applicable in this case since tokens are not grid based. Using 1D should
# be fine since the 2D spatial relationships are implied by the tokenization strategy (?).
class RoPE(torch.nn.Module):
    def __init__(self, seq: int, dim: int, theta: float=10000.0, learned: bool=False, training: bool=True):
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
            self.inv_freq = torch.nn.Parameter(inv_freq, requires_grad=True)
            self.register_buffer("t", t, persistent=False)
        else:
            self.register_buffer("inv_freq", inv_freq)
            self.register_buffer("t", t, persistent=False)
            emb = torch.cat((self.freqs, self.freqs), dim=-1)
            self.register_buffer("cos_emb", emb.cos(), persistent=False)
            self.register_buffer("sin_emb", emb.sin(), persistent=False)
    
    def rotate_half(self, x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x):       
        if self.learned:
            t = self.t[:x.shape[2]].to(x.device)
            self.freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((self.freqs, self.freqs), dim=-1)
            cos = emb.cos().unsqueeze(0).unsqueeze(0)
            sin = emb.sin().unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_emb[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
            sin = self.sin_emb[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
        return (x * cos) + (self.rotate_half(x) * sin)

# Multi-Head Latent Distribution Attention
class VMLA_Block(torch.nn.Module):
    def __init__(
        self,
        heads: int,
        dim1: int,
        dim2: int,
        mean_var_hidden: int,
        seq_length: int,
        seq_len_reduce:int,
        seq_len_new:int,
        mlp_dim: int,
        force_reduce: bool=True, # Force reduction even if input and output dims / seq lengths are the same (NOT RECOMMENDED; REQUIRES A LOT MORE PARAMETERS)
        t_force_reduce: bool=False,
        dropout:float=0.0,
        use_mlp: bool=True,
        is_cross: bool=False,
        training: bool=True,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        # Layer Scaling Parameters (As if there isn't a ton of learnable parameters already...)
        self.ls_att = torch.nn.Parameter(torch.ones(dim2), requires_grad=True)
        self.ls_mlp = torch.nn.Parameter(torch.ones(dim2), requires_grad=True) if use_mlp else None
        self.training = training
        self.heads = heads
        self.head_dim_content = dim2 // heads // 2
        self.head_dim_rope = dim2 // heads // 2
        self.head_dim = self.head_dim_content + self.head_dim_rope
        #! This is ugly but I don't care :)
        # One-Shot Encoders
        self.t_reduce = seq_len_new != seq_length or t_force_reduce
        self.reduce = dim1 != dim2 or force_reduce
        # Mean and Variance Bottleneck
        self.ln_q = norm_layer(dim1, bias=False)
        self.ln_kv = norm_layer(dim1, bias=False) if is_cross else None
        self.t_encoder_q = None
        self.t_encoder_kv = None
        # Only apply token temporal reduction if seq_len_reduce is different from seq_length (or forced)
        if self.t_reduce:
            self.t_encoder_q = sn(torch.nn.Linear(seq_length, seq_len_reduce, bias=False))
            self.t_encoder_kv = sn(torch.nn.Linear(seq_length, seq_len_reduce, bias=False))
        # Only compute the mean and variance projections if input and output dims differ
        self.encoder_q = None
        self.encoder_kv = None
        if self.reduce:
            self.encoder_q = sn(torch.nn.Linear(dim1, mean_var_hidden * 2, bias=False))
            self.encoder_kv = sn(torch.nn.Linear(dim1, mean_var_hidden * 2, bias=False))
        # Decoders
        # Similarly, only apply token temporal upsampling if seq_len_reduce is different from seq_len_new
        self.t_qz_upsample = None
        self.t_kz_upsample = None
        self.t_vz_upsample = None
        self.t_qr_proj = None
        self.t_kr_proj = None
        if self.t_reduce:
            self.t_qz_upsample = sn(torch.nn.Linear(seq_len_reduce, seq_len_new, bias=False))
            self.t_kz_upsample = sn(torch.nn.Linear(seq_len_reduce, seq_len_new, bias=False))
            self.t_vz_upsample = sn(torch.nn.Linear(seq_len_reduce, seq_len_new, bias=False))
            self.t_qr_proj = sn(torch.nn.Linear(seq_len_reduce, seq_len_new, bias=False))
            self.t_kr_proj = sn(torch.nn.Linear(seq_length, seq_len_new, bias=False))
        # Only compute the upsample projections if input and output dims differ
        self.qz_upsample = None
        self.kz_upsample = None
        self.vz_upsample = None
        # Projections
        self.q_proj = sn(torch.nn.Linear(dim2 if dim1 == dim2 and not force_reduce else mean_var_hidden,
                                      (self.heads * self.head_dim_content) if self.reduce else (self.heads * self.head_dim),
                                      bias=False)
        )
        self.k_proj = sn(torch.nn.Linear(dim2 if dim1 == dim2 and not force_reduce else mean_var_hidden,
                                      (self.heads * self.head_dim_content) if self.reduce else (self.heads * self.head_dim),
                                      bias=False)
        )
        self.v_proj = sn(torch.nn.Linear(dim2 if dim1 == dim2 and not force_reduce else mean_var_hidden, dim2, bias=False))
        # Decoupled RoPE Projections as per Multi-Head Latent Attention Paper
        self.qr_proj = None
        self.kr_proj = None
        if self.reduce:
            self.qr_proj = sn(torch.nn.Linear(mean_var_hidden, self.head_dim_rope * self.heads, bias=False))
            self.kr_proj = sn(torch.nn.Linear(dim1, self.head_dim_rope * self.heads, bias=False))
        self.input_t_proj = None
        self.input_proj = None
        # Dimension Adjustments, Do not force reduction here, we won't modify seq_length or dim1 unless necessary
        # This isn't optimal but closer to how residual connections are supposed to work.
        if seq_len_new != seq_length:
            self.input_t_proj = sn(torch.nn.Linear(seq_length, seq_len_new, bias=False))
        if dim1 != dim2:
            self.input_proj = sn(torch.nn.Linear(dim1, dim2, bias=False))
        # Attention
        self.rope_q = RoPE(seq_len_new, self.head_dim_rope if self.reduce else self.head_dim, learned=True)
        self.rope_k = RoPE(seq_len_new, self.head_dim_rope if self.reduce else self.head_dim, learned=True)
        self.linear_mask = torch.nn.Sequential(
            sn(torch.nn.Linear(seq_len_new, seq_len_new * 2, bias=True)),
            torch.nn.GELU(approximate='none'),
            sn(torch.nn.Linear(seq_len_new * 2, seq_len_new, bias=True)),
            # torch.nn.Tanh()
        )
        self.out_proj = sn(torch.nn.Linear(dim2, dim2, bias=False))
        self.dropout = torch.nn.Dropout(dropout)
        self.ln_2 = norm_layer(dim2, bias=False)
        self.mlp = None
        if use_mlp:
            self.mlp = torch.nn.Sequential(
                sn(torch.nn.Linear(dim2, mlp_dim, bias=False)),
                torch.nn.GELU(approximate='none'),
                torch.nn.Dropout(dropout, inplace=False),
                sn(torch.nn.Linear(mlp_dim, dim2, bias=False)),
            )

    def forward(self, input_q, input_kv=None, state_manager=None, mask=False):
        # One-Shot Encoders
        residual = input_q
        if input_kv is None:
            xq = self.ln_q(input_q)
            xkv = xq
        else:
            xq = self.ln_q(input_q)
            xkv = self.ln_kv(input_kv)
         # Mean and Variance Bottleneck
        qz = xq
        kz = xkv
        vz = xkv
        qr = xq
        kr = xkv
        if self.reduce:
            if self.t_reduce:
                xq = xq.permute(0, 2, 1)
                xkv = xkv.permute(0, 2, 1)
                xq = self.t_encoder_q(xq)
                xkv = self.t_encoder_kv(xkv)
                xq = xq.permute(0, 2, 1)
                xkv = xkv.permute(0, 2, 1)
            mean_var_q = self.encoder_q(xq)
            mean_var_kv = self.encoder_kv(xkv)
            mean_zq, var_zq_raw = mean_var_q.chunk(2, dim=-1)
            mean_zkv, var_zkv_raw = mean_var_kv.chunk(2, dim=-1)
            var_zq = F.softplus(var_zq_raw) + 1e-6
            var_zkv = F.softplus(var_zkv_raw) + 1e-6
            # Compute samples
            if self.training:
                zq = mean_zq + torch.randn_like(var_zq) * var_zq
                zkv = mean_zkv + torch.randn_like(var_zkv) * var_zkv
            else:
                zq = mean_zq
                zkv = mean_zkv
            if state_manager is not None:
                zq, zkv = state_manager.get_sums(zq, zkv, mean_zq, var_zq, mean_zkv, var_zkv)
            qr = zq
            qz = zq
            kz = zkv
            vz = zkv
            if self.t_reduce:
                qz = qz.permute(0, 2, 1)
                qz = self.t_qz_upsample(qz)
                qz = qz.permute(0, 2, 1)
                kz = kz.permute(0, 2, 1)
                kz = self.t_kz_upsample(kz)
                kz = kz.permute(0, 2, 1)
                vz = vz.permute(0, 2, 1)
                vz = self.t_vz_upsample(vz)
                vz = vz.permute(0, 2, 1)
                qr = qr.permute(0, 2, 1)
                qr = self.t_qr_proj(qr)
                qr = qr.permute(0, 2, 1)
                kr = kr.permute(0, 2, 1)
                kr = self.t_kr_proj(kr)
                kr = kr.permute(0, 2, 1)
        qz = self.q_proj(qz)
        kz = self.k_proj(kz)
        vz = self.v_proj(vz)
        batch_size = qz.shape[0]
        seq_len_q = qz.shape[1]
        seq_len_kv = kz.shape[1]
        q = qz.view(batch_size, seq_len_q, self.heads, self.head_dim_content if self.reduce else self.head_dim).transpose(1, 2)
        k = kz.view(batch_size, seq_len_kv, self.heads, self.head_dim_content if self.reduce else self.head_dim).transpose(1, 2)
        v = vz.view(batch_size, seq_len_kv, self.heads, self.head_dim).transpose(1, 2)
        # Decoupled RoPE if we are reducing dimensions
        if self.reduce:
            qr = self.qr_proj(qr)
            kr = self.kr_proj(kr)
            qr = qr.view(batch_size, seq_len_q, self.heads, self.head_dim_rope).transpose(1, 2)
            kr = kr.view(batch_size, seq_len_kv, self.heads, self.head_dim_rope).transpose(1, 2)
            q = torch.cat((q, self.rope_q(qr)), dim=-1)
            k = torch.cat((k, self.rope_k(kr)), dim=-1)
        # Standard RoPE if not reducing
        else:
            q = self.rope_q(q)
            k = self.rope_k(k)
        # Techichally we are computing QK^T twice here, once for attention and once for the mask,
        # Inefficent but we want to leverage PyTorch's optimized scaled_dot_product_attention function.
        q_mask = q.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.heads * self.head_dim)
        k_mask = k.transpose(1, 2).contiguous().view(batch_size, seq_len_kv, self.heads * self.head_dim)
        mask_mat = self.linear_mask(q_mask @ k_mask.permute(0, 2, 1)) if mask else None
        mask_mat = mask_mat.unsqueeze(1)
        # Attention
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask_mat if mask else None,
            dropout_p=0.0,
            is_causal=False
        )
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.heads * self.head_dim)
        x = self.out_proj(x) * self.ls_att
        x = self.dropout(x)
        if residual.shape != x.shape:
            if self.input_t_proj is not None:
                residual = residual.permute(0, 2, 1)
                residual = self.input_t_proj(residual)
                residual = residual.permute(0, 2, 1)
            if self.input_proj is not None:
                residual = self.input_proj(residual)
        x = x + residual
        if self.mlp is not None:
            y = self.ln_2(x)
            y = self.mlp(y)
            if self.ls_mlp is not None:
                y = y * self.ls_mlp
        return x + y if self.mlp is not None else self.ln_2(x)

class Block(torch.nn.Module):
    def __init__(
        self,
        heads: int,
        dim1: int,
        dim_step: int,
        mean_var_hidden: int,
        seq_length: int,
        seq_len_step: int,
        is_first_block: bool,
        is_last_block: bool,
        seq_len_reduce:int,
        force_reduce: bool=False,
        training: bool=True,
        use_ape: bool=True,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),
        out_features_override: int = None,
    ):
        super().__init__()
        self.is_first_block = is_first_block
        self.encoder = VMLA_Block(
            heads=heads,
            dim1=dim1,
            dim2=dim1,
            mean_var_hidden=mean_var_hidden,
            seq_length=seq_length,
            seq_len_reduce=seq_len_reduce,
            seq_len_new=seq_length,
            mlp_dim = dim1 * 2,
            force_reduce=force_reduce,
            training=training,
            use_mlp=True
        )
        self.decoder = VMLA_Block(
            heads=heads,
            dim1=dim1,
            dim2=dim1,
            mean_var_hidden=mean_var_hidden,
            seq_length=seq_length,
            seq_len_reduce=seq_len_reduce,
            seq_len_new=seq_length,
            mlp_dim=dim1 * 2,
            force_reduce=force_reduce,
            training=training,
            use_mlp=True
        )
        self.cross = VMLA_Block(
            heads=heads,
            dim1=dim1,
            dim2=(dim1 + (dim_step * 3)) if out_features_override is None else out_features_override, # Need to adjust output features if this is the last block in the encoder/decoder stack
            mean_var_hidden=mean_var_hidden,
            seq_length=seq_length,
            seq_len_reduce=seq_len_reduce,
            seq_len_new=seq_length + (seq_len_step * 3),
            mlp_dim=(dim1 + (dim_step * 3)) * 2,
            force_reduce=force_reduce,
            is_cross=True,
            training=training,
            use_mlp=True
        )
        #  we CNN now
        hidden_channels = 32
        self.proj = torch.nn.Sequential(
            sn(torch.nn.Conv2d(3, hidden_channels, kernel_size=1, groups=1, bias=True)),
            torch.nn.GELU(approximate='none'),
            sn(torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True, groups=hidden_channels, padding_mode='zeros')),
            torch.nn.GELU(approximate='none'),
            sn(torch.nn.Conv2d(hidden_channels, 3, kernel_size=1, bias=True))
        )

    def forward(self, x, esm=None, dsm=None, csm=None, mask=True):
        xq = x
        if self.is_first_block:
            xq = xq.permute(0, 2, 3, 1)
            xq = xq.reshape(xq.shape[0], xq.shape[1], xq.shape[2] * xq.shape[3])
        xq = self.encoder(xq, state_manager=esm, mask=mask)
        xkv = xq
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[1], 3).permute(0, 2, 1, 3)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[2] * xkv.shape[3])
        xkv = self.decoder(xkv, state_manager=dsm, mask=mask)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[1], 3).permute(0, 2, 1, 3)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[2] * xkv.shape[3])
        x = self.cross(xq, input_kv=xkv, state_manager=csm, mask=mask)
        x_img = self.proj(x.reshape(x.shape[0], x.shape[1], x.shape[1], 3).permute(0, 3, 1, 2))
        x_img = x_img.permute(0, 2, 3, 1)
        x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], x_img.shape[2] * x_img.shape[3])
        return x + x_img

# 8 total Encoder-Decoder Blocks, 24 attention layers, For Auto-Regressive Generation
# Defaults are for 256x256x3 images.
class EncoderDecoder_8(torch.nn.Module):
    def __init__(
        self,
        heads: int=12,
        dim1: int=768,
        dim_step: int=48,
        mean_var_hidden: int=192,
        seq_length: int=256,
        seq_len_step: int=16,
        seq_len_reduce:int=128,
        out_features_override: int = None, #Not used here, but useful for last block in encoder/decoder stacks
        force_reduce: bool=False,
        training: bool=True,
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
                    dim_step=-dim_step,
                    mean_var_hidden=mean_var_hidden,
                    is_first_block=(i == 0),
                    is_last_block=False,
                    seq_length=seq_length,
                    seq_len_step=-seq_len_step,
                    seq_len_reduce=seq_len_reduce,
                    out_features_override=None, # Not used here, but useful for last block in encoder/decoder stacks
                    force_reduce=force_reduce,
                    training=training
                )
            )
            dim1 -= (dim_step * 3)
            seq_length -= (seq_len_step * 3)
        self.block_bottle_neck_1 = Block(
            heads=heads,
            dim1=dim1,
            dim_step=0,
            mean_var_hidden=mean_var_hidden,
            is_first_block=False,
            is_last_block=False,
            seq_length=seq_length,
            seq_len_step=0,
            seq_len_reduce=seq_len_reduce,
            out_features_override=None,
            force_reduce=force_reduce,
            training=training
        )
        # Decoder-Blocks
        self.block_bottle_neck_2 = Block(
            heads=heads,
            dim1=dim1,
            dim_step=0,
            mean_var_hidden=mean_var_hidden,
            is_first_block=False,
            is_last_block=False,
            seq_length=seq_length,
            seq_len_step=0,
            seq_len_reduce=seq_len_reduce,
            out_features_override=None,
            force_reduce=force_reduce,
            training=training
        )
        self.decoder_blocks = torch.nn.ModuleList()
        for i in range(3):
            self.decoder_blocks.append(
                Block(
                    heads=heads,
                    dim1=dim1,
                    dim_step=dim_step,
                    mean_var_hidden=mean_var_hidden,
                    is_first_block=False,
                    is_last_block=(i == 2),
                    seq_length=seq_length,
                    seq_len_step=seq_len_step,
                    seq_len_reduce=seq_len_reduce,
                    out_features_override=out_features_override if i == 2 else None,
                    force_reduce=force_reduce,
                    training=training
                )
            )
            dim1 += (dim_step * 3)
            seq_length += (seq_len_step * 3)
        self.ln_final = norm_layer(dim1, bias=False)

    def forward(self, x):
        esm = ResidualStateManager(mode="sum") if self.force_reduce else None # The encoder residual state manager will never be used if the reduction mechanism is not forced
        dsm = ResidualStateManager(mode="sum") if self.force_reduce else None # Same for the decoder
        csm = ResidualStateManager(mode="sum") # the cross residual state manager will always be used
        skip_1 = None
        skip_2 = None
        skip_bn_1 = None
        skip_bn_2 = None
        for i, block in enumerate(self.encoder_blocks):
            x = block(x, esm=esm, dsm=dsm, csm=csm, mask=True)
            if i == 0:
                skip_1 = x
            elif i == 1:
                skip_2 = x
            else:
                skip_bn_1 = x
        x = self.block_bottle_neck_1(x, esm=esm, dsm=dsm, csm=csm, mask=True)
        x += skip_bn_1
        skip_bn_2 = x
        x = self.block_bottle_neck_2(x, esm=esm, dsm=dsm, csm=csm, mask=True)
        x += skip_bn_2 + skip_bn_1
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, esm=esm, dsm=dsm, csm=csm, mask=True)
            if i == 0:
                x += skip_2
            elif i == 1:
                x += skip_1
        x = self.ln_final(x)
        # final_mean_zq, final_var_zq, final_mean_zkv, final_var_zkv = csm.mean_q_sum, csm.var_q_sum, csm.mean_kv_sum, csm.var_kv_sum
        # final_mean_zq = final_mean_zq
        # final_var_zq = final_var_zq
        # final_mean_zkv = final_mean_zkv
        # final_var_zkv = final_var_zkv
        # kl_q = -0.5 * torch.mean(1 + final_var_zq - final_mean_zq.pow(2) - torch.exp(final_var_zq))
        # kl_kv = -0.5 * torch.mean(1 + final_var_zkv - final_mean_zkv.pow(2) - torch.exp(final_var_zkv))
        kl_loss = csm.get_kl_loss()
        kl_loss = esm.get_kl_loss() + dsm.get_kl_loss() + kl_loss if self.force_reduce else kl_loss
        return x, kl_loss
    
class CALMLatentDiffusion(torch.nn.Module):
    def __init__(
        self,
        heads: int=12,
        dim1: int=672,
        dim_step: int=48,
        mean_var_hidden: int=204,
        mean_var_hidden_diffusion: int=96,
        seq_length: int=224,
        seq_len_step: int=16,
        seq_len_reduce:int=80,
        seq_len_reduce_diffusion:int=32,
        out_features_override: int = None, #Not used here, but useful for last block in encoder/decoder stacks
        force_reduce: bool=False,
        training: bool=True,
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
                    dim_step=-dim_step,
                    mean_var_hidden=mean_var_hidden,
                    is_first_block=(i == 0),
                    is_last_block=False,
                    seq_length=seq_length,
                    seq_len_step=-seq_len_step,
                    seq_len_reduce=seq_len_reduce,
                    out_features_override=None, # Not used here, but useful for last block in encoder/decoder stacks
                    force_reduce=force_reduce,
                    training=training
                )
            )
            dim1 -= (dim_step * 3)
            seq_length -= (seq_len_step * 3)
        self.decoder_blocks = torch.nn.ModuleList()
        for i in range(3):
            self.decoder_blocks.append(
                Block(
                    heads=heads,
                    dim1=dim1,
                    dim_step=dim_step,
                    mean_var_hidden=mean_var_hidden,
                    is_first_block=False,
                    is_last_block=(i == 2),
                    seq_length=seq_length,
                    seq_len_step=seq_len_step,
                    seq_len_reduce=seq_len_reduce,
                    out_features_override=out_features_override if i == 2 else None,
                    force_reduce=force_reduce,
                    training=training
                )
            )
            dim1 += (dim_step * 3)
            seq_length += (seq_len_step * 3)
        self.ln_final = norm_layer(dim1, bias=False)


# 8 Total Encoder-Decoder Blocks, 24 attention layers, For Classification, Defaults are for 224x224x3 images.
# Encoder Only.
class Encoder_8(torch.nn.Module):
    def __init__(
        self,
        heads: int=12,
        dim1: int=672,
        dim_step: int=24,
        mean_var_hidden: int=192,
        seq_length: int=224,
        seq_len_step: int=8,
        seq_len_reduce:int=96,
        out_features_override: int = None,
        force_reduce: bool=False,
        training: bool=True,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.encoder_blocks = torch.nn.ModuleList()
        self.force_reduce = force_reduce
        seq_length_initial = seq_length
        dim1_initial = dim1
        for i in range(8):
            step = i == 2 or i == 5
            self.encoder_blocks.append(
                Block(
                    heads=heads,
                    dim1=dim1,
                    dim_step=-dim_step if step else 0,
                    mean_var_hidden=mean_var_hidden,
                    is_first_block=(i == 0),
                    is_last_block=(i == 7),
                    seq_length=seq_length,
                    seq_len_step=-seq_len_step if step else 0,
                    seq_len_reduce=seq_len_reduce,
                    out_features_override=None,
                    force_reduce=force_reduce,
                    training=training
                )
            )
            dim1 -= (dim_step * 3) if step else 0
            seq_length -= (seq_len_step * 3) if step else 0
        self.ln_final = norm_layer(dim1, bias=False)
    
    def forward(self, x):
        # Do not use state managers for classification encoder, since each layer should learn independent representations.
        esm = None
        dsm = None
        csm = None
        skip = None
        for i, block in enumerate(self.encoder_blocks):
            x = block(x, esm=esm, dsm=dsm, csm=csm, mask=True)
            if skip is None or x.shape != skip.shape:
                skip = x
            else:
                x = x + skip
                skip = x
        x = self.ln_final(x)
        return x