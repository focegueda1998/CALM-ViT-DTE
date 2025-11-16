import torch
from typing import Callable
from functools import partial

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
        static_epsilon_weight: bool=False,
        e2de_ratio: float=1.0,
        dropout:float=0.0,
        attention_dropout=0.0,
        use_mlp: bool=True,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.heads = heads
        self.ln_q = norm_layer(dim1, bias=False)
        self.ln_kv = norm_layer(dim1, bias=False)
        #! This is ugly but I don't care :)
        # One-Shot Encoders
        self.e2de_ratio = e2de_ratio
        self.static_epsilon_weight = static_epsilon_weight
        if not static_epsilon_weight:
            self.e2de_ratio = torch.nn.Parameter(torch.tensor([e2de_ratio]), requires_grad=True)
        # Cheating by expressing epsilon as a learnable parameter
        self.dirty_epsilon_zq = torch.nn.Parameter(torch.randn(1, seq_len_reduce, mean_var_hidden), requires_grad=True)
        self.dirty_epsilon_zkv = torch.nn.Parameter(torch.randn(1, seq_len_reduce, mean_var_hidden), requires_grad=True)
        # Mean and Variance Bottleneck
        #! Since our query window can differ in size, only compute the head reduction if it's not the first block
        #! or our QKV values are the same. We can probably fix the sequence length to be the same for all blocks.
        #! But the hypothetical peformance gain is minimal.
        self.t_mean_zq = None
        self.t_var_zq = None
        self.t_mean_zq = torch.nn.Sequential(
            torch.nn.Linear(seq_length, seq_len_reduce, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.t_var_zq = torch.nn.Sequential(
            torch.nn.Linear(seq_length, seq_len_reduce, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.qkv_ratio = None
        self.t_mean_zkv = torch.nn.Sequential(
            torch.nn.Linear(seq_length, seq_len_reduce, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.t_var_zkv = torch.nn.Sequential(
            torch.nn.Linear(seq_length, seq_len_reduce, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.mean_zq = torch.nn.Sequential(
            torch.nn.Linear(dim1, dim2, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
            torch.nn.Linear(dim2, mean_var_hidden, bias=False),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.var_zq = torch.nn.Sequential(
            torch.nn.Linear(dim1, dim2, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
            torch.nn.Linear(dim2, mean_var_hidden, bias=False),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.mean_zkv = torch.nn.Sequential(
            torch.nn.Linear(dim1, dim2, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
            torch.nn.Linear(dim2, mean_var_hidden, bias=False),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.var_zkv = torch.nn.Sequential(
            torch.nn.Linear(dim1, dim2, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
            torch.nn.Linear(dim2, mean_var_hidden, bias=False),
            torch.nn.Dropout(dropout, inplace=False),
        )
        # Decoders
        self.t_qz_upsample = torch.nn.Sequential(
            torch.nn.Linear(seq_len_reduce, seq_len_new, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
        )
        self.t_kz_upsample = torch.nn.Sequential(
            torch.nn.Linear(seq_len_reduce, seq_len_new, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
        )
        self.t_vz_upsample = torch.nn.Sequential(
            torch.nn.Linear(seq_len_reduce, seq_len_new, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
        )
        self.qz_upsample_1 = torch.nn.Sequential( #! Since pytorch expects embed_dim to match query_dim, we need to upsample the query to dim3
            torch.nn.Linear(mean_var_hidden, dim2, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.kz_upsample = torch.nn.Sequential( #! Luckily, key and value can have different dimensions
            torch.nn.Linear(mean_var_hidden, dim2, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.vz_upsample = torch.nn.Sequential(
            torch.nn.Linear(mean_var_hidden, dim2, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.linear_mask = torch.nn.Sequential(
            torch.nn.Linear(seq_len_new, seq_len_new * 2, bias=False),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
            torch.nn.Linear(seq_len_new * 2, seq_len_new, bias=False),
            torch.nn.Dropout(dropout, inplace=False),
        )
        self.qz_upsample_2 = torch.nn.Sequential(
            torch.nn.Linear(dim2, dim3, bias=False),
            torch.nn.Dropout(dropout, inplace=False)
        )
        # Attention
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=dim3,
            num_heads=heads,
            dropout=attention_dropout,
            kdim=dim2,
            vdim=dim2,
            bias=False,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.ln_2 = norm_layer(dim3, bias=False)
        self.mlp = None
        if use_mlp:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dim3, mlp_dim, bias=False),
                torch.nn.GELU(approximate='none'),
                torch.nn.Dropout(dropout, inplace=False),
                torch.nn.Linear(mlp_dim, dim3, bias=False),
                torch.nn.Dropout(dropout, inplace=False)
            )

    def forward(self, input_q, input_kv=None, state_manager=None, mask=False):
        # One-Shot Encoders
        input_kv = input_q if input_kv is None else input_kv
        xq = input_q
        xkv = input_kv
        xq = self.ln_q(xq)
        xkv = self.ln_kv(xkv)
        if self.t_mean_zq:
            xq = xq.permute(0, 2, 1)
            mean_zq = self.t_mean_zq(xq)
            var_zq = self.t_var_zq(xq)
            mean_zq = mean_zq.permute(0, 2, 1)
            var_zq = var_zq.permute(0, 2, 1)
        else:
            mean_zq = xq
            var_zq = xq
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
        # Decoders
        if self.t_mean_zq:
            qz = zq.permute(0, 2, 1)
            qz = self.t_qz_upsample(qz)
            qz = qz.permute(0, 2, 1)
        zkv = zkv.permute(0, 2, 1)
        kz = self.t_kz_upsample(zkv)
        kz = kz.permute(0, 2, 1)
        vz = self.t_vz_upsample(zkv)
        vz = vz.permute(0, 2, 1)
        qz = self.qz_upsample_1(qz if self.t_mean_zq else zq)
        kz = self.kz_upsample(kz)
        vz = self.vz_upsample(vz)
        mask_mat = self.linear_mask(qz @ kz.permute(0, 2, 1)) if mask else None
        mask_mat = mask_mat.reshape(mask_mat.shape[0], 1, mask_mat.shape[1], mask_mat.shape[2]).repeat(1, self.heads, 1, 1)
        mask_mat = mask_mat.reshape(mask_mat.shape[0] * mask_mat.shape[1] , mask_mat.shape[2], mask_mat.shape[3])
        qz = self.qz_upsample_2(qz)
        # Attention
        x, _ = self.attention(qz, kz, vz, attn_mask=mask_mat)
        x = self.dropout(x)
        x = x + qz
        if x.shape == input_q.shape: x += input_q
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
        is_last_block: bool,
        seq_len_reduce:int,
        class_token: bool
    ):
        super().__init__()
        self.is_first_block = is_first_block
        self.class_token = class_token
        if is_first_block and class_token:
            self.adv_xq = torch.nn.Parameter(torch.randn(1, 1, dim1), requires_grad=True)
            self.adv_xkv = torch.nn.Parameter(torch.randn(1, 1, dim1), requires_grad=True)
            seq_length -= 1
        self.time_embedding_xq = torch.nn.Parameter(torch.randn(1, 1, dim1), requires_grad=True)
        self.encoder= VMLA_Block(
            heads=heads,
            dim1=dim1,
            dim2=dim2,
            dim3=dim1,
            mean_var_hidden=mean_var_hidden,
            seq_length=seq_length,
            seq_len_reduce=seq_len_reduce,
            seq_len_new=seq_length,
            mlp_dim = dim1 * 2,
            use_mlp=True
        )
        self.time_embedding_xkv = torch.nn.Parameter(torch.randn(1, 1, dim1), requires_grad=True)
        self.decoder = VMLA_Block(
            heads=heads,
            dim1=dim1,
            dim2=dim2,
            dim3=dim1,
            mean_var_hidden=mean_var_hidden,
            seq_length=seq_length,
            seq_len_reduce=seq_len_reduce,
            seq_len_new=seq_length,
            mlp_dim= dim1 * 2,
            use_mlp=True
        )
        if is_first_block:
            seq_length += 1
        self.cross= VMLA_Block(
            heads=heads,
            dim1=dim1,
            dim2=dim2,
            dim3=(dim1 + (dim_step * 3)),
            mean_var_hidden=mean_var_hidden,
            seq_length=seq_length,
            seq_len_reduce=seq_len_reduce,
            #! Stupid hack but we need a way to reduce and induce the class token while maintaing dimensional divisibility
            seq_len_new=seq_length + (seq_len_step * 3) - (1 if is_first_block and class_token else -1 if is_last_block and class_token else 0),
            mlp_dim=(dim1 + (dim_step * 3)) * 2,
            use_mlp=True
        )

    def forward(self, x, esm=None, dsm=None, csm=None, mask=True):
        xq = x
        if self.is_first_block:
            xq = xq.permute(0, 2, 3, 1)
            xq = xq.reshape(xq.shape[0], xq.shape[1], xq.shape[2] * xq.shape[3])
        xq += self.time_embedding_xq
        xq = self.encoder(xq, state_manager=esm, mask=mask)
        xkv = xq
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[1], 3).permute(0, 2, 1, 3)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[2] * xkv.shape[3]) + self.time_embedding_xkv
        xkv = self.decoder(xkv, state_manager=dsm, mask=mask)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[1], 3).permute(0, 2, 1, 3)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[2] * xkv.shape[3])
        if self.is_first_block and self.class_token:
            adv_xq = self.adv_xq.expand(xq.shape[0], -1, -1)
            xq = torch.cat((adv_xq, xq), dim=1)
            adv_xkv = self.adv_xkv.expand(xkv.shape[0], -1, -1)
            xkv = torch.cat((adv_xkv, xkv), dim=1)
        x = self.cross(xq, xkv, state_manager=csm, mask=mask)
        return x

# 2 total Encoder-Decoder Blocks, 6 attention layers
# Defaults are for 256x256x3 images.

# 4 total Encoder-Decoder Blocks, 12 attention layers
# Defaults are for 256x256x3 images.

# 8 total Encoder-Decoder Blocks, 24 attention layers
# Defaults are for 256x256x3 images.
class EncoderDecoder_8(torch.nn.Module):
    def __init__(
        self,
        heads: int=16,
        dim1: int=768,
        dim2: int=256,
        dim_step: int=48,
        mean_var_hidden: int=192,
        seq_length: int=257,
        seq_len_step: int=16,
        seq_len_reduce:int=128,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
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
                    is_last_block=False,
                    seq_length=seq_length,
                    seq_len_step=-seq_len_step,
                    seq_len_reduce=seq_len_reduce,
                    class_token=True
                )
            )
            dim1 -= (dim_step * 3)
            seq_length -= (seq_len_step * 3) + (1 if i == 0 else 0)
        self.block_bottle_neck_1 = Block(
            heads=heads,
            dim1=dim1,
            dim2=dim2,
            dim_step=0,
            mean_var_hidden=mean_var_hidden,
            is_first_block=False,
            is_last_block=False,
            seq_length=seq_length,
            seq_len_step=0,
            seq_len_reduce=seq_len_reduce,
            class_token=True
        )
        # Decoder-Blocks
        self.block_bottle_neck_2 = Block(
            heads=heads,
            dim1=dim1,
            dim2=dim2,
            dim_step=0,
            mean_var_hidden=mean_var_hidden,
            is_first_block=False,
            is_last_block=False,
            seq_length=seq_length,
            seq_len_step=0,
            seq_len_reduce=seq_len_reduce,
            class_token=True
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
                    is_last_block=(i == 2),
                    seq_length=seq_length,
                    seq_len_step=seq_len_step,
                    seq_len_reduce=seq_len_reduce,
                    class_token=True
                )
            )
            dim1 += (dim_step * 3)
            seq_length += (seq_len_step * 3)
        self.ln = norm_layer(dim1, bias=False)

    def forward(self, x):
        esm = ResidualStateManager()
        dsm = ResidualStateManager()
        csm = ResidualStateManager()
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
        x = self.block_bottle_neck_1(x, esm=esm, dsm=dsm, csm=csm, mask=True) + skip_bn_1
        skip_bn_2 = x
        x = self.block_bottle_neck_2(x, esm=esm, dsm=dsm, csm=csm, mask=True) + skip_bn_1 + skip_bn_2
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, esm=esm, dsm=dsm, csm=csm, mask=True)
            if i == 0:
                x += skip_2
            elif i == 1:
                x += skip_1
        x = self.ln(x)
        return x

# 12 total Encoder-Decoder Blocks, 36 attention layers
# Defaults are for 256x256x3 images.