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
    
    def get_sums(self, mean_q, var_q, mean_kv, var_kv):
        self.mean_q_sum = mean_q if self.mean_q_sum is None else self.mean_q_sum + mean_q
        self.var_q_sum = var_q if self.var_q_sum is None else self.var_q_sum + var_q
        self.mean_kv_sum = mean_kv if self.mean_kv_sum is None else self.mean_kv_sum + mean_kv
        self.var_kv_sum = var_kv if self.var_kv_sum is None else self.var_kv_sum + var_kv
        return self.mean_q_sum, self.var_q_sum, self.mean_kv_sum, self.var_kv_sum

# Multi-Head Latent Distribution Attention
class MLDA_Block(torch.nn.Module):
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
        constrain_epsilon: bool=True,
        e2de_ratio: float=1.0,
        dropout:float=0.0,
        attention_dropout=0.0,
        use_mlp: bool=True,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.heads = heads
        self.ln_q = norm_layer(dim1)
        self.ln_kv = norm_layer(dim1)
        #! This is ugly but I don't care :)
        # One-Shot Encoders
        self.e2de_ratio = e2de_ratio
        self.static_epsilon_weight = static_epsilon_weight
        self.constrain_epsilon = constrain_epsilon
        if not static_epsilon_weight:
            self.e2de_ratio = torch.nn.Parameter(torch.tensor([e2de_ratio]), requires_grad=True)
            self.sigmoid = torch.nn.Tanh()
        self.tanh_1 = torch.nn.Tanh()
        self.tanh_2 = torch.nn.Tanh()
        # Cheating by expressing epsilon as a learnable parameter
        self.dirty_epsilon_cq = torch.nn.Parameter(torch.randn(1, seq_len_reduce, mean_var_hidden), requires_grad=True)
        self.dirty_epsilon_ckv = torch.nn.Parameter(torch.randn(1, seq_len_reduce, mean_var_hidden), requires_grad=True)
        # Mean and Variance Bottleneck
        #! Since our query window can differ in size, only compute the head reduction if it's not the first block
        #! or our QKV values are the same. We can probably fix the sequence length to be the same for all blocks.
        #! But the hypothetical peformance gain is minimal.
        self.t_mean_cq = None
        self.t_var_cq = None
        self.t_mean_cq = torch.nn.Sequential(
        torch.nn.Linear(seq_length, seq_len_reduce, bias=True),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.t_var_cq = torch.nn.Sequential(
            torch.nn.Linear(seq_length, seq_len_reduce, bias=True),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.qkv_ratio = None
        self.t_mean_ckv = torch.nn.Sequential(
            torch.nn.Linear(seq_length, seq_len_reduce, bias=True),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.t_var_ckv = torch.nn.Sequential(
            torch.nn.Linear(seq_length, seq_len_reduce, bias=True),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.mean_cq = torch.nn.Sequential(
            torch.nn.Linear(dim1, dim2, bias=True),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
            torch.nn.Linear(dim2, mean_var_hidden, bias=True),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.var_cq = torch.nn.Sequential(
            torch.nn.Linear(dim1, dim2, bias=True),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
            torch.nn.Linear(dim2, mean_var_hidden, bias=True),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.mean_ckv = torch.nn.Sequential(
            torch.nn.Linear(dim1, dim2, bias=True),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
            torch.nn.Linear(dim2, mean_var_hidden, bias=True),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.var_ckv = torch.nn.Sequential(
            torch.nn.Linear(dim1, dim2, bias=True),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
            torch.nn.Linear(dim2, mean_var_hidden, bias=True),
            torch.nn.Dropout(dropout, inplace=False),
        )
        # Decoders
        self.t_qc_upsample = torch.nn.Sequential(
            torch.nn.Linear(seq_len_reduce, seq_len_new, bias=True),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
        )
        self.t_kc_upsample = torch.nn.Sequential(
            torch.nn.Linear(seq_len_reduce, seq_len_new, bias=True),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
        )
        self.t_vc_upsample = torch.nn.Sequential(
            torch.nn.Linear(seq_len_reduce, seq_len_new, bias=True),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
        )
        self.qc_upsample = torch.nn.Sequential( #! Since pytorch expects embed_dim to match query_dim, we need to upsample the query to dim3
            torch.nn.Linear(mean_var_hidden, dim2, bias=True),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False),
            torch.nn.Linear(dim2, dim3, bias=True),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.kc_upsample = torch.nn.Sequential( #! Luckily, key and value can have different dimensions
            torch.nn.Linear(mean_var_hidden, dim2, bias=True),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        self.vc_upsample = torch.nn.Sequential(
            torch.nn.Linear(mean_var_hidden, dim2, bias=True),
            torch.nn.GELU(approximate='none'),
            torch.nn.Dropout(dropout, inplace=False)
        )
        # Attention
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=dim3,
            num_heads=heads,
            dropout=attention_dropout,
            kdim=dim2,
            vdim=dim2,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.ln_2 = norm_layer(dim3)
        self.mlp = None
        if use_mlp:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dim3, mlp_dim, bias=True),
                torch.nn.GELU(approximate='none'),
                torch.nn.Dropout(dropout, inplace=False),
                torch.nn.Linear(mlp_dim, mlp_dim, bias=True),
                torch.nn.Dropout(dropout, inplace=False),
                torch.nn.Linear(mlp_dim, dim3, bias=True),
                torch.nn.Dropout(dropout, inplace=False),
            )

    def forward(self, input_q, input_kv=None, state_manager=None, mask=False):
        # One-Shot Encoders
        input_kv = input_q if input_kv is None else input_kv
        xq = input_q
        xkv = input_kv
        xq = self.ln_q(xq)
        xkv = self.ln_kv(xkv)
        if self.t_mean_cq:
            xq = xq.permute(0, 2, 1)
            mean_cq = self.t_mean_cq(xq)
            var_cq = self.t_var_cq(xq)
            mean_cq = mean_cq.permute(0, 2, 1)
            var_cq = var_cq.permute(0, 2, 1)
        else:
            mean_cq = xq
            var_cq = xq
        mean_ckv = self.t_mean_ckv(xkv.permute(0, 2, 1))
        var_ckv = self.t_var_ckv(xkv.permute(0, 2, 1))
        mean_ckv = mean_ckv.permute(0, 2, 1)
        var_ckv = var_ckv.permute(0, 2, 1)
        mean_cq = self.mean_cq(mean_cq)
        var_cq = self.var_cq(var_cq)
        mean_ckv = self.mean_ckv(mean_ckv)
        var_ckv = self.var_ckv(var_ckv)
        # Get residual states from state manager
        if state_manager is not None:
            mean_cq, var_cq, mean_ckv, var_ckv = state_manager.get_sums(mean_cq, var_cq, mean_ckv, var_ckv)
        # Compute samples
        if not self.static_epsilon_weight and self.constrain_epsilon:
            e2de_ratio_param = self.sigmoid(self.e2de_ratio)
        zcq = mean_cq + self.tanh_1((e2de_ratio_param * torch.randn_like(var_cq)) + ((1 - e2de_ratio_param) * self.dirty_epsilon_cq)) * torch.exp(0.5 * var_cq)
        zckv = mean_ckv + self.tanh_2((e2de_ratio_param * torch.randn_like(var_ckv)) + ((1 - e2de_ratio_param) * self.dirty_epsilon_ckv)) * torch.exp(0.5 * var_ckv)
        # Decoders
        if self.t_mean_cq:
            qc = zcq.permute(0, 2, 1)
            qc = self.t_qc_upsample(qc)
            qc = qc.permute(0, 2, 1)
        ckv = zckv.permute(0, 2, 1)
        kc = self.t_kc_upsample(ckv)
        kc = kc.permute(0, 2, 1)
        vc = self.t_vc_upsample(ckv)
        vc = vc.permute(0, 2, 1)
        qc = self.qc_upsample(qc if self.t_mean_cq else zcq)
        kc = self.kc_upsample(kc)
        vc = self.vc_upsample(vc)
        # Attention
        mask_mat = None
        if mask:
            mask_mat = torch.zeros(qc.shape[1], qc.shape[1]).bool().to(qc.device)
            mask_mat = torch.triu(mask_mat, diagonal=1)
        x, _ = self.attention(qc, kc, vc, attn_mask=mask_mat)
        x = self.dropout(x)
        x = x + qc
        if x.shape == input_q.shape: x += input_q
        if self.mlp is not None:
            y = self.ln_2(x)
            y = self.mlp(y)
        return x + y if self.mlp is not None else self.ln_2(x)

class EncoderBlock(torch.nn.Module):
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
        kernel_size: int=3
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.is_first_block = is_first_block
        if is_first_block:
            # self.cls_xq = torch.nn.Parameter(torch.randn(1, 1, dim1), requires_grad=True)
            # self.cls_xkv = torch.nn.Parameter(torch.randn(1, 1, dim1), requires_grad=True)
            self.adv_xq = torch.nn.Parameter(torch.randn(1, 1, dim1), requires_grad=True)
            self.adv_xkv = torch.nn.Parameter(torch.randn(1, 1, dim1), requires_grad=True)
            seq_length -= 1
        self.time_embedding_xq = torch.nn.Parameter(torch.randn(1, 1, dim1), requires_grad=True)
        self.xq_proj = torch.nn.Conv2d(3, dim1, kernel_size=(kernel_size, seq_length), stride=(1, seq_length), padding=(padding, 0), bias=True)
        self.encoder= MLDA_Block(
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
        self.xkv_proj = torch.nn.Conv2d(3, dim1, kernel_size=(seq_length, kernel_size), stride=(seq_length, 1), padding=(0, padding), bias=True)
        self.decoder = MLDA_Block(
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
        self.cross= MLDA_Block(
            heads=heads,
            dim1=dim1,
            dim2=dim2,
            dim3=(dim1 + (dim_step * 3)),
            mean_var_hidden=mean_var_hidden,
            seq_length=seq_length,
            seq_len_reduce=seq_len_reduce,
            #! Stupid hack but we need a way to reduce and induce the class token while maintaing dimensional divisibility
            seq_len_new=seq_length + (seq_len_step * 3) - (1 if is_first_block else 0),
            mlp_dim=(dim1 + (dim_step * 3)) * 2,
            use_mlp=True
        )

    def forward(self, x, esm=None, dsm=None, csm=None, mask=False):
        if self.is_first_block:
            xq = x
        else:
            xq = x.reshape(x.shape[0], x.shape[1], x.shape[1], 3).permute(0, 3, 1, 2)
        xq = self.xq_proj(xq)
        xq = xq.reshape(xq.shape[0], xq.shape[1], xq.shape[2]).permute(0, 2, 1) + self.time_embedding_xq
        xq = self.encoder(xq, state_manager=esm, mask=mask)
        xkv = xq.reshape(xq.shape[0], xq.shape[1], xq.shape[1], 3).permute(0, 3, 1, 2)
        xkv = self.xkv_proj(xkv)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[3]).permute(0, 2, 1) + self.time_embedding_xkv
        xkv = self.decoder(xkv, state_manager=dsm, mask=mask)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[1], 3).permute(0, 2, 1, 3)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[2] * xkv.shape[3])
        if self.is_first_block:
            adv_xq = self.adv_xq.expand(xq.shape[0], -1, -1)
            xq = torch.cat((adv_xq, xq), dim=1)
            adv_xkv = self.adv_xkv.expand(xkv.shape[0], -1, -1)
            xkv = torch.cat((adv_xkv, xkv), dim=1)
        x = self.cross(xq, xkv, state_manager=csm, mask=mask)
        return x

#! Effectively the same as EncoderBlock but with a different forward pass bc
#! the DDP does not like the fact that we are using the same class for both.
#! Oh well.
class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        heads: int,
        dim1: int,
        dim2: int,
        dim_step: int,
        mean_var_hidden: int,
        seq_length: int,
        seq_len_step: int,
        is_last_block: bool,
        seq_len_reduce:int,
        kernel_size: int=3
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.is_last_block = is_last_block
        self.time_embedding_q = torch.nn.Parameter(torch.randn(1, 1, dim1), requires_grad=True)
        self.xq_proj = torch.nn.ConvTranspose2d(seq_length, 1, kernel_size=(kernel_size, seq_length), stride=(1, seq_length), padding=(padding, 0), bias=True)
        self.encoder= MLDA_Block(
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
        self.xkv_proj = torch.nn.ConvTranspose2d(seq_length, 1, kernel_size=(seq_length, kernel_size), stride=(seq_length, 1), padding=(0, padding), bias=True)
        self.decoder = MLDA_Block(
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
        self.cross= MLDA_Block(
            heads=heads,
            dim1=dim1,
            dim2=dim2,
            dim3=(dim1 + (dim_step * 3)),
            mean_var_hidden=mean_var_hidden,
            seq_length=seq_length,
            seq_len_reduce=seq_len_reduce,
            #! Stupid hack but we need a way to reduce and induce the class token while maintaing dimensional divisibility
            seq_len_new=seq_length + (seq_len_step * 3) - (-1 if is_last_block else 0),
            mlp_dim=(dim1 + (dim_step * 3)) * 2,
            use_mlp=True
        )
        self.filter = None
        if kernel_size == 3:
            self.filter = torch.nn.ConvTranspose2d(3, 3, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(padding, padding), bias=True)

    def forward(self, x, esm=None, dsm=None, csm=None, mask=False):
        xq = x
        xq = xq.reshape(xq.shape[0], xq.shape[1], xq.shape[1], 3)
        xq = self.xq_proj(xq)
        xq = xq.reshape(xq.shape[0], xq.shape[2], xq.shape[3]) + self.time_embedding_q
        xq = self.encoder(xq, state_manager=esm, mask=mask)
        xkv = xq
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], 3, xkv.shape[1])
        xkv = self.xkv_proj(xkv).permute(0, 1, 3, 2)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[2], xkv.shape[1] * xkv.shape[3]) + self.time_embedding_xkv
        xkv = self.decoder(xkv, state_manager=dsm, mask=mask)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[1], 3).permute(0, 2, 1, 3)
        xkv = xkv.reshape(xkv.shape[0], xkv.shape[1], xkv.shape[2] * xkv.shape[3])
        x = self.cross(xq, xkv, state_manager=csm, mask=mask)
        adv = None
        if self.is_last_block and self.filter is not None:
            adv = x[:, 0]
            adv = adv.reshape(adv.shape[0], 1, adv.shape[1])
            x = x[:, 1:]
        if self.filter is not None:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[1], 3).permute(0, 3, 1, 2)
            x_orig = x
            x = self.filter(x) + x_orig
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        if adv is not None:
            x = torch.cat((adv, x), dim=1)
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
                EncoderBlock(
                    heads=heads,
                    dim1=dim1,
                    dim2=dim2,
                    dim_step=-dim_step,
                    mean_var_hidden=mean_var_hidden,
                    is_first_block=(i == 0),
                    seq_length=seq_length,
                    seq_len_step=-seq_len_step,
                    seq_len_reduce=seq_len_reduce
                )
            )
            dim1 -= (dim_step * 3)
            seq_length -= (seq_len_step * 3) + (1 if i == 0 else 0)
        self.block_bottle_neck_1 = EncoderBlock(
            heads=heads,
            dim1=dim1,
            dim2=dim2,
            dim_step=0,
            mean_var_hidden=mean_var_hidden,
            is_first_block=False,
            seq_length=seq_length,
            seq_len_step=0,
            seq_len_reduce=seq_len_reduce
        )
        # Decoder-Blocks
        self.block_bottle_neck_2 = DecoderBlock(
            heads=heads,
            dim1=dim1,
            dim2=dim2,
            dim_step=0,
            mean_var_hidden=mean_var_hidden,
            is_last_block=False,
            seq_length=seq_length,
            seq_len_step=0,
            seq_len_reduce=seq_len_reduce
        )
        self.decoder_blocks = torch.nn.ModuleList()
        for i in range(3):
            self.decoder_blocks.append(
                DecoderBlock(
                    heads=heads,
                    dim1=dim1,
                    dim2=dim2,
                    dim_step=dim_step,
                    mean_var_hidden=mean_var_hidden,
                    is_last_block=(i == 2),
                    seq_length=seq_length,
                    seq_len_step=seq_len_step,
                    seq_len_reduce=seq_len_reduce
                )
            )
            dim1 += (dim_step * 3)
            seq_length += (seq_len_step * 3)
        self.ln = norm_layer(dim1)

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