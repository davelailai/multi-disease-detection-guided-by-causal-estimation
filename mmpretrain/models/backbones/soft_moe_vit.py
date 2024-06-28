
from __future__ import annotations
from functools import partial
from typing import Optional, Union, cast

import torch
from einops import rearrange
from torch import Tensor, nn



from copy import deepcopy
from typing import Callable, Optional, Union

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.transformer import _get_activation_fn

from .vision_transformer import VisionTransformer




import torch
from einops import einsum, rearrange




import math

import torch
from einops import einsum, rearrange
from torch import Tensor, nn
from mmpretrain.registry import MODELS


class MultiExpertLayer(nn.Module):
    """A more efficient alternative to creating 'n' separate expert layers (likely
    from 'nn.Linear' modules).  Instead, we create a single set of batched weights
    and biases, and apply all 'experts' in parallel.

    Args:
        embed_dim (int): embedding dimension (d)
        num_experts (int): number of experts (n)
        bias (bool): whether to include a bias term. Default: True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.empty(
                (num_experts, in_features, out_features), device=device, dtype=dtype
            )
        )
        bias_param: Optional[nn.Parameter] = None
        if bias:
            bias_param = nn.Parameter(
                torch.empty((num_experts, out_features), device=device, dtype=dtype)
            )
        # Include type annotation for mypy :D
        self.bias: Optional[nn.Parameter]
        self.register_parameter("bias", bias_param)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # NOTE: Mostly copy-pasta from 'nn.Linear.reset_parameters'
        #
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected input with embed_dim={self.in_features} (dim=-1), but "
                f"found {x.size(-1)}"
            )
        elif x.size(1) != self.num_experts:
            raise ValueError(
                f"Expected input with num_experts={self.num_experts} (dim=1), but "
                f"found {x.size(1)}"
            )

        # NOTE: 'd1' and 'd2' are both equal to 'embed_dim'. But for 'einsum' to
        # work correctly, we have to give them different names.
        x = einsum(x, self.weight, "b n ... d1, n d1 d2 -> b n ... d2")

        if self.bias is not None:
            # NOTE: When used with 'SoftMoE' the inputs to 'MultiExpertLayer' will
            # always be 4-dimensional.  But it's easy enough to generalize for 3D
            # inputs as well, so I decided to include that here.
            if x.ndim == 3:
                bias = rearrange(self.bias, "n d -> () n d")
            elif x.ndim == 4:
                bias = rearrange(self.bias, "n d -> () n () d")
            else:
                raise ValueError(
                    f"Expected input to have 3 or 4 dimensions, but got {x.ndim}"
                )
            x = x + bias

        return x

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_experts={self.num_experts}, bias={self.bias is not None}"
        )


class SoftMoE(nn.Module):
    """A PyTorch module for Soft-MoE, as described in the paper:
        "From Sparse to Soft Mixtures of Experts"
        https://arxiv.org/pdf/2308.00951.pdf

    einstein notation:
    - b: batch size
    - m: input sequence length
    - d: embedding dimension
    - n: num experts
    - p: num slots per expert
    - (n * p): total number of slots

    Args:
        embed_dim (int): embedding dimension (d)
        num_experts (int): number of experts (n)
        slots_per_expert (int): number of slots per expert (p)
        bias (bool): whether to include a bias term. Default: True.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        slots_per_expert: int,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.bias = bias

        self.phi = nn.Parameter(
            torch.empty(
                (in_features, num_experts, slots_per_expert),
                device=device,
                dtype=dtype,
            )
        )
        self.experts = MultiExpertLayer(
            in_features=in_features,
            out_features=out_features,
            num_experts=num_experts,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # NOTE: Copy weight initialization from 'nn.Linear.reset_parameters'
        # TODO: Check for initialization strategy from the paper
        nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the Soft-MoE layer, as described in:
            https://arxiv.org/pdf/2308.00951.pdf
        See: equations (1-3), algorithm 1, and figure 2

        einstein notation:
        - b: batch size
        - m: input sequence length
        - d: embedding dimension
        - n: num experts
        - p: num slots per expert
        - (n * p): total number of slots

        Args:
            x (Tensor): input tensor of shape (b, m, d)

        Returns:
            Tensor: output tensor of shape (b, m, d)
        """
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected x.size(-1)={x.size(-1)} to match embed_dim={self.in_features}, "
                f"but got {x.size(-1)}."
            )
        elif x.ndim != 3:
            raise ValueError(f"Expected input to have 3 dimensions, but got {x.ndim}.")

        logits = einsum(x, self.phi, "b m d, d n p -> b m n p")
        dispatch_weights = logits.softmax(dim=1)  # denoted 'D' in the paper
        # NOTE: The 'torch.softmax' function does not support multiple values for the
        # 'dim' argument (unlike jax), so we are forced to flatten the last two dimensions.
        # Then, we rearrange the Tensor into its original shape.
        combine_weights = rearrange(
            logits.flatten(start_dim=2).softmax(dim=-1),
            "b m (n p) -> b m n p",
            n=self.num_experts,
        )

        # NOTE: To save memory, I don't rename the intermediate tensors Y, Ys, Xs.
        # Instead, I just overwrite the 'x' variable.  The names from the paper are
        # included in a comment for each line below.
        x = einsum(x, dispatch_weights, "b m d, b m n p -> b n p d")  # Xs
        x = self.experts(x)  # Ys
        x = einsum(x, combine_weights, "b n p d, b m n p -> b m d")  # Y

        return x

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_experts={self.num_experts}, slots_per_expert={self.slots_per_expert}, "
            f"bias={self.bias}"
        )


class SoftMoEEncoderLayer(nn.Module):
    """PyTorch module for Soft-MoE Transformer Encoder Layer, as described in:
        https://arxiv.org/pdf/2308.00951.pdf

    NOTE: Nearly identical to a standard Transformer encoder layer, except that we
    replace the second feedforward layer with 'SoftMoE'.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward: int = 2048,
        num_experts: int = 128,
        slots_per_expert: int = 1,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.d_model = d_model
        self.norm_first = norm_first
        self.activation = activation

        self.dropout = nn.Dropout(dropout)

        # self-attention block
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

        # feedforward / soft-moe block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.moe = SoftMoE(
            in_features=dim_feedforward,
            out_features=d_model,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
            device=device,
            dtype=dtype,
        )

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            # is_causal=is_causal,
        )
        return self.dropout(x)

    # feedforward / soft-moe block
    def _ff_block(self, x: Tensor) -> Tensor:
        """Forward pass for the FeedForward block, which now includes a SoftMoE layer.
        Mostly copy-pasta from 'nn.TransformerEncoderLayer'.  The only difference
        is swapping 'self.linear2' for 'self.moe'.
        """
        x = self.moe(self.dropout(self.activation(self.linear(x))))
        return self.dropout(x)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x


class SoftMoEEncoder(nn.Module):
    """PyTorch module for Soft-MoE Transformer Encoder, as described in:
        https://arxiv.org/pdf/2308.00951.pdf

    NOTE: Nearly identical to a standard Transformer encoder, except that we
    replace the second feedforward (nn.Linear) in each layer with 'SoftMoE'.
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = src
        for layer in self.layers:
            x = layer(
                x,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
        return x


class SoftMoEDecoderLayer(nn.Module):
    """PyTorch module for Soft-MoE Transformer Decoder Layer, as described in:
        https://arxiv.org/pdf/2308.00951.pdf

    NOTE: Nearly identical to a standard Transformer decoder layer, except that we
    replace the second feedforward layer with 'SoftMoE'.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_experts: int = 128,
        slots_per_expert: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.d_model = d_model
        self.norm_first = norm_first
        self.activation = activation

        self.dropout = nn.Dropout(dropout)

        # self-attention block
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

        # cross-attention block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

        # feedforward / soft-moe block
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.moe = SoftMoE(
            in_features=dim_feedforward,
            out_features=d_model,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
            device=device,
            dtype=dtype,
        )

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )
        return self.dropout(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x, _ = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )
        return self.dropout(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.moe(self.dropout(self.activation(self.linear(x))))
        return self.dropout(x)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x


class SoftMoEDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [deepcopy(decoder_layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )
        return x
    
class ViTWrapper(nn.Module):
    def __init__(
        self,
        num_classes: Optional[int],
        encoder: Union[SoftMoEEncoder, nn.TransformerEncoder],
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        dropout: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if not image_size % patch_size == 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by "
                f"patch_size ({patch_size})"
            )
        self.num_classes = num_classes
        self.encoder = encoder
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        # Extract model dimension from the first layer of the encoder.
        # TODO: Find a cleaner way to do this?  Unfortunately, TransformerEncoder
        # and TransformerEncoderLayer don't have a 'd_model' property.
        encoder_layer = cast(
            Union[SoftMoEEncoderLayer, nn.TransformerEncoderLayer],
            encoder.layers[0],
        )
        norm_layer = cast(nn.LayerNorm, encoder_layer.norm1)
        d_model = norm_layer.normalized_shape[0]
        num_patches = (image_size // patch_size) ** 2
        patch_dim = num_channels * patch_size**2

        self.patch_to_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim, device=device, dtype=dtype),
            nn.Linear(patch_dim, d_model, device=device, dtype=dtype),
            nn.LayerNorm(d_model, device=device, dtype=dtype),
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, d_model, device=device, dtype=dtype)
        )
        self.dropout = nn.Dropout(dropout)

        self.out: nn.Module
        if num_classes is not None:
            self.out = nn.Linear(d_model, num_classes, device=device, dtype=dtype)
        else:
            self.out = nn.Identity()

    def forward(self, x: Tensor, return_features: bool = False) -> Tensor:
        if not x.size(1) == self.num_channels:
            raise ValueError(
                f"Expected num_channels={self.num_channels} but found {x.size(1)}"
            )
        elif not x.size(2) == x.size(3) == self.image_size:
            raise ValueError(
                f"Expected image_size={self.image_size} but found {x.size(2)}x{x.size(3)}"
            )

        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        x = self.patch_to_embedding(x)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.encoder(x)

        if return_features:
            return x

        x = x.mean(dim=-2)
        return self.out(x)


class ViT(ViTWrapper):
    def __init__(
        self,
        num_classes: Optional[int],
        image_size: int = 224,
        patch_size: int = 16,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_channels: int = 3,
        dropout: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            device=device,
            dtype=dtype,
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        super().__init__(
            num_classes=num_classes,
            encoder=encoder,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )

@MODELS.register_module()
class SoftMoEViT(VisionTransformer):
    def __init__(
        self,
        arch='base',
        img_size=224,
        patch_size=16,
        num_experts: int = 128,
        slots_per_expert: int = 1,
        in_channels=3,
        drop_rate=0.,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None, 
        **kwargs       
    ) -> None:
        
            
        d_model= self.arch_settings['embed_dims']
        num_encoder_layers =self.arch_settings['num_layers']
        nhead =self.arch_settings['num_heads'],
        dim_feedforward =self.arch_settings['feedforward_channels']

        encoder_layer = SoftMoEEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for _ in range(num_encoder_layers)]
        )
        super().__init__(
            arch=arch,
            image_size=img_size,
            patch_size=patch_size,
            num_channels=in_channels,
            dropout=drop_rate,
            device=device,
            dtype=dtype,
        )


def _build_vit(
    num_classes: Optional[int],
    image_size: int,
    patch_size: int,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    mlp_ratio: float = 4.0,
    num_channels: int = 3,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> ViT:
    return ViT(
        num_classes=num_classes,
        image_size=image_size,
        patch_size=patch_size,
        d_model=d_model,
        dim_feedforward=int(d_model * mlp_ratio),
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_channels=num_channels,
        device=device,
        dtype=dtype,
    )


vit_small = partial(_build_vit, d_model=384, nhead=6, num_encoder_layers=12)
vit_base = partial(_build_vit, d_model=768, nhead=12, num_encoder_layers=12)
vit_large = partial(_build_vit, d_model=1024, nhead=16, num_encoder_layers=24)
vit_huge = partial(_build_vit, d_model=1280, nhead=16, num_encoder_layers=32)


def _build_soft_moe_vit(
    num_classes: Optional[int],
    image_size: int,
    patch_size: int,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    num_experts: int,
    slots_per_expert: int = 1,
    mlp_ratio: float = 4.0,
    num_channels: int = 3,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> SoftMoEViT:
    return SoftMoEViT(
        num_classes=num_classes,
        image_size=image_size,
        patch_size=patch_size,
        d_model=d_model,
        dim_feedforward=int(d_model * mlp_ratio),
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        num_channels=num_channels,
        device=device,
        dtype=dtype,
    )


soft_moe_vit_small = partial(
    _build_soft_moe_vit, d_model=384, nhead=6, num_encoder_layers=12
)
soft_moe_vit_base = partial(
    _build_soft_moe_vit, d_model=768, nhead=12, num_encoder_layers=12
)
soft_moe_vit_large = partial(
    _build_soft_moe_vit, d_model=1024, nhead=16, num_encoder_layers=24
)
soft_moe_vit_huge = partial(
    _build_soft_moe_vit, d_model=1280, nhead=16, num_encoder_layers=32
)
