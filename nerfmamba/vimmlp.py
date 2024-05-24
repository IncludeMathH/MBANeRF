# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi Layer Perceptron
"""
from typing import Dict, Literal, Optional, Set, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.utils.external import TCNN_EXISTS, tcnn
from nerfstudio.utils.printing import print_tcnn_speed_warning
from nerfstudio.utils.rich_utils import CONSOLE

# for mamba begin
# for mamba
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

from einops import rearrange, repeat
from timm.models.layers import DropPath
# for mamba end

class VimMLP(FieldComponent):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
        mamba_idxes: 哪些层需要使用Mamba
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        mamba_idxes: Optional[Tuple[int]] = None,
        ssm_cfg: Optional[Dict[str, Union[str, int]]] = {'expand': 1},
        bimamba_type: Optional[str] = 'v2',
        init_layer_scale: Optional[float] = None,
        if_devide_out: Optional[bool] = True,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        norm_epsilon: Optional[float] = 1e-5, 
        rms_norm: Optional[bool] = False, 
        residual_in_fp32: Optional[bool] = False, 
        fused_add_norm: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.mamba_idxes = mamba_idxes
        self._mamba_idxes: Set[int] = set(mamba_idxes) if mamba_idxes else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out
        self.init_layer_scale = init_layer_scale
        if ssm_cfg is None:
            self.ssm_cfg = {}
        else:
            self.ssm_cfg = ssm_cfg
        self.norm_epsilon = norm_epsilon
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize the torch version of the multi-layer perceptron."""
        
        layers = []

        drop_path_rate = 0.1
        depth = 1

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr

        if self.num_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    if i not in self._mamba_idxes:
                        layers.append(nn.Linear(self.in_dim, self.layer_width))
                    else:
                        layers.append(nn.Sequential(
                                create_block(
                                    d_model=self.in_dim,
                                    ssm_cfg={'expand': 1},
                                    norm_epsilon=self.norm_epsilon,
                                    rms_norm=False,
                                    residual_in_fp32=False,
                                    fused_add_norm=False,
                                    layer_idx=i,
                                    if_bimamba=False,       # if True, use 'v1'
                                    bimamba_type='v2',
                                    drop_path=inter_dpr[i],
                                    if_devide_out=True,
                                    init_layer_scale=None,
                                    ),
                            nn.Linear(self.in_dim, self.layer_width),
                                                    )
                                     )
                elif i in self._skip_connections:
                    if i not in self._mamba_idxes:
                        layers.append(nn.Linear(self.layer_width + self.in_dim, self.layer_width))
                    else:
                        layers.append(
                            nn.Sequential(
                                nn.Linear(self.layer_width + self.in_dim, self.layer_width),
                                create_block(
                                    d_model=self.layer_width,
                                    ssm_cfg={'expand': 1},
                                    norm_epsilon=self.norm_epsilon,
                                    rms_norm=False,
                                    residual_in_fp32=False,
                                    fused_add_norm=False,
                                    layer_idx=i,
                                    if_bimamba=False,       # if True, use 'v1'
                                    bimamba_type='v2',
                                    drop_path=inter_dpr[i],
                                    if_devide_out=True,
                                    init_layer_scale=None,
                                    ),)
                        )
                else:
                    if i not in self._mamba_idxes:
                        layers.append(nn.Linear(self.layer_width, self.layer_width))
                    else:
                        layers.append(                                
                                create_block(
                                    d_model=self.layer_width,
                                    ssm_cfg={'expand': 1},
                                    norm_epsilon=self.norm_epsilon,
                                    rms_norm=False,
                                    residual_in_fp32=False,
                                    fused_add_norm=False,
                                    layer_idx=i,
                                    if_bimamba=False,       # if True, use 'v1'
                                    bimamba_type='v2',
                                    drop_path=inter_dpr[i],
                                    if_devide_out=True,
                                    init_layer_scale=None,
                                    ),)
            if self.num_layers - 1 in self._mamba_idxes:
                print(f'using mamba in last layer')
                layers.append(
                                create_block(
                                    d_model=self.layer_width,
                                    ssm_cfg={'expand': 1},
                                    norm_epsilon=self.norm_epsilon,
                                    rms_norm=False,
                                    residual_in_fp32=False,
                                    fused_add_norm=False,
                                    layer_idx=i,
                                    if_bimamba=False,       # if True, use 'v1'
                                    bimamba_type='v2',
                                    drop_path=inter_dpr[i],
                                    if_devide_out=True,
                                    init_layer_scale=None,
                                    ),
                )
            else:
                layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)

        # TODO: 灵活表达这个东西
        self.norm_alpha = (nn.LayerNorm if not self.rms_norm else RMSNorm)(
            self.layer_width, eps=self.norm_epsilon
        )

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        if self._mamba_idxes is not None:
            residual = None
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            if i in self._mamba_idxes:
                x, residual = layer(x, residual)
            else:
                x = layer(x)
                if self.activation is not None and i < len(self.layers) - 1:
                    x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)

        if not self.fused_add_norm:
            residual = (x + residual) if residual is not None else x
            h = self.norm_alpha(residual.to(dtype=self.norm_alpha.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_alpha, RMSNorm) else layer_norm_fn
            x = fused_add_norm_fn(
                x,
                self.norm_alpha.weight,
                self.norm_alpha.bias,
                eps=self.norm_alpha.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return x

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        return self.pytorch_fwd(in_tensor)

# Model
def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
