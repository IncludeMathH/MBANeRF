from nerfmamba.vimmlp import VimMLP
from nerfstudio.fields.vanilla_nerf_field import NeRFField

from typing import Optional, Tuple, Type
from torch import nn

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import FieldHead, RGBFieldHead

class NerfMambaField(NeRFField):
    """Compound Field that uses TCNN

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.
    """

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        field_heads: Optional[Tuple[Type[FieldHead]]] = (RGBFieldHead,),
        use_integrated_encoding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        base_mamba_idxes: Optional[Tuple[int]] = None,
        head_mamba_idxes: Optional[Tuple[int]] = None,
    ) -> None:
        super().__init__(
            position_encoding,
            direction_encoding,
            base_mlp_num_layers,
            base_mlp_layer_width,
            head_mlp_num_layers,
            head_mlp_layer_width,
            skip_connections,
            field_heads,
            use_integrated_encoding,
            spatial_distortion,
        )

        self.mlp_base = VimMLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
            mamba_idxes=base_mamba_idxes,
        )

        if field_heads:
            self.mlp_head = VimMLP(
                in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
                num_layers=head_mlp_num_layers,
                layer_width=head_mlp_layer_width,
                out_activation=nn.ReLU(),
                mamba_idxes=head_mamba_idxes,
            )
