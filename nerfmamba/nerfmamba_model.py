"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Type

from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig  # for subclassing Nerfacto model
from nerfmamba.nerfmamba_field import NerfMambaField  # for subclassing NeRFField
from nerfstudio.field_components.encodings import NeRFEncoding


@dataclass
class NerfMambaConfig(VanillaModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: NerfMambaModel)
    skip_connections: Tuple[int] = (4,)
    base_mlp_num_layers: int = 2
    base_mamba_idxes: Tuple[int] = (1,)
    
    head_mlp_num_layers: int = 2
    head_mamba_idxes: Tuple[int] = (1,)



class NerfMambaModel(NeRFModel):
    """Template Model."""

    config: NerfMambaConfig

    def populate_modules(self):
        super().populate_modules()

        # fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.field_coarse = NerfMambaField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            base_mlp_num_layers=self.config.base_mlp_num_layers,
            head_mlp_num_layers=self.config.head_mlp_num_layers,
            skip_connections=self.config.skip_connections,
            base_mamba_idxes=self.config.base_mamba_idxes,
            head_mamba_idxes=self.config.head_mamba_idxes,
        )

        self.field_fine = NerfMambaField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            base_mlp_num_layers=self.config.base_mlp_num_layers,
            head_mlp_num_layers=self.config.head_mlp_num_layers,
            skip_connections=self.config.skip_connections,
            base_mamba_idxes=self.config.base_mamba_idxes,
            head_mamba_idxes=self.config.head_mamba_idxes,
        )