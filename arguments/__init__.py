#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os
import json
import yaml

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                elif t == list:
                    group.add_argument("--" + key, default=value, nargs='+', type=type(value[0]) if value else str)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true", help=f"Enable {key}")
                    group.add_argument("--no-" + key, dest=key, default=value, action="store_false", help=f"Disable {key}")
                elif t == list:
                    group.add_argument("--" + key, default=value, nargs='+', type=type(value[0]) if value else str)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.random_background = True
        self.data_device = "cuda"
        self.eval = False
        # self._resolution = 1
        self.duration = 50
        self.test_views = [0]
        self.loader = "colmap"
        self._pcl_path = ""
        # add
        self.surface = True
        self._use_mask = 1
        self.normalize_depth = True
        self.perpix_depth = True
        self.mono_normal = False
        self.use_flow = False
        self.motion_degree = 3
        self.opac_init = 0.1
        self.random_init = False
        self.dup = 1
        self.sgmd_gaussian = 0
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.velocity_aggregation = "max"
        self.gtisint8 = 0 # 0 means gt is used as float .
        self.prefilter_for_raster = False
        self.prefilter_threshold = 0.1
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 0
        self.position_lr_init = 0.00016 * 1
        self.position_lr_final = 0.0000016 * 1
        self.position_lr_delay_mult = 0.01 # useless
        self.position_lr_max_steps = 0 #self.iterations
        self.feature_lr = 0.0025 * 1
        self.opacity_lr = 0.05 * 1
        self.scaling_lr = 0.005 * 1
        self.rotation_lr = 0.001
        self.camera_lr = 0.000
        self.trbfc_lr = 0.0001 # 
        self.trbfs_lr = 0.06
        self.trbfslinit = 0.0 #-4.0 # 
        self.batch = 1
        self.movelr = 3.5 * 10
        self.omega_lr = 0.0001 # * 10
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.pruning_interval = 100
        self.opacity_reset_interval = 3000
        self.opacity_reset_ratio = 0.12
        self.densify_from_iter = 200
        self.densify_until_iter = 0 #self.iterations / 2
        self.densify_grad_threshold = 0.0001
        self.min_opacity = 0.1
        self.velocity_based_sampling = False
        self.velocity_sample_from_iter = 15000
        self.velocity_sample_interval = 100
        self.opac_loss = "gsurfel"
        self.t_scale_loss = "none"
        self.lambda_opac = 0.1
        self.lambda_t_scale = 0.01
        self.lambda_surface = 0.3
        self.lambda_mask = 0.1
        self.lambda_smooth_coeff = 0.
        self.lambda_shape_reg = 0.
        self.lambda_flow = 0.01
        self.smooth_coeff_interval = 100
        self.knn_prune_interval = 30001 # this is larger than total iterations such that we don't use it
        self.knn_prune_ratio = 0.25

        # flow related stuff
        self.flow_psnr_threshold = 25.0
        self.lambda_flow = 0.001
        self.flow_sample_ratio = 1.0
        self.soft_select_fg = True
        self.flow_k = 20
        self.flow_weight_decay = True
        self.flow_warmup_step = 10000
        
        # not used but present in MAGS flow - keeping as reminder for later use
        self.dynamic_attn = True
        self.flow_render = False
        self.flow_bwd = True
        
        super().__init__(parser, "Optimization Parameters")

def load_config_file(filepath):
    """Load configuration file in either JSON or YAML format."""
    print(f"Loading configuration from: {filepath}")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    ext = os.path.splitext(filepath)[-1].lower()
    with open(filepath, 'r') as f:
        if ext == '.json':
            return json.load(f)
        elif ext in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")


def get_combined_args(args, config):
    """Merge command-line arguments with config file values."""
    args_merged = vars(args).copy()
    if config:
        # Merge configurations: Config file values always take precedence.
        args_merged.update({k: v for k, v in config.items()})

    return Namespace(**args_merged)
