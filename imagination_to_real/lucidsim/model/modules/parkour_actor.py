from copy import deepcopy
from logging import warning
from typing import Tuple, Union

import torch
from .actor_critic import Actor, get_activation
from params_proto import ParamsProto
from torch import nn
from torchtyping import TensorType

from .depth_backbone import DepthOnlyFCBackbone, RecurrentDepthBackbone
from .estimator import Estimator


def get_parkour_teacher_policy():
    actor = ParkourActor()

    if torch.cuda.is_available():
        return actor.to("cuda")

    return actor


class PolicyArgs(ParamsProto, cli=False, prefix="policy"):
    n_proprio = 53
    n_scan = 132
    num_actions = 12
    scan_encoder_dims = [128, 64, 32]
    actor_hidden_dims = [512, 256, 128]
    priv_encoder_dims = [64, 20]
    estimator_hidden_dims = [128, 64]
    depth_hidden_dims = 512
    depth_shape = [80, 45]

    n_priv_latent = 4 + 1 + 12 + 12
    n_priv = 3 + 3 + 3
    history_len = 10
    activation = "elu"
    tanh_encoder_output = False

    use_camera = True
    direction_distillation = False


class DeploymentParams(ParamsProto, cli=False, prefix="deployment"):
    obs_scales = dict(
        lin_vel=2.0,
        ang_vel=0.25,
        dof_pos=1.0,
        dof_vel=0.05,
    )

    default_joint_angles = {
        "FL_hip_joint": 0.1,  # [rad]
        "RL_hip_joint": 0.1,  # [rad]
        "FR_hip_joint": -0.1,  # [rad]
        "RR_hip_joint": -0.1,  # [rad]
        "FL_thigh_joint": 0.8,  # [rad]
        "RL_thigh_joint": 1.0,  # [rad]
        "FR_thigh_joint": 0.8,  # [rad]
        "RR_thigh_joint": 1.0,  # [rad]
        "FL_calf_joint": -1.5,  # [rad]
        "RL_calf_joint": -1.5,  # [rad]
        "FR_calf_joint": -1.5,  # [rad]
        "RR_calf_joint": -1.5,  # [rad]
    }

    # depth
    width = 80
    height = 45
    near_clip = 0.28
    far_clip = 2.0
    camera_id = "realsense"
    fps = 30
    capture_resolution = (360, 640)

    update_interval = 5

    control_type = "P"
    stiffness_dict = dict(joint=20)
    damping_dict = dict(joint=0.5)

    action_scale = 0.25
    dt = 0.02

    clip_actions = 1.2
    clip_observations = 100

    flat_mask = False


class ParkourActor(nn.Module):
    def __init__(self):
        super().__init__()

        activation_fn = get_activation(PolicyArgs.activation)

        self.n_scan = PolicyArgs.n_scan
        self.n_proprio = PolicyArgs.n_proprio

        self.actor = Actor(
            n_proprio=PolicyArgs.n_proprio,
            n_scan=PolicyArgs.n_scan,
            num_actions=PolicyArgs.num_actions,
            scan_encoder_dims=PolicyArgs.scan_encoder_dims,
            actor_hidden_dims=PolicyArgs.actor_hidden_dims,
            priv_encoder_dims=PolicyArgs.priv_encoder_dims,
            n_priv_latent=PolicyArgs.n_priv_latent,
            n_priv=PolicyArgs.n_priv,
            history_len=PolicyArgs.history_len,
            activation_fn=activation_fn,
            tanh_encoder_output=PolicyArgs.tanh_encoder_output,
        )

        self.estimator = Estimator(
            input_dim=PolicyArgs.n_proprio,
            output_dim=PolicyArgs.n_priv,
            hidden_dims=PolicyArgs.estimator_hidden_dims,
        )

        self.use_camera = PolicyArgs.use_camera
        self.direction_distillation = PolicyArgs.direction_distillation

        if self.use_camera:
            depth_backbone = DepthOnlyFCBackbone(
                PolicyArgs.n_proprio,
                PolicyArgs.scan_encoder_dims[-1],
                PolicyArgs.depth_hidden_dims,
                PolicyArgs.depth_shape[::-1],
            )
            self.depth_encoder = RecurrentDepthBackbone(depth_backbone, PolicyArgs.n_proprio)
            self.depth_actor = deepcopy(self.actor)
            # singleton
            self.register_buffer("last_latent", torch.zeros([1, 32]).float())
        else:
            self.depth_encoder = None
            self.depth_actor = None
            # singleton
            self.last_latent = None

        self.policy = self.depth_actor if PolicyArgs.use_camera else self.actor

    def _parse_ac_params(self, params):
        actor_params = {}
        for k, v in params.items():
            if k.startswith("actor."):
                actor_params[k[6:]] = v
        return actor_params

    def load(self, logger_prefix, map_location=None):
        from ml_logger import logger

        warning(
            "add a decorator for handling network denied situationis, always load"
            "from local cache first. Otherwise update the local cache upon successful"
            "load."
        )
        # always load into CPU first.
        state_dict = logger.torch_load(logger_prefix, map_location=map_location)

        self.load_state_dict(state_dict)

    # overwrite load state dict
    def load_state_dict(self, state_dict):
        assert isinstance(state_dict, dict), "state_dict should be an ordered dict."

        actor_sdict = self._parse_ac_params(state_dict["model_state_dict"])
        self.actor.load_state_dict(actor_sdict)

        self.estimator.load_state_dict(state_dict["estimator_state_dict"])

        if self.depth_encoder:
            self.depth_encoder.load_state_dict(state_dict["depth_encoder_state_dict"])
        if self.depth_actor:
            dactor_sdict = state_dict.get("depth_actor_state_dict", actor_sdict)
            self.depth_actor.load_state_dict(dactor_sdict)

    def reset_hiddens(self):
        self.depth_encoder.hidden_states = None

    def forward(
            self,
            ego,
            obs: TensorType["batch", "num_observations"],
            vision_latent: Union[TensorType["batch", "scan_encoder_dim"], None] = None,
    ) -> Tuple[TensorType["batch", 12], TensorType["batch", "depth_latent_dim"]]:
        obs = obs.float()

        if not PolicyArgs.use_camera:
            if ego is not None:
                warning("this is a blind policy. Ego view is not None.")
            if vision_latent is not None:
                warning("this is a blind policy. Vision latent is not None.")
            vision_latent = None
        elif ego is not None:
            obs_student = obs[:, : PolicyArgs.n_proprio].clone()
            obs_student[:, 6:8] = 0
            depth_latent_and_yaw = self.depth_encoder(ego, obs_student)
            vision_latent = depth_latent_and_yaw[:, :-2]
            yaw = depth_latent_and_yaw[:, -2:]
            if PolicyArgs.direction_distillation:
                # Use the predicted yaw target
                obs[:, 6:8] = 1.5 * yaw
            self.last_latent = vision_latent
        elif vision_latent is None:
            vision_latent = self.last_latent

        priv_states_estimated = self.estimator(obs[:, : PolicyArgs.n_proprio])

        obs[
        :,
        PolicyArgs.n_proprio + PolicyArgs.n_scan: PolicyArgs.n_proprio + PolicyArgs.n_scan + PolicyArgs.n_priv,
        ] = priv_states_estimated

        actions = self.policy(obs, hist_encoding=True, scandots_latent=vision_latent)

        return actions, vision_latent
