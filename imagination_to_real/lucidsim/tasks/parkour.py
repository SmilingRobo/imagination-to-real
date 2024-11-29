from typing import Literal

from dm_control.rl import control
from gym_dmc.wrappers import FlattenObservation

from imagination_to_real.lucidsim import ChDir, add_env
from imagination_to_real.lucidsim.tasks import ROOT
from imagination_to_real.lucidsim.tasks.base.go1_base import Go1, Physics
from imagination_to_real.lucidsim.wrappers.depth_midas_render_wrapper import MidasRenderDepthWrapper
from imagination_to_real.lucidsim.wrappers.history_wrapper import HistoryWrapper
from imagination_to_real.lucidsim.wrappers.lucid_env import LucidEnv
from imagination_to_real.lucidsim.wrappers.render_rgb_wrapper import RenderRGBWrapper
from imagination_to_real.lucidsim.wrappers.reset_wrapper import ResetWrapper
from imagination_to_real.lucidsim.wrappers.scandots_wrapper import ScandotsWrapper
from imagination_to_real.lucidsim.wrappers.segmentation_wrapper import SegmentationWrapper

DEFAULT_TIME_LIMIT = 25

PHYSICS_TIMESTEP = 0.005  # in XML
DECIMATION = 4
CONTROL_TIMESTEP = PHYSICS_TIMESTEP * DECIMATION


def entrypoint(
        xml_path,
        mode: Literal["heightmap", "vision", "depth", "segmentation", "heightmap_splat"],
        # waypoint randomization
        y_noise,
        x_noise,
        # whether to add cones as visible geoms at waypoints
        use_cones=False,
        time_limit=DEFAULT_TIME_LIMIT,
        random=None,
        device=None,
        move_speed_range=[0.8, 0.8],
        # for vision
        stack_size=1,
        check_contact_termination=False,
        **kwargs,
):
    """Returns the Walk task."""
    with ChDir(ROOT):
        physics = Physics.from_xml_path(xml_path)

    if not use_cones:
        model = physics.model
        named_model = physics.named.model
        all_geom_names = [model.geom(i).name for i in range(model.ngeom)]
        # set transparency to 0
        for geom_name in all_geom_names:
            if geom_name.startswith("cone"):
                named_model.geom_rgba[geom_name] = [0, 0, 0, 0]

    task = Go1(vision=True, move_speed_range=move_speed_range, y_noise=y_noise, x_noise=x_noise, random=random,
               **kwargs)
    env = control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=CONTROL_TIMESTEP,
        flat_observation=True,
    )
    env = LucidEnv(env)
    env = FlattenObservation(env)
    env = HistoryWrapper(env, history_len=10)
    env = ResetWrapper(env, check_contact_termination=check_contact_termination)

    if mode == "midas_depth":
        env = MidasRenderDepthWrapper(
            env,
            width=1280,
            height=720,
            camera_id="ego-rgb",
            device=device,
        )
    elif mode == "render_rgb":
        env = ScandotsWrapper(
            env,
            **kwargs,
            device=device,
        )
        env = RenderRGBWrapper(
            env,
            width=1280,
            height=720,
            camera_id="ego-rgb",
        )
    elif mode == "heightmap":
        env = ScandotsWrapper(env, **kwargs, device=device)
    elif mode == "segmentation":
        # pick these for the segmentation.
        groups = kwargs.pop("groups")
        return_masks = kwargs.pop("return_masks", None)

        env = ScandotsWrapper(env, **kwargs, device=device)
        env = MidasRenderDepthWrapper(
            env,
            width=1280,
            height=768,
            camera_id="ego-rgb-render",
            update_interval=1,
            device=device,
        )
        env = SegmentationWrapper(
            env,
            width=1280,
            height=768,
            camera_id="ego-rgb-render",
            groups=groups,
            return_masks=return_masks,
            device=device,
        )
    elif mode == "lucidsim":
        groups = kwargs.pop("groups")
        return_masks = kwargs.pop("return_masks", None)
        env = ScandotsWrapper(env, **kwargs, device=device)
        env = MidasRenderDepthWrapper(
            env,
            width=1280,
            height=768,
            camera_id="ego-rgb-render",
            device=device,
        )
        env = SegmentationWrapper(
            env,
            width=1280,
            height=768,
            camera_id="ego-rgb-render",
            groups=groups,
            return_masks=return_masks,
            device=device,
        )
    else:
        raise NotImplementedError(f"mode {mode} is not implemented.")

    return env

add_env(
    env_id="Parkour",
    entrypoint=entrypoint,
    kwargs=dict(
        xml_path="parkour.xml",
        mode="lucidsim",
        n_proprio=53,
        groups=[["platform", "ramp-1", "ramp-2", "ramp-3"], ["cone*"]],
        x_noise=0.1,
        y_noise=0.1,
        use_cones=True,
    ),
)
