from imagination_to_real.lucidsim import add_env
from imagination_to_real.lucidsim.tasks import parkour

DEFAULT_TIME_LIMIT = 25

PHYSICS_TIMESTEP = 0.005  # in XML
DECIMATION = 4
CONTROL_TIMESTEP = PHYSICS_TIMESTEP * DECIMATION

add_env(
    env_id="Stairs_v2",
    entrypoint=parkour.entrypoint,
    kwargs=dict(
        xml_path="stairs_v2.xml",
        mode="lucidsim",
        n_proprio=53,
        groups=[["step*"], ["cone*"]],
        x_noise=0,
        y_noise=0.1,
        use_cones=True,
    ),
)
