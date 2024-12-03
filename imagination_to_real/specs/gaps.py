from imagination_to_real.lucidsim import EXAMPLE_CHECKPOINTS_ROOT
from imagination_to_real.lucidsim.traj_samplers import unroll_flow_stream as unroll

gen = unroll.main(
    env_name="Gaps",
    checkpoint=f"{EXAMPLE_CHECKPOINTS_ROOT}/expert.pt",
    vision_key=None,
    render=True,
    num_steps=500,
    seed=30,
    imagen_on=False,
)

for data in gen:
    pass
