from params_proto import ParamsProto, Proto

from imagination_to_real.lucidsim.utils.utils import make_video


class PlayArgs(ParamsProto):
    env_name = Proto("stairs_v2", help="Environment name. One of (gaps, hurdle, parkour, stairs_v1, stairs_v2)")
    save_path = Prot
    o(required=True, help="Path to save the video")

    num_steps = 500
    seed = 42


def play(**deps):
    from imagination_to_real.lucidsim import EXAMPLE_CHECKPOINTS_ROOT
    from imagination_to_real.lucidsim.traj_samplers import unroll_flow_stream as unroll

    PlayArgs._update(deps)
    gen = unroll.main(
        env_name=PlayArgs.env_name.capitalize(),
        checkpoint=f"{EXAMPLE_CHECKPOINTS_ROOT}/expert.pt",
        vision_key=None,
        render=True,
        num_steps=PlayArgs.num_steps,
        seed=PlayArgs.seed,
        baseline_interval=7,
        imagen_on=False,
    )

    frames = []
    for data in gen:
        single_width = 1280 // 5
        top_images = [
            data["foreground_mask"],
            data["background_mask"],
            data["cone_mask"],
            data["flow_image"],
            data["midas_depth"],
        ]

        main_image = data["render"].copy()

        resized_top_images = [
            img.resize((single_width, int(img.height * (single_width / 1280)))) for img in top_images
        ]

        for i, img in enumerate(resized_top_images):
            main_image.paste(img, (i * single_width, 0), 0)

        frames.append(main_image)

    make_video(frames, PlayArgs.save_path)


if __name__ == '__main__':
    play()
