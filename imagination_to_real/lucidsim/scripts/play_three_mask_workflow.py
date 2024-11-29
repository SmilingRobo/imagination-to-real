import random

from params_proto import ParamsProto, Proto



class PlayArgs(ParamsProto):
    env_name = Proto("stairs_v2", help="Environment name. One of (gaps, hurdle, parkour, stairs_v1, stairs_v2)")
    save_path = Proto(required=True, help="Path to save the video")

    prompt_collection = Proto(required=True,
                              help="Path to the prompt collection, absolute")

    num_steps = 500
    seed = 42


def play(**deps):
    from weaver.workflows.three_mask_workflow import ImagenCone
    from lucidsim.utils.utils import pick, make_video, image_grid
    from lucidsim import EXAMPLE_CHECKPOINTS_ROOT
    from lucidsim.traj_samplers import unroll_flow_stream as unroll
    from params_proto.hyper import Sweep

    PlayArgs._update(deps)

    random.seed(PlayArgs.seed)
    imagen = ImagenCone()

    gen = unroll.main(
        env_name=PlayArgs.env_name.capitalize(),
        checkpoint=f"{EXAMPLE_CHECKPOINTS_ROOT}/expert.pt",
        vision_key=None,
        render=True,
        num_steps=PlayArgs.num_steps,
        seed=PlayArgs.seed,
        baseline_interval=7,
        imagen_on=True,
    )

    prompts = Sweep.read(PlayArgs.prompt_collection)

    frames = []

    step = 0
    data = next(gen)

    while True:
        try:
            prompt = prompts[random.randint(0, len(prompts) - 1)]
            prompt = pick(prompt, "foreground_prompt", "background_prompt", "cone_prompt", "negative_prompt")

            generated_image = data.get("generated_image", None)
            if generated_image is None:
                generated_image = imagen.generate(
                    **pick(data, "midas_depth", "foreground_mask", "background_mask", "cone_mask"),
                    **prompt,
                )

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

            # Paste each top image next to each other at the top of the composite image
            for i, img in enumerate(resized_top_images):
                main_image.paste(img, (i * single_width, 0), 0)

            frame = image_grid([[main_image, generated_image]])
            frames.append(frame)

            data = gen.send(generated_image)
            step += 1

        except StopIteration:
            break

    make_video(frames, PlayArgs.save_path)


if __name__ == '__main__':
    play()
