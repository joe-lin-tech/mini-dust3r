import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
from argparse import ArgumentParser
import torch
import matplotlib.pyplot as plt

from mini_dust3r.api import OptimizedResult, inference_dust3r, log_optimized_result
from mini_dust3r.model import AsymmetricCroCo3DStereo
import os
import numpy as np

def create_blueprint(image_name_list: list[str], log_path: Path) -> rrb.Blueprint:
    # dont show 2d views if there are more than 4 images as to not clutter the view
    len_images = len(os.listdir(image_name_list))
    
    if len_images > 4:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin=f"{log_path}"),
            ),
            collapse_panels=True,
        )
    else:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                contents=[
                    rrb.Spatial3DView(origin=f"{log_path}"),
                    rrb.Vertical(
                        contents=[
                            rrb.Spatial2DView(
                                origin=f"{log_path}/camera_{i}/pinhole/",
                                contents=[
                                    "+ $origin/**",
                                ],
                            )
                            for i in range(len_images)
                        ]
                    ),
                ],
                column_shares=[3, 1],
            ),
            collapse_panels=True,
        )
    return blueprint


def main(image_dir: Path):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = AsymmetricCroCo3DStereo.from_pretrained(
        "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)

    optimized_results: OptimizedResult = inference_dust3r(
        image_dir_or_list=image_dir,
        model=model,
        device=device,
        batch_size=1,
    )

    if not optimized_results:
        return False

    blueprint = create_blueprint(image_dir, "world")
    rr.send_blueprint(blueprint)

    ground_map = log_optimized_result(optimized_results, Path("world"))
    folder = str(image_dir).split('/')[-1]
    os.makedirs(f"../pedmotion/ground/{folder}", exist_ok=True)
    np.save(f"../pedmotion/ground/{folder}/{sorted(os.listdir(image_dir))[0]}".replace("jpg", "npy"), ground_map)
    plt.imshow(ground_map)
    plt.savefig(f"../pedmotion/ground/{folder}/{sorted(os.listdir(image_dir))[0]}")

    return True

if __name__ == "__main__":
    parser = ArgumentParser("mini-dust3r rerun demo script")
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images to process",
        required=True,
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "mini-dust3r")
    # main(args.image_dir)
    for dir in os.listdir(args.image_dir):
        dir = "jPx8Wgn9dOc_21"
        # if os.path.exists(f"../pedmotion/ground/{dir}"):
        #     continue
        folder = Path(f"{args.image_dir}/{dir}")
        # main(folder)
        if main(folder):
            break
    rr.script_teardown(args)