import os
import sys
import h5py
import numpy as np
from PIL import Image
from pathlib import Path

# Add parent dir to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from load_libero_dataset import download_libero_dataset, LIBERO_DATASETS
except ImportError:
    # Fallback definition if import fails
    LIBERO_DATASETS = {
        "libero_spatial": {},
        "libero_object": {},
        "libero_goal": {},
        "libero_10": {},
        "libero_90": {},
    }

    def download_libero_dataset(**kwargs):
        print(
            "Warning: load_libero_dataset module not found or download function unavailable."
        )
        return None


def visualize_libero_datasets(
    base_dir="./datasets/libero", output_dir="./results/libero_visualization"
):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)

    print(f"Base Directory: {base_dir.resolve()}")
    print(f"Output Directory: {output_dir.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all defined LIBERO suites
    suites = list(LIBERO_DATASETS.keys())

    total_images_saved = 0

    for suite in suites:
        print(f"\n[{suite}] Processing...")
        suite_dir = base_dir / suite

        # Check if dataset exists
        dataset_exists = suite_dir.exists() and (
            list(suite_dir.glob("*.hdf5")) or list(suite_dir.glob("**/*.hdf5"))
        )

        if not dataset_exists:
            print(f"[{suite}] Not found locally. Attempting to download...")
            try:
                # Assuming download_libero_dataset handles check and download
                # We pass the parent of suite_dir as local_dir argument to load_libero_dataset logic usually checks,
                # but download_libero_dataset takes 'local_dir' as the destination for the suite.
                # Let's check the signature in load_libero_dataset.py:
                # def download_libero_dataset(task_suite="libero_10", local_dir=None, ...)
                # if local_dir is None, it defaults to ./datasets/libero/task_suite

                download_libero_dataset(task_suite=suite, local_dir=suite_dir)

                # Recheck existence
                dataset_exists = suite_dir.exists() and (
                    list(suite_dir.glob("*.hdf5")) or list(suite_dir.glob("**/*.hdf5"))
                )
                if not dataset_exists:
                    print(
                        f"[{suite}] Download failed or no HDF5 files found after download. Skipping."
                    )
                    continue
            except Exception as e:
                print(f"[{suite}] Error downloading: {e}")
                continue

        # Create output subdirectory for this suite
        suite_out_dir = output_dir / suite
        suite_out_dir.mkdir(exist_ok=True)

        # Find all HDF5 files (tasks)
        hdf5_files = list(suite_dir.glob("*.hdf5"))
        if not hdf5_files:
            hdf5_files = list(suite_dir.glob("**/*.hdf5"))

        print(f"[{suite}] Found {len(hdf5_files)} tasks.")

        for hdf5_path in hdf5_files:
            task_name = hdf5_path.stem

            # Skip if already visualized (optional, but good for speed if re-running)
            # if (suite_out_dir / f"{task_name}.png").exists():
            #     continue

            try:
                with h5py.File(hdf5_path, "r") as f:
                    # Find the first demo
                    demos = []
                    if "data" in f:
                        demos = [k for k in f["data"].keys() if k.startswith("demo")]
                        data_root = f["data"]
                    else:
                        demos = [k for k in f.keys() if k.startswith("demo")]
                        data_root = f

                    if not demos:
                        print(f"  [WARN] {task_name}: No demos found.")
                        continue

                    # Sort look for demo_0 or similar
                    demos.sort()
                    first_demo = demos[0]
                    demo_grp = data_root[first_demo]

                    # Access observations
                    # Structure is typically demo_grp['obs'] -> datasets
                    obs = demo_grp["obs"] if "obs" in demo_grp else demo_grp

                    # Target image: agentview_rgb
                    image_data = None

                    # Priority list
                    keys_to_check = [
                        "agentview_rgb",
                        "agentview_image",
                        "robot0_agentview_rgb",
                        "robot0_agentview_image",
                        "eye_in_hand_rgb",  # Fallback
                        "robot0_eye_in_hand_rgb",
                    ]

                    for k in keys_to_check:
                        if k in obs:
                            image_data = obs[k][0]  # Get first frame
                            break

                    # If still not found, search anything with 'rgb' or 'image'
                    if image_data is None:
                        for k in obs.keys():
                            if ("rgb" in k or "image" in k) and isinstance(
                                obs[k], h5py.Dataset
                            ):
                                val = obs[k]
                                if len(val.shape) >= 3:
                                    image_data = val[0]
                                    break

                    if image_data is not None:
                        # Process image
                        # Expecting (H, W, C) or (C, H, W)
                        # Usually Libero is (H, W, 3) uint8 or (H, W) for depth?
                        # It's RGB, usually uint8

                        # If channels first (C, H, W), transpose
                        if image_data.shape[0] == 3 and image_data.shape[2] != 3:
                            image_data = np.transpose(image_data, (1, 2, 0))

                        # Check normalization
                        if image_data.max() <= 1.0 and image_data.dtype != np.uint8:
                            image_data = (image_data * 255).astype(np.uint8)

                        if image_data.dtype != np.uint8:
                            image_data = image_data.astype(np.uint8)

                        # Save
                        img = Image.fromarray(image_data)
                        save_path = suite_out_dir / f"{task_name}.png"
                        img.save(save_path)
                        total_images_saved += 1
                    else:
                        print(f"  [WARN] {task_name}: No image data found.")

            except Exception as e:
                print(f"  [ERROR] Failed to process {task_name}: {e}")

    print(f"\nDone. Saved {total_images_saved} visualization images to {output_dir}")


if __name__ == "__main__":
    visualize_libero_datasets()
