"""
Run the full region information-gain pipeline for one object/image and automatically
produce the four-panel visualization -- so you don't have to chain the individual
scripts by hand.

Runs, in order:
    region_information_gain.py <object> [max_regions]
    region_information_gain_size_controlled.py <object> [K] [max_regions]
    visualize_four_panel.py <object>

Requires SAM regions to already exist at regions_test/output/sam_masks/<object>.json --
run generate_regions.py first if they don't (see regions_test/README.md).

Usage:
    python run_all.py <object_name> [K] [max_regions]

    K            number of random-translation controls per region (default 6)
    max_regions  cap analysis to the N largest SAM regions -- useful for busy photos
                 with many small regions (e.g. door.jpg's 50); omit for no cap
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.realpath(__file__))
REGIONS_TEST = os.path.join(os.path.dirname(HERE), "regions_test")


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    # cwd is NOT forced to HERE -- inherits the caller's actual working directory, so
    # the visualization scripts' `os.getcwd()`-based output paths save where the user
    # is actually standing, not buried in affordance-test/. Script paths are absolute
    # so this doesn't break locating them.
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    object_name = sys.argv[1]
    k = sys.argv[2] if len(sys.argv) > 2 else "6"
    max_regions = sys.argv[3] if len(sys.argv) > 3 else None

    sam_path = os.path.join(REGIONS_TEST, "output", "sam_masks", f"{object_name}.json")
    if not os.path.isfile(sam_path):
        print(f"No SAM regions found at {sam_path}.")
        print("Run generate_regions.py on this image first, e.g.:")
        print(
            f"  python ../regions/region_features/generate_regions.py --input <path to image> "
            f"--output {os.path.join(REGIONS_TEST, 'output', 'sam_masks')} "
            f"--checkpoint <sam checkpoint .pth> --model-type vit_b --convert-to-rle"
        )
        sys.exit(1)

    naive_cmd = [sys.executable, os.path.join(HERE, "region_information_gain.py"), object_name]
    controlled_cmd = [
        sys.executable, os.path.join(HERE, "region_information_gain_size_controlled.py"), object_name, k,
    ]
    if max_regions is not None:
        naive_cmd.append(max_regions)
        controlled_cmd.append(max_regions)

    run(naive_cmd)
    run(controlled_cmd)
    run([sys.executable, os.path.join(HERE, "visualize_four_panel.py"), object_name])

    print(f"\nDone. See figures/{object_name}_four_panel.png")


if __name__ == "__main__":
    main()
