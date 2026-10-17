"""
Participant-group assignment for the size-annotation study: partitions (object, size) pairs into
balanced groups, each shown to several participants, so every (object, size) pair accumulates a
target number of independent human annotations.

Parametrized by object count on purpose -- this started at 10 novel objects (the only ones with
generated region/label assets at the time) and has now been bumped to the full 30-object novel-
object pool once the other 20 were fetched and processed (see generate_regions_sam2.py,
generate_study_object_assets.py). Nothing else in this script assumes a specific count, so it can
be bumped again later the same way.

THE DESIGN (confirmed with the user over two rounds):
  - Each group = OBJECTS_PER_GROUP DISTINCT objects, each shown at ONE size -- but which size is
    SHUFFLED per object within a group, not uniform across the whole group (a group is NOT "10
    objects all at hand-scale"; it's a mix, e.g. "obj2=medium, obj10=small, obj11=medium, ...",
    matching the example the user gave directly).
  - Every (object, size) cell appears in EXACTLY ONE group (no cell repeated across groups) --
    groups partition the full object x size grid, they don't overlap.
  - Every object appears in exactly len(SIZES) different groups (once per size), never twice in the
    same group.
  - Each group is then shown to ANNOTATIONS_PER_CELL different participants, so every (object, size)
    pair in it collects that many independent annotations.

This is a straightforward balanced-assignment / constraint-satisfaction problem, not a classic
Latin-square (object counts don't have to be a multiple of len(SIZES) for this construction to
work), solved here by randomized greedy placement with retry: for each object, place its
len(SIZES) cells into len(SIZES) DISTINCT, currently-least-loaded groups; retry with a new shuffle
if a placement ever gets stuck (rare -- the constraint is loose enough that this converges almost
immediately in practice, verified below).

Usage:
    python3 generate_participant_groups.py
Output: json/participant_groups.json -- {"groups": [[{"object":.., "size":..}, ...], ...],
"participants_needed": N, ...}
"""
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.realpath(__file__))
JSON_DIR = os.path.join(HERE, "json")
os.makedirs(JSON_DIR, exist_ok=True)

# ---- The only things to change when scaling the object pool up or down ----
OBJECTS = ["shape_1", "shape_2", "shape_3", "shape_10", "shape_11", "shape_12", "shape_13",
           "shape_14", "shape_15", "shape_16", "shape_17", "shape_18", "shape_19", "shape_20",
           "shape_21", "shape_22", "shape_23", "shape_24", "shape_25", "shape_26", "shape_27",
           "shape_28", "shape_29", "shape_30", "shape_31", "shape_32", "shape_33", "shape_38",
           "shape_39", "shape_40"]  # all 30 novel objects in the dataset (ids scattered 1-40)
SIZES = ["hand", "human", "building"]  # matches generate_size_charts.py's SCENARIOS labels
# How many distinct objects one participant session (one "group") covers -- 10, per the original
# discussion (30-object pool -> 9 groups). Earlier, with only 10 objects available, this was
# temporarily set to 5 so the partitioning logic was actually exercised instead of degenerating to
# "every group = all objects"; now that the pool is the full 30, back to the originally-designed 10.
OBJECTS_PER_GROUP = 10
ANNOTATIONS_PER_CELL = 4  # how many independent participants should see each (object, size) pair
SEED = 7  # fixed for reproducibility -- re-run with a different seed (or None) for a fresh shuffle


def build_groups(objects, sizes, objects_per_group, rng, max_attempts=2000):
    total_cells = len(objects) * len(sizes)
    if total_cells % objects_per_group != 0:
        raise ValueError(
            f"{len(objects)} objects x {len(sizes)} sizes = {total_cells} cells, not evenly "
            f"divisible by objects_per_group={objects_per_group} -- pick a divisor of {total_cells} "
            f"(or allow an uneven last group, not implemented here since it wasn't needed for 10 or "
            f"30 objects x 3 sizes)."
        )
    num_groups = total_cells // objects_per_group

    for attempt in range(1, max_attempts + 1):
        groups = [[] for _ in range(num_groups)]
        counts = [0] * num_groups
        objects_in_group = [set() for _ in range(num_groups)]
        order = objects[:]
        rng.shuffle(order)
        ok = True
        for o in order:
            szs = sizes[:]
            rng.shuffle(szs)
            for sz in szs:
                candidates = [
                    g for g in range(num_groups)
                    if counts[g] < objects_per_group and o not in objects_in_group[g]
                ]
                if not candidates:
                    ok = False
                    break
                min_count = min(counts[g] for g in candidates)
                tied = [g for g in candidates if counts[g] == min_count]
                g = rng.choice(tied)
                groups[g].append({"object": o, "size": sz})
                counts[g] += 1
                objects_in_group[g].add(o)
            if not ok:
                break
        if ok and all(c == objects_per_group for c in counts):
            return groups, attempt
    raise RuntimeError(f"Could not find a valid assignment in {max_attempts} attempts.")


def verify(groups, objects, sizes, objects_per_group):
    """Every claim this script makes about the assignment, checked directly rather than assumed."""
    all_cells = [(g["object"], g["size"]) for group in groups for g in group]
    assert len(all_cells) == len(set(all_cells)), "a cell was placed in more than one group"
    assert set(all_cells) == {(o, s) for o in objects for s in sizes}, "not every cell was covered"
    for group in groups:
        assert len(group) == objects_per_group, "group size mismatch"
        objs = [g["object"] for g in group]
        assert len(objs) == len(set(objs)), "an object repeats within one group"
    per_object_group_count = Counter()
    for gi, group in enumerate(groups):
        for g in group:
            per_object_group_count[g["object"]] += 1
    for o in objects:
        assert per_object_group_count[o] == len(sizes), f"{o} doesn't appear in exactly {len(sizes)} groups"


def main():
    rng = random.Random(SEED)
    groups, attempt = build_groups(OBJECTS, SIZES, OBJECTS_PER_GROUP, rng)
    verify(groups, OBJECTS, SIZES, OBJECTS_PER_GROUP)

    num_groups = len(groups)
    participants_needed = num_groups * ANNOTATIONS_PER_CELL

    print(f"{len(OBJECTS)} objects x {len(SIZES)} sizes = {len(OBJECTS) * len(SIZES)} cells")
    print(f"{OBJECTS_PER_GROUP} objects/group -> {num_groups} groups (found on attempt {attempt})")
    print(f"{ANNOTATIONS_PER_CELL} annotations/cell -> {participants_needed} participants needed total "
          f"({ANNOTATIONS_PER_CELL} per group)")
    print()
    for gi, group in enumerate(groups):
        size_mix = Counter(g["size"] for g in group)
        print(f"group {gi} ({dict(size_mix)}):")
        for g in sorted(group, key=lambda x: x["object"]):
            print(f"    {g['object']}: {g['size']}")

    out_path = os.path.join(JSON_DIR, "participant_groups.json")
    with open(out_path, "w") as f:
        json.dump({
            "objects": OBJECTS,
            "sizes": SIZES,
            "objects_per_group": OBJECTS_PER_GROUP,
            "annotations_per_cell": ANNOTATIONS_PER_CELL,
            "num_groups": num_groups,
            "participants_needed": participants_needed,
            "groups": groups,
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
