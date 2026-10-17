# AGD20K vocabulary gaps

`AGD20K_VERBS` in `affordance_labeling.py` is the standard 36-verb AGD20K/OOAL set. Some objects in
our set have a real, primary affordance that set has no word for at all, which the forced-choice
classifier can't express no matter how well it's reasoning -- it either answers "none" honestly, or
picks the least-bad wrong option. This file tracks every gap found, so all additions beyond the
standard 36 verbs are deliberate and reviewed, not accumulated ad hoc.

Status values: **added** (in `AGD20K_VERBS` now) / **candidate** (found, not yet added, pending review).

| verb | status | evidence |
|---|---|---|
| `wear` | added | `fedora`: all 6 parts scored "none" ("no functional purpose... all parts seem equally important") -- a hat's only real affordance is being worn, and no AGD20K verb covers that at all. `bow-tie` shows the same pattern for the same reason. |
| `pull` | candidate | `fire-alarm`: the pull-handle/lever part (the object's actual activation mechanism) scored "take photo" -- there's a "push" but no "pull", so the model picks the least-bad wrong option instead of the honest "none" the fedora case got. |

## How to add one

1. Add the verb to `AGD20K_VERBS` in `affordance_labeling.py` (keep the "Additions beyond the
   standard..." comment block current).
2. Re-run `python batch_final_labeling.py <object> [...]` for the specific object(s) that motivated the
   addition (it overwrites their `<object>_affordance_justification.json`) -- a new vocabulary option changes
   that object's softmax normalization, so it needs fresh forced-choice scores, not just a fresh justification.
3. Note here whether re-running the full 33-object set was judged necessary (adding one implausible
   option among 36 has a real but typically small effect on unrelated objects' top-1 labels, since it
   only meaningfully competes for probability mass on objects where the existing 36 already fit
   poorly) or deferred as low-value for objects unaffected by the specific gap.
