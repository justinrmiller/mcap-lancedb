#!/usr/bin/env bash
# Convert nuScenes (in data/nuscenes) to one MCAP file per scene (in data/mcap)
# with Foxglove's nuscenes2mcap. Extra arguments go to its convert_to_mcap.py,
# for example `--scene scene-0061`.
set -euo pipefail

converter=data/nuscenes2mcap
commit=0bcaab269379069b8a2df8ad4762f128c2552bb9

if [ ! -d "$converter" ]; then
    git clone -q https://github.com/foxglove/nuscenes2mcap.git "$converter"
fi
git -C "$converter" checkout -q "$commit"

# The converter needs Python 3.11. Its nuscenes-devkit pins matplotlib 3.5, which
# has no Python 3.11 wheel for Apple silicon and doesn't build from source;
# 3.7 works, and 3.8 drops a plot style the devkit uses.
uv venv -q --allow-existing --python 3.11 "$converter/.venv"
echo "matplotlib>=3.6,<3.8" > "$converter/overrides.txt"
uv pip install -q --python "$converter/.venv" \
    -r "$converter/pyproject.toml" --override "$converter/overrides.txt"

"$converter/.venv/bin/python" "$converter/convert_to_mcap.py" \
    --data-dir data/nuscenes --output-dir data/mcap "$@"
