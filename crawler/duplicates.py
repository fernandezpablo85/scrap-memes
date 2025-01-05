import os

seen_file = os.path.join(os.path.dirname(__file__), "seen.txt")


def already_seen(vox_id: str):
    """Read the seen vox IDs from a file and return them as a set."""
    with open(seen_file, "r") as f:
        seen = f.read().splitlines()
    return vox_id in seen
