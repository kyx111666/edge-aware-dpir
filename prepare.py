"""Download missing upstream code, data and pretrained weights."""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def prepare_assets():
    manifest = json.loads((ROOT / "sources.json").read_text())
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            print("Downloading", item["path"], flush=True)
            try:
                with urllib.request.urlopen(item["url"], timeout=120) as response:
                    with temporary.open("wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    print("Assets ready.", flush=True)


if __name__ == "__main__":
    prepare_assets()
