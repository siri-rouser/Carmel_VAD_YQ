import argparse
from pathlib import Path
import re

PATTERN = re.compile(r"^(?P<intersection>.+)_av_(?P<idx>\d+)_(?P<label>-?\d+)_(?P<severity>-?\d+)\.mp4$")

def main():
    ap = argparse.ArgumentParser(description="Remove OTA videos by category (label).")
    ap.add_argument("--root", required=True, help="Root folder to search (recursively).")
    ap.add_argument("--cats", default="-1",
                    help="Comma-separated label IDs to remove, e.g. 1,4,5,7")
    ap.add_argument("--dry", action="store_true", help="Dry-run (show what would be removed).")
    ap.add_argument("--exts", default=".mp4,.MP4", help="Comma-separated extensions.")
    args = ap.parse_args()

    root = Path(args.root)
    cats = {int(x.strip()) for x in args.cats.split(",") if x.strip()}
    exts = [e.strip() for e in args.exts.split(",") if e.strip()]

    total, removed, skipped = 0, 0, 0
    for ext in exts:
        for p in root.rglob(f"*{ext}"):
            total += 1
            m = PATTERN.match(p.name)
            if not m:
                print(p)
                skipped += 1
                continue
            label = int(m.group("label"))
            if label in cats:
                print(("DRY  " if args.dry else "DEL  ") + str(p))
                if not args.dry:
                    try:
                        p.unlink()
                        removed += 1
                    except Exception as e:
                        print(f"  [WARN] Failed to delete {p}: {e}")

    print("\nSummary:")
    print(f"  Scanned:  {total}")
    print(f"  Skipped (name not matching pattern): {skipped}")
    print(f"  Removed:  {removed if not args.dry else '(dry-run)'}")

if __name__ == "__main__":
    main()

# python filter_testvideos.py --root ./OTA/RangelineS116thSt/testdata_selected/videos/ --cats="12"