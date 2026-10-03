import argparse
import glob
import json
import os
import statistics


def load_chunked(save_dir, basename):
    files = sorted(glob.glob(os.path.join(save_dir, f"{basename}_*.json")))
    if not files:
        single = os.path.join(save_dir, f"{basename}.json")
        if os.path.exists(single):
            files = [single]
    rows = []
    for f in files:
        with open(f) as fp:
            rows.extend(json.load(fp))
    return rows


def quality_block(rows, label):
    if not rows:
        return f"{label}: no rows found"
    inter_sum = sum(r["intersection"] for r in rows)
    union_sum = sum(r["union"] for r in rows)
    per_query_iou = [(r["intersection"] / r["union"]) if r["union"] > 0 else 0.0 for r in rows]
    g_iou = statistics.fmean(per_query_iou)
    c_iou = (inter_sum / union_sum) if union_sum > 0 else 0.0
    return (
        f"{label}\n"
        f"  entries: {len(rows)}\n"
        f"  gIoU (mean per-query IoU): {g_iou:.4f}\n"
        f"  cIoU (sum-pooled IoU):     {c_iou:.4f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", required=True)
    p.add_argument("--dataset", required=True, choices=["instructpart", "partimagenet", "pascalpart"])
    p.add_argument("--method", default="")
    args = p.parse_args()

    sections = [f"=== {args.method} on {args.dataset} ==="]
    if args.dataset == "pascalpart":
        sections.append(quality_block(load_chunked(args.save_dir, "objects_results"), "objects"))
    sections.append(quality_block(load_chunked(args.save_dir, "parts_results"), "parts"))

    text = "\n".join(sections) + "\n"
    with open(os.path.join(args.save_dir, "summary.txt"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
