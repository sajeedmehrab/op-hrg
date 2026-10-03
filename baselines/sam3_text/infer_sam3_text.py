import argparse
import json
import os
import sys
import time

import torch
from transformers import Sam3Model, Sam3Processor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks"))
from benchmark_datasets import DATASET_CHOICES, batched, compute_iou, iter_records, mask_area, result_paths


def union_of_instances(masks):
    if masks.dtype != torch.bool:
        masks = masks > 0.5
    return masks.any(dim=0).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p.add_argument("--benchmark_root", required=True)
    p.add_argument("--model_path", default="facebook/sam3")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--chunk_id", type=int, default=None)
    p.add_argument("--num_chunks", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--mask_threshold", type=float, default=0.5)
    p.add_argument("--max_units", type=int, default=None)
    p.add_argument("--save_every", type=int, default=200)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    paths = result_paths(args.save_dir, args.chunk_id)
    rows = {"parts": [], "objects": []}

    def flush():
        for kind, path in paths.items():
            if rows[kind]:
                with open(path, "w") as f:
                    json.dump(rows[kind], f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained(args.model_path).to(device)
    model.eval()
    processor = Sam3Processor.from_pretrained(args.model_path)

    records = iter_records(args.dataset, args.benchmark_root, chunk_id=args.chunk_id, num_chunks=args.num_chunks, max_units=args.max_units)

    n_written = 0
    n_empty = 0
    t_start = time.time()
    for batch in batched(records, args.batch_size):
        inputs = processor(
            images=[r["image"] for r in batch],
            text=[r["query"] for r in batch],
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )
        assert len(results) == len(batch)

        for rec, res in zip(batch, results):
            row = {
                "image_id": rec["image_id"],
                "query": rec["query"],
                "object_name": rec.get("object_name"),
                "part_name": rec.get("part_name"),
            }
            if args.dataset == "partimagenet":
                row["annotation_idx"] = rec.get("annotation_idx")
            if len(res["masks"]) == 0:
                n_empty += 1
                row.update({"n_instances": 0, "intersection": 0, "union": mask_area(rec["gt_mask"]), "iou": 0.0})
            else:
                intersection, union = compute_iou(union_of_instances(res["masks"]), rec["gt_mask"])
                row.update({
                    "n_instances": int(len(res["masks"])),
                    "intersection": intersection,
                    "union": union,
                    "iou": (intersection / union) if union > 0 else 0.0,
                })
            rows[rec["result_type"]].append(row)
            n_written += 1

        if n_written % args.save_every < args.batch_size:
            flush()
            print(f"[{args.dataset} chunk {args.chunk_id}] {n_written} rows ({n_empty} empty) {(time.time()-t_start)/60:.1f} min", flush=True)
        del inputs, outputs, results
        torch.cuda.empty_cache()

    flush()
    print(f"[{args.dataset} chunk {args.chunk_id}] done: {n_written} rows, {n_empty} empty predictions", flush=True)


if __name__ == "__main__":
    main()
