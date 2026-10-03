import argparse

from datasets import DatasetDict, concatenate_datasets, load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("--visionreasoner_dataset_dir", required=True)
parser.add_argument("--instructpart_dataset_dir", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

merged = concatenate_datasets([
    load_from_disk(args.visionreasoner_dataset_dir)["train"],
    load_from_disk(args.instructpart_dataset_dir)["train"],
])
DatasetDict({"train": merged}).save_to_disk(args.output_dir)
