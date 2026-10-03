import argparse

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

dataset = load_dataset("Ricky06662/VisionReasoner_multi_object_7k_840", revision="8056f125d15bda2c0cdc75ca84b47bbdd83dbfc7")
dataset.save_to_disk(args.output_dir)
