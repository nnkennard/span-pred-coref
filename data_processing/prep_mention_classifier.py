import os
import random
import sys


import convert_lib

DATA_DIR_NAME = "mention_clf"

def main():
  data_path = sys.argv[1]

  for dataset in [convert_lib.DatasetName.conll]:
    output_dir = os.path.join(data_path, "processed", dataset, DATA_DIR_NAME)
    convert_lib.create_dir(output_dir)
    for seg_len in [convert_lib.ProcessingStage.SEGMENTED_384,
                    convert_lib.ProcessingStage.SEGMENTED_512]:
      input_train_jsonl = os.path.join(
          data_path, "processed", dataset, "train_" + seg_len + ".jsonl")

      with open(input_train_jsonl, 'r') as f:
        examples = f.readlines()
        random.shuffle(examples)
        train_end = int(0.6 * len(examples))
        dev_end = int(0.8 * len(examples))
        mention_splits = {
        convert_lib.DatasetSplit.train: examples[:train_end],
        convert_lib.DatasetSplit.dev: examples[train_end:dev_end],
        convert_lib.DatasetSplit.test: examples[dev_end:]
        }
        for set_name, example_lines in mention_splits.items():
          output_file = "".join([
              output_dir, "/mention_", set_name, "_", seg_len, ".jsonl"])
          with open(output_file, 'w') as g:
            g.write("\n".join(example_lines))


if __name__ == "__main__":
  main()
