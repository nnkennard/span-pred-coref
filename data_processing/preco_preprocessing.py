import glob
import json
import os
import random
import sys

import convert_lib


def get_records_from_preco_file(filename):
  with open(filename, 'r') as f:
    return f.readlines()


def preprocess(data_dir):
  """Resplits PreCo files into train, dev, test splits."""

  preco_orig_dir = os.path.join(data_dir, "original", "PreCo_1.0")
  preco_dir = os.path.join(data_dir, "original", "preco")
  
  convert_lib.create_dir(preco_dir)

  resplit_datasets = {}

  resplit_datasets[
      convert_lib.DatasetSplit.test] = get_records_from_preco_file(
      os.path.join(preco_orig_dir, "dev.jsonl"))

  temp_original_train = get_records_from_preco_file(
      os.path.join(preco_orig_dir, "train.jsonl"))
  random.seed(43)
  random.shuffle(temp_original_train)
  total_train = len(temp_original_train)
  boundary = int(0.8 * total_train)
  resplit_datasets[
      convert_lib.DatasetSplit.train] = temp_original_train[:boundary]
  resplit_datasets[
      convert_lib.DatasetSplit.dev] = temp_original_train[boundary:]

  for split_name, records in resplit_datasets.items():
    # Write resplit
    with open(os.path.join(preco_dir, split_name + ".jsonl"), 'w') as f:
      f.write("".join(records))


def update_label(old_label, new_label):
  if old_label == "-":
    return new_label
  else:
    return old_label + "|" + new_label


def clusters_to_label_list(mention_clusters, sentences):
  labels = []
  for sentence in sentences:
    labels.append(["-"] * len(sentence))
  
  for i, cluster in enumerate(mention_clusters):
    for (sentence_idx, start, end) in cluster:
      inc_end = end - 1

      if start == inc_end:
        labels[sentence_idx][start] = update_label(
          labels[sentence_idx][start], "({})".format(i))
      else:
        labels[sentence_idx][start] = update_label(
          labels[sentence_idx][start], "({}".format(i))
        labels[sentence_idx][inc_end] = update_label(
          labels[sentence_idx][inc_end], "{})".format(i))
  return labels


CONLL_PLACEHOLDER = "_POS\t_PARSE\t_\t_\t_\t_SPEAKER\t*"


def jsonl_to_conll(filename):
  with open(filename, 'r') as f:
    assert filename.endswith(".jsonl")
    out_filename = filename.replace(".jsonl", ".txt")
    with open(out_filename, 'w') as g:
      for line in f:
        line_obj = json.loads(line)
        sentences = line_obj["sentences"]
        coref_labels = clusters_to_label_list(line_obj["mention_clusters"],
                                              sentences)
        document_lines = []
        for sentence, labels in zip(sentences, coref_labels):
          for i, (token, label) in enumerate(zip(sentence, labels)):
            document_lines.append("\t".join(
                  [line_obj["id"], "0", str(i), token, CONLL_PLACEHOLDER, label]))
          document_lines.append("")

        g.write(
        "#begin document ({}); part {}\n".format(line_obj["id"], 0))
        g.write("\n".join([line for line in document_lines]))
        g.write("\n#end document\n")
        

def convert_preco_to_conll(data_home):
  for filename in glob.glob(data_home + "/original/preco/*.jsonl"):
    jsonl_to_conll(filename)


def main():
  
  data_home = sys.argv[1]

  # Resplit the Preco files to get a train/dev/test split
  preprocess(data_home)

  # Convert preco jsonl format to conll format
  convert_preco_to_conll(data_home)


if __name__ == "__main__":
  main()
