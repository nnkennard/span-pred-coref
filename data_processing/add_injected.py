import collections
import os
import sys
import tqdm

import conll_lib
import convert_lib


INJECTED_SET = {
  convert_lib.DatasetName.conll: [
      convert_lib.Variation.classic,
      convert_lib.Variation.gold,
      convert_lib.Variation.goldconst,
      convert_lib.Variation.predconst,
      convert_lib.Variation.predsing,
],
  convert_lib.DatasetName.preco: [
      convert_lib.Variation.classic,
      convert_lib.Variation.gold,
      convert_lib.Variation.predconst,
      convert_lib.Variation.goldsing,
      convert_lib.Variation.predsing,
],
}

def flatten(nonflat):
  return sum([list(i) for i in nonflat], [])

def non_singletons(spans):
  return [span for span in spans if len(span) > 1]

def get_classic_injected_mentions(doc_level_maps):
  return [] 

def get_gold_injected_mentions(doc_level_maps):
  return flatten(non_singletons(doc_level_maps["COREF"].values()))

def get_predconst_injected_mentions(doc_level_maps):
  return list(doc_level_maps["PREDPARSE"].keys())

def get_goldconst_injected_mentions(doc_level_maps):
  return list(doc_level_maps["PARSE"].keys())

def get_predsing_injected_mentions(doc_level_maps):
  max_token = max(flatten(list(doc_level_maps["PREDPARSE"].keys())))
  token_mentions = [(i, i) for i in range(max_token + 1)]
  np_mentions = [key for key, value in doc_level_maps["PREDPARSE"].items() if value.startswith("(NP")]
  return sorted(list(set(token_mentions).union(set(np_mentions))))

def get_goldsing_injected_mentions(doc_level_maps):
  assert len(coref_spans) > len(non_singletons(coref_spans))
  return sum(coref_spans, [])


FN_MAP = {
  convert_lib.Variation.classic: get_classic_injected_mentions,
  convert_lib.Variation.gold: get_gold_injected_mentions,

  convert_lib.Variation.goldconst: get_goldconst_injected_mentions,
  convert_lib.Variation.goldsing: get_goldsing_injected_mentions,

  convert_lib.Variation.predconst: get_predconst_injected_mentions,
  convert_lib.Variation.predsing: get_predsing_injected_mentions,
}


def add_sentence(curr_doc, curr_sent, doc_level_maps,
                 sentence_offset):
  sequences = conll_lib.get_sequences(curr_sent, conll_lib.MINICONLL_FIELD_MAP)
  curr_doc.speakers.append(sequences[conll_lib.LabelSequences.SPEAKER])
  curr_doc.sentences.append(sequences[conll_lib.LabelSequences.WORD])

  coref_span_map = conll_lib.build_coref_span_map(
      sequences[conll_lib.LabelSequences.COREF], sentence_offset)

  doc_level_maps["COREF"] = conll_lib.ldd_append(doc_level_maps["COREF"], coref_span_map)

  pred_parse_span_map = conll_lib.build_parse_span_map(
      sequences[conll_lib.LabelSequences.PREDPARSE], sentence_offset)
  doc_level_maps["PREDPARSE"].update(pred_parse_span_map)

  parse_span_map = conll_lib.build_parse_span_map(
      sequences[conll_lib.LabelSequences.PARSE], sentence_offset)
  doc_level_maps["PARSE"].update(parse_span_map)
  
  sentence_offset += len(sequences[conll_lib.LabelSequences.WORD])

  return doc_level_maps, sentence_offset


def convert(document, dataset_name):

  sentence_offset = 0
  doc_level_maps = {
    conll_lib.LabelSequences.COREF: collections.defaultdict(list),
    conll_lib.LabelSequences.PARSE: collections.defaultdict(list),
    conll_lib.LabelSequences.PREDPARSE: collections.defaultdict(list),
  }

  # Get document metadata from begin line
  begin_line = document[0][0]
  assert begin_line[0] == "#begin"
  curr_doc_id = begin_line[2][1:-2]
  curr_doc_part = begin_line[-1]

  curr_doc = convert_lib.CorefDocument(
      curr_doc_id, curr_doc_part,
      init_status=convert_lib.ProcessingStage.TOKENIZED)

  sentences = document[1:-1] # Excluding the #begin and #end lines
  for sentence in sentences:
    doc_level_maps, sentence_offset = add_sentence(
      curr_doc, sentence, doc_level_maps, sentence_offset)

  curr_doc.clusters = list(doc_level_maps["COREF"].values())

  injected_mentions = {}
  for injected_type in INJECTED_SET[dataset_name]: 
    injected_mentions[injected_type] = FN_MAP[injected_type](doc_level_maps)

  curr_doc.injected_mentions = injected_mentions

  return curr_doc


def main():
  data_home = sys.argv[1]

  for dataset in convert_lib.DatasetName.ALL:
    for subset in convert_lib.DatasetSplit.ALL:
      input_file = os.path.join(data_home, "original", dataset,
                                subset + ".miniconll")
      print(dataset, subset)
      listified_dataset = conll_lib.listify_conll_dataset(input_file)
      new_dataset = convert_lib.Dataset(dataset + "_" + subset)
      for document in tqdm.tqdm(listified_dataset):
        converted_document = convert(document, dataset)
        new_dataset.documents[convert_lib.ProcessingStage.TOKENIZED].append(converted_document)


      output_file = os.path.join(data_home, "processed", dataset,
                                subset + ".jsonl")
      new_dataset.dump_to_jsonl(output_file)


if __name__ == "__main__":
  main()
