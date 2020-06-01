"""Just some random things to make things easier."""
import collections
import json
import numpy as np
import os
import tqdm

from bert import tokenization

VOCAB_FILE = "/mnt/nfs/scratch1/nnayak/mention_bottleneck/convert/cased_config_vocab/vocab.txt"
TOKENIZER = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=False)

CLS = "[CLS]"
SPL = "[SPL]"
SEP = "[SEP]"


class DatasetName(object):
  conll = 'conll'
  preco = 'preco'
  ALL = [conll, preco]

class Variation(object):

  classic = 'classic'
  gold = 'gold'

  goldconst = 'goldconst'
  predconst = 'predconst'

  goldsing = 'goldsing'
  predsing = 'predsing'

class DatasetSplit(object):
  train = 'train'
  dev = 'dev'
  test = 'test'
  ALL = [train, dev, test]


def create_dir(path):
  try:
      os.makedirs(path)
  except OSError:
      print ("Creation of the directory %s failed" % path)
  else:
      print ("Successfully created the directory %s " % path)

class Dataset(object):
  def __init__(self, dataset_name):
    self.name = dataset_name
    self.documents = collections.defaultdict(list)

  def dump_to_jsonl(self, file_name):

    assert ProcessingStage.TOKENIZED in self.documents

    if ProcessingStage.BPE_TOKENIZED not in self.documents:
      print("BPE tokenizing documents")
      for document in tqdm.tqdm(self.documents[ProcessingStage.TOKENIZED]):
        self.documents[ProcessingStage.BPE_TOKENIZED].append(
          bpe_tokenize_document(document, TOKENIZER))

    for new_stage in [
      ProcessingStage.SEGMENTED_384, ProcessingStage.SEGMENTED_512]:
      if new_stage not in self.documents:
        print("Resegmenting documents (", new_stage, ")")
        for document in tqdm.tqdm(self.documents[ProcessingStage.BPE_TOKENIZED]):
          self.documents[new_stage].append(segment_document(document, new_stage))

      lines = [doc.dump_to_json() for doc in self.documents[new_stage]]
    
      assert file_name.endswith(".jsonl")
      seg_filename =  file_name.replace(".jsonl", "_" + new_stage + ".jsonl")
      create_dir(seg_filename.rsplit("/", 1)[0])
      with open(seg_filename, 'w') as f:
        f.write("\n".join(lines))

  
def flatten(nonflat):
  return sum(nonflat,[])


class ProcessingStage(object):
  UNINITIALIZED = "UNINITIALIZED"
  TOKENIZED = "TOKENIZED"
  BPE_TOKENIZED = "BPE_TOKENIZED"
  SEGMENTED_384 = "SEGMENTED_384"
  SEGMENTED_512 = "SEGMENTED_512"


class CorefDocument(object):
  def __init__(self, doc_id, part, init_status=ProcessingStage.UNINITIALIZED):

    self.doc_id = doc_id
    self.doc_part = part
    self.status=init_status

    self.clusters = []
    self.injected_mentions = {}
    self.sentences = []
    self.speakers = []

    self.subtoken_map = []
    self.sentence_map = []

    self.token_sentences = [] # TODO: remove this?


  def dump_to_json(self):
    assert self.status in [ProcessingStage.SEGMENTED_512,
        ProcessingStage.SEGMENTED_384]
 
    return json.dumps({
          "doc_key": self.doc_id + "_" + str(int(self.doc_part)),
          "sentences": self.sentences,
          "sentence_map": self.sentence_map,
          "subtoken_map": self.subtoken_map,
          "speakers": self.speakers,
          "clusters": self.clusters,
          "inject_mentions": self.injected_mentions,
          "token_sentences": self.token_sentences,
          "bpe_maps": self.bpe_maps,
          "subtoken_offsets": self.subtoken_offsets,
          "format": self.status,
        })

  def _update_label(self, original_label, additional_label):
    if original_label == "-":
      return additional_label
    else:
      return original_label + "|" + additional_label


def all_same(l):
  return len(set(l)) == 1

def same_len(l):
  return all_same(len(i) for i in l)

def bpe_tokenize_document(document, tokenizer):
  assert document.status == ProcessingStage.TOKENIZED

  bpe_document = CorefDocument(document.doc_id, document.doc_part,
                    ProcessingStage.BPE_TOKENIZED)

  bpe_document.token_sentences = document.sentences
  bpe_document.token_clusters = document.clusters

  token_to_starting_subtoken = []
  token_to_ending_subtoken = []
  cum_doc_token_count = 0
  previous_token = 0

  for sentence_idx, sentence in enumerate(document.sentences):

    # All tokens in the sentence should have the same speaker, just checking
    multi_speakers = document.speakers[sentence_idx]
    assert all_same(multi_speakers)
    speaker, = tuple(set(multi_speakers))

    subword_list = [TOKENIZER.tokenize(token) for token in sentence]

    # Construct mapping to subtoken for use in cluster stuff later
    subtoken_offset = len(bpe_document.subtoken_map) # subtokens included so far
    for i, (token, subwords) in enumerate(zip(sentence, subword_list)):
      token_to_starting_subtoken.append(subtoken_offset)
      subtoken_offset += len(subwords)
      token_to_ending_subtoken.append(subtoken_offset - 1) # inclusive

    # For each subword, which original token did it come from (index from flat list)
    subtoken_map = flatten(
      [
        [in_sentence_token_idx + cum_doc_token_count] * len(token_subwords)
        for in_sentence_token_idx, token_subwords in
        enumerate(subword_list)])

    flattened_subword = sum(subword_list, [])

    # Build various fields
    bpe_document.sentences.append(flattened_subword)
    bpe_document.subtoken_map += subtoken_map
    bpe_document.speakers.append([speaker] * len(flattened_subword))

    # For each subtoken, which sentence did it come from (by idx)
    bpe_document.sentence_map += [sentence_idx] * len(flattened_subword) 

    cum_doc_token_count += len(sentence)

  assert same_len([token_to_ending_subtoken, token_to_starting_subtoken,
                   flatten(document.sentences)])
  assert same_len([bpe_document.sentence_map,
                   bpe_document.subtoken_map, flatten(bpe_document.sentences)])
           

  # Remap clusters

  bpe_document.clusters = remap_clusters(document.clusters,
                                         token_to_starting_subtoken,
                                         token_to_ending_subtoken,
                                         cumulative=False)

  for injected_type, mentions in document.injected_mentions.items():
    (bpe_document.injected_mentions[injected_type], ) = remap_clusters(
      [mentions], token_to_starting_subtoken, token_to_ending_subtoken,
      cumulative=False)

  bpe_document.bpe_maps = [token_to_starting_subtoken,
                           token_to_ending_subtoken]
                            
  
  return bpe_document
  

STAGE_TO_LEN ={ProcessingStage.SEGMENTED_384: 384,
               ProcessingStage.SEGMENTED_512: 512}

def remap_clusters(clusters, start_offsets, optional_end_offsets=None,
                   cumulative=False):
  """Remap clusters. If cumulative, add offset to indices."""
  if optional_end_offsets is None:
    end_offsets = start_offsets
  else:
    end_offsets = optional_end_offsets

  new_clusters = []
  for cluster in clusters:
    new_cluster = []
    for start, end in cluster:
      if cumulative:
        new_start = start + start_offsets[start]
        new_end = end + end_offsets[end]
      else:
        new_start = start_offsets[start]
        new_end = end_offsets[end]
      new_cluster.append([new_start, new_end])
    new_clusters.append(new_cluster)

  return new_clusters


def segment_document(bpe_document, new_stage):
  assert bpe_document.status == ProcessingStage.BPE_TOKENIZED
  max_segment_len = STAGE_TO_LEN[new_stage]

  seg_document = CorefDocument(
      bpe_document.doc_id, bpe_document.doc_part, new_stage)

  seg_document.token_sentences = bpe_document.token_sentences
  seg_document.token_clusters = bpe_document.token_clusters
  seg_document.bpe_maps = bpe_document.bpe_maps

  # For each segment, a list of sentence indices which are part of that segment
  segment_maps = []
  current_segment = []
  current_segment_len = 0
  
  # Building segment maps
  for i, sentence in enumerate(bpe_document.sentences):
    # 2 is added for CLS and SEP
    if len(sentence) + current_segment_len + 2 <= max_segment_len:
      current_segment.append(i)
      current_segment_len += len(sentence)
    else:
      segment_maps.append(current_segment)
      current_segment = [i]
      current_segment_len = len(sentence)
  if current_segment:
    segment_maps.append(current_segment)


  # For each subtoken, how many indices it is bumped by due to CLS and SEP tokens
  subtoken_offsets = []
  subtoken_offset = 0

  prev_subtoken = 0
  for sentence_indices in segment_maps:
    start_sentence_idx = sentence_indices[0]
    exc_end_sentence_idx = sentence_indices[-1] + 1
    first_subtoken_index = len(flatten(bpe_document.sentences[:start_sentence_idx]))
    last_subtoken_index = len(flatten(bpe_document.sentences[:exc_end_sentence_idx]))

    segment = [CLS] + flatten(bpe_document.sentences[start_sentence_idx:exc_end_sentence_idx]) + [SEP]
    seg_document.sentences.append(segment)
    seg_document.speakers.append(
      [SPL] + flatten(bpe_document.speakers[start_sentence_idx:exc_end_sentence_idx]) + [SPL])

    seg_document.subtoken_map.append( # CLS takes previous subtoken index
        prev_subtoken)
    seg_document.subtoken_map +=  bpe_document.subtoken_map[first_subtoken_index:last_subtoken_index]
    this_sentence_last_subtoken = bpe_document.subtoken_map[last_subtoken_index - 1]
    seg_document.subtoken_map.append( # Presumably SEP shares index of last word
        this_sentence_last_subtoken)
    prev_subtoken = this_sentence_last_subtoken

    seg_document.sentence_map.append( # CLS shares index of first word
        bpe_document.sentence_map[first_subtoken_index])
    seg_document.sentence_map +=  bpe_document.sentence_map[first_subtoken_index:last_subtoken_index]
    seg_document.sentence_map.append( # SEP has index of presumptive next sentence
        bpe_document.sentence_map[last_subtoken_index -  1] + 1)

    subtoken_offset += 1 # for the CLS token
    subtoken_offsets += [subtoken_offset] * (len(segment) - 2)
    subtoken_offset += 1 # for the SEP token
  
   
  assert same_len([subtoken_offsets,
                   flatten(bpe_document.sentences)])
  assert same_len([seg_document.sentence_map,
                   seg_document.subtoken_map, flatten(seg_document.sentences)])

  seg_document.clusters = remap_clusters(bpe_document.clusters,
                                         subtoken_offsets,
                                         cumulative=True)

  for injected_type, mentions in bpe_document.injected_mentions.items():
    (seg_document.injected_mentions[injected_type], ) = remap_clusters(
      [mentions], subtoken_offsets, cumulative=True)
  
  seg_document.subtoken_offsets = subtoken_offsets
                             
  return seg_document
    
 
def write_converted(dataset, prefix):
  dataset.dump_to_jsonl(prefix + ".jsonl")
