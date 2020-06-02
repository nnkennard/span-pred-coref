import collections

def listify_conll_dataset(filename):

  dataset = []
  curr_doc = []
  curr_sent = []

  with open(filename, 'r') as f:
    for line in f:
      if "\t" in line:
        fields = line.strip().split("\t")
      else:
        fields = line.strip().split()

      if line.startswith("#begin"):
        assert not curr_doc 
        curr_doc.append([fields])
      
      elif line.startswith("#end"):
        curr_doc.append([fields])
        dataset.append(curr_doc)
        curr_doc = []
      
      elif not line.strip():
        if curr_sent:
          curr_doc.append(curr_sent)
          curr_sent = []
          
      else: # Empty line signifies the end of a sentence
        curr_sent.append(fields)

  return dataset


def write_listified_dataset_to_file(dataset, filename):
  with open(filename, 'w') as g:
    for document in dataset:
      for sentence in document:
        if len(sentence) == 1 and sentence[0][0].startswith("#"):
          g.write(" ".join(sentence[0]) + "\n")
        else:
          for token in sentence:
            g.write("\t".join(token) + "\n")
          g.write("\n")

def ldd_append(ldd, to_append):
  for k, v in to_append.items():
    ldd[k] += v
  return ldd


class LabelSequences(object):
  WORD = "WORD"
  POS = "POS"
  NER = "NER"
  PARSE = "PARSE"
  COREF = "COREF"
  SPEAKER = "SPEAKER"
  PREDPOS = "PREDPOS"
  PREDPARSE = "PREDPARSE"

MINICONLL_FIELD_MAP = {
  LabelSequences.WORD: 3,
  LabelSequences.POS: 4, 
  LabelSequences.PARSE: 5, 
  LabelSequences.SPEAKER: 9, 
  LabelSequences.COREF: 10,
  LabelSequences.PREDPOS: 11,
  LabelSequences.PREDPARSE: 12,
}

def get_index(list_of_lists, index):
  return [record[index] for record in list_of_lists]

def get_sequences(sentence, field_map=MINICONLL_FIELD_MAP):
  sequences = {}
  for field, index in field_map.items():
    sequences[field] = get_index(sentence, index)
  return sequences


def build_coref_span_map(coref_col, offset=0):
  span_starts = collections.defaultdict(list)
  complete_spans = []
  for i, orig_label in enumerate(coref_col):
    if orig_label == '-': # no coref label
      continue
    else:
      labels = orig_label.split("|") # split for multiple (nested) case
      for label in labels:
        if label.startswith("("): # Span start
          if label.endswith(")"): # Single-token span
            complete_spans.append((i, i, label[1:-1]))
          else:
            span_starts[label[1:]].append(i) # Register span start for later
        elif label.endswith(")"):
          ending_cluster = label[:-1] # Which cluster is ending here
          assert len(span_starts[ending_cluster]) in [1, 2, 3]
          # Sometimes it's closing a nested span but apparently never more than
          # three levels for the same entity
          start_idx = span_starts[ending_cluster].pop(-1)
          # The one added latest is the match
          complete_spans.append((start_idx, i, ending_cluster))

  span_dict = collections.defaultdict(list)
  for start, end, cluster in complete_spans:
    span_dict[cluster].append((offset + start, offset + end))
    # offset is the token offset of the sentence within the document
  return span_dict


def split_parse_label(label):
  curr_chunk = ""
  chunks = []
  for c in label:
    if c in "()": # A chunk is everything up to a paren
      if curr_chunk:
        chunks.append(curr_chunk)
      curr_chunk = c
    else:
      curr_chunk += c
  chunks.append(curr_chunk)
  return chunks


def build_parse_span_map(parse_col, offset=0):

  if set(parse_col) == set(["_PARSE"]): # This is for empty sentences and empty gold parses in PreCo
    return {}

  span_starts = collections.defaultdict(list)
  stack = []
  label_map = {}
  for i, orig_label in enumerate(parse_col):
    labels = split_parse_label(orig_label) # Chunking around parens
    for label in labels:
      if label.startswith("("): # Register start of a label
        stack.insert(0, [label, i + offset]) # Goes on the top of the stack
        # ^ build up label in [0], remember start (with offset) in [1]
      elif label.endswith(")"): # End of chunk, hopefully start was registered
        span_prefix, start_idx = stack.pop(0)
        assert (span_prefix, i) not in label_map # This is an unclosed span
        label_map[
            (start_idx, i + offset)] = span_prefix + label # Label is suffix
      else:
        stack[0][0] += label # This is part of the label we're currently collecting

  return label_map
