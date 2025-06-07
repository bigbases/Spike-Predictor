import torch
import torch.nn as nn
# import pickle
import argparse
import torch.nn.functional as F
import torch.optim as optim
from model import new_spikformer, new_spikformer_legacy
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset # TensorDataset added
from tqdm import tqdm
import numpy as np
import os
import sys # Added for DataProcessors
import csv # Added for DataProcessors
# import json # Removed
from transformers import BertTokenizer #, default_data_collator, DataCollatorWithPadding
from spikingjelly.activation_based import functional
# import math # Removed as unused
from utils.public import set_seed
from datasets import Dataset # Changed from load_dataset, ClassLabel

print(torch.__version__)
# csv.field_size_limit(sys.maxsize) # Potentially needed by processors, keep if issues arise

# Removed task_to_keys dictionary

# <----------------- INSERTED FROM spiking_bert_task_distill.py (modified) ----------------->
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
    
    @classmethod
    def _read_arrow(cls, input_file):
        """Reads an Arrow file using Hugging Face datasets."""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Expected Arrow file not found: {input_file}")
        return Dataset.from_file(input_file)


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return [None] # Regression task

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: # skip header
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                print(f"Skipping line {i+1} in {set_type} for QQP due to IndexError: {line}")
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched") # Or "dev"

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1] # question
            text_b = line[2] # sentence
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
        
class CBProcessor(DataProcessor):
    """Processor for the CommitmentBank (CB) dataset (SuperGLUE version)."""
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_arrow(os.path.join(data_dir, "super_glue-train.arrow")), "train") # Assuming train.arrow or similar

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_arrow(os.path.join(data_dir, "super_glue-validation.arrow")), "dev") # Assuming val.arrow or similar

    def get_labels(self):
        # Based on SuperGLUE, CB has 3 labels often mapped to 0, 1, 2
        return ["0", "1", "2"] # Or ["0", "1", "2"] if labels are numeric

    def _create_examples(self, dataset_items, set_type):
        examples = []
        # Arrow dataset from HF `datasets` is an iterable of dicts
        for i, item in enumerate(dataset_items):
            guid = "%s-%s" % (set_type, item.get("idx", i)) 
            text_a = item["premise"]
            text_b = item["hypothesis"]
            # Ensure label is string as get_labels() returns strings.
            # SuperGLUE CB labels might be 'entailment', 'contradiction', 'neutral' or 0,1,2
            # This mapping might be needed depending on raw data.
            # For now, assume item["label"] is already in the form expected by label_map
            label = str(item["label"]) 
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class SickProcessor(DataProcessor):
    """Processor for the SICK data set."""
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_arrow(os.path.join(data_dir, "sick-train.arrow")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_arrow(os.path.join(data_dir, "sick-validation.arrow")), "dev")

    def get_labels(self):
        # SICK labels are often 'NEUTRAL', 'ENTAILMENT', 'CONTRADICTION' or mapped to 0,1,2
        # Example: return ["NEUTRAL", "ENTAILMENT", "CONTRADICTION"]
        return ["1", "0", "2"] # Assuming numeric string labels for consistency with other GLUE

    def _create_examples(self, dataset_items, set_type):
        examples = []
        for i, item in enumerate(dataset_items):
            guid = f"{set_type}-{i}"
            text_a = item["text1"] # Or "text1", "premise" - check SICK arrow schema
            text_b = item["text2"] # Or "text2", "hypothesis"
            label = str(item["label"]) 
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    if output_mode == "classification":
        label_map = {label: i for i, label in enumerate(label_list)}
    elif output_mode == "regression" and label_list == [None]: # STS-B
        pass # No label_map needed for regression if labels are already floats
    else: # Regression with string labels that need to be converted? Unlikely for GLUE.
        label_map = {label: float(label) for i, label in enumerate(label_list)}


    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        # Prepare segment_ids for the unpadded sequence part by part
        current_segment_ids = [0] * len(tokens) # For CLS and tokens_a

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            current_segment_ids += [1] * (len(tokens_b) + 1) # For tokens_b and its SEP
        
        # Now `tokens` (list of string tokens) and `current_segment_ids` (list of 0s and 1s)
        # represent the complete unpadded sequence.

        input_ids_converted = tokenizer.convert_tokens_to_ids(tokens)
        
        # seq_length is the original length of the sequence (number of actual tokens including CLS, SEP(s))
        seq_length = len(input_ids_converted)

        # Create unpadded input_mask (list of 1s for actual tokens)
        input_mask_unpadded = [1] * seq_length
        
        # Calculate how many padding tokens are needed
        num_padding_tokens = max_seq_length - seq_length

        # Pad input_ids with tokenizer.pad_token_id
        input_ids = input_ids_converted + [tokenizer.pad_token_id] * num_padding_tokens

        # Pad input_mask with 0
        input_mask = input_mask_unpadded + [0] * num_padding_tokens

        # Pad segment_ids (which is current_segment_ids) with 0
        segment_ids = current_segment_ids + [0] * num_padding_tokens
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label) # STS-B labels are already floats
        else:
            raise KeyError(output_mode)

        if ex_index < 1: # Print first example
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: {}".format(example.label))
            print("label_id: {}".format(label_id))
            print("seq_length: {}".format(seq_length))


        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features

def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    else:
        raise ValueError(f"Unsupported output_mode: {output_mode}")

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long) # Added for completeness
    
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


processors = {
    "sst-2": Sst2Processor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "cb": CBProcessor,
    "sick": SickProcessor,
    # Add other processors if needed, e.g. CoLA
    "cola": Sst2Processor, # Placeholder, CoLA is single sentence, similar to SST-2 but diff metrics
}

output_modes = {
    "sst-2": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "cb": "classification",
    "sick": "classification",
    "cola": "classification",
}
# <----------------- END OF INSERTED CODE ----------------->

# Removed InputExample, InputFeatures, DataProcessor and its subclasses (MrpcProcessor, MnliProcessor, Sst2Processor, RteProcessor, QnliProcessor, QqpProcessor)

# task_to_keys mapping from Hugging Face run_glue.py
# task_to_keys = {
# "cola": ("sentence", None),
# "mnli": ("premise", "hypothesis"),
# "mrpc": ("sentence1", "sentence2"),
# "qnli": ("question", "sentence"),
# "qqp": ("question1", "question2"),
# "rte": ("sentence1", "sentence2"),
# "sst-2": ("sentence", None),
# "stsb": ("sentence1", "sentence2"),
# "wnli": ("sentence1", "sentence2"),
# } #This is now handled by processors


# Removed convert_examples_to_features and _truncate_seq_pair

# Removed processors and output_modes dictionaries


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return [to_device(b, device) for b in batch]
    elif isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--fine_tune_lr", default=6e-4, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--depths", default=6, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--dim", default=768, type=int)
    parser.add_argument("--tau", default=10.0, type=float)
    parser.add_argument("--common_thr", default=1.0, type=float)
    parser.add_argument("--num_step", default=16, type=int)
    parser.add_argument("--tokenizer_path", default="bert-base-cased", type=str)
    parser.add_argument("--output_path", default="saved_models/glue_spikformer", type=str)
    parser.add_argument("--task_name", default="SST-2", type=str, help="The name of the GLUE task to train.")
    parser.add_argument("--data_dir", default="./data/glue_data/", type=str, help="The input data dir. Should contain the .tsv files (or other data files) for the task. Also used as cache_dir for datasets library.")
    # Added for consistency with HF scripts, though not strictly used if pad_to_max_length is True
    # parser.add_argument("--pad_to_max_length", type=bool, default=True, help="Whether to pad all samples to `max_seq_length` in preprocessing.") # This is handled by convert_examples_to_features
    parsed_args = parser.parse_args()
    return parsed_args


def train(args_param):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args_param.task_name = args_param.task_name.lower()
    # if args_param.task_name not in task_to_keys: # Old check
    # raise ValueError(f"Task not found: {args_param.task_name}. Supported tasks: {list(task_to_keys.keys())}")
    if args_param.task_name not in processors:
        raise ValueError(f"Task not found: {args_param.task_name}. Supported tasks: {list(processors.keys())}")

    processor = processors[args_param.task_name]()
    output_mode = output_modes[args_param.task_name]
    label_list = processor.get_labels()
    num_labels = 1 if output_mode == "regression" else len(label_list)


    # Load tokenizer
    do_lower_case = "uncased" in args_param.tokenizer_path.lower()
    tokenizer = BertTokenizer.from_pretrained(args_param.tokenizer_path, do_lower_case=do_lower_case)
    if tokenizer.pad_token_id is None:
        # Common practice for BERT tokenizers if pad_token is not set
        tokenizer.pad_token_id = 0 # Or tokenizer.eos_token_id if applicable and model expects it for padding
        print(f"Warning: tokenizer.pad_token_id was None, set to {tokenizer.pad_token_id}")

    # Determine the specific data directory for the task
    # Processors expect data_dir to be the folder containing train.tsv, dev.tsv etc.
    # e.g. ./data/glue_data/SST-2/ or ./data/glue_data/CB/
    task_specific_data_dir = os.path.join(args_param.data_dir, args_param.task_name.upper())
    # For tasks like 'cb' or 'sick' if they are not in uppercase subdirs or have different naming.
    # This might need adjustment if user's data_dir structure is different.
    # For now, assuming TASK_NAME.upper() is the subfolder.
    if not os.path.exists(task_specific_data_dir):
         # Fallback if task_specific_data_dir (e.g. SST-2) does not exist, maybe data_dir itself is the task folder
        if os.path.exists(os.path.join(args_param.data_dir, "train.tsv")) or \
           os.path.exists(os.path.join(args_param.data_dir, "train.arrow")):
            task_specific_data_dir = args_param.data_dir
        else:
            # Attempt to find files for tasks like 'mnli-mm' where task_name has a hyphen
            potential_task_dir_name = args_param.task_name.split('-')[0].upper()
            if os.path.exists(os.path.join(args_param.data_dir, potential_task_dir_name)):
                 task_specific_data_dir = os.path.join(args_param.data_dir, potential_task_dir_name)
            else:
                raise ValueError(
                    f"Data directory for task {args_param.task_name} not found at {task_specific_data_dir} or {args_param.data_dir}. "
                    f"Please ensure data files (e.g., train.tsv) are present."
                )
    print(f"Using data from: {task_specific_data_dir}")


    train_examples = processor.get_train_examples(task_specific_data_dir)
    
    # For MNLI, there's also mnli-mm. If task is 'mnli', dev is matched.
    # If task is 'mnli-mm', processor (MnliMismatchedProcessor) handles dev_mismatched.
    eval_examples = processor.get_dev_examples(task_specific_data_dir)

    print(f"Loaded {len(train_examples)} train examples and {len(eval_examples)} eval examples.")

    train_features = convert_examples_to_features(
        train_examples, label_list, args_param.max_length, tokenizer, output_mode)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args_param.max_length, tokenizer, output_mode)

    train_data, _ = get_tensor_data(output_mode, train_features)
    eval_data, _ = get_tensor_data(output_mode, eval_features) # eval_labels from get_tensor_data not directly used by current eval loop

    # Removed old data loading and preprocessing (lines 92-179 of original file)
    # try:
    # raw_datasets = load_dataset("glue", args_param.task_name, cache_dir=args_param.data_dir)
    # ...
    # processed_datasets = raw_datasets.map(...)

    model = new_spikformer_legacy(depths=args_param.depths, length=args_param.max_length, T=args_param.num_step,
                           tau=args_param.tau, common_thr=args_param.common_thr, 
                           vocab_size=tokenizer.vocab_size,
                           dim=args_param.dim, num_classes=num_labels, 
                           mode="train")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded.")
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(params=model.parameters(), 
                                   lr=args_param.fine_tune_lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args_param.epochs, eta_min=0)

    # Create DataLoaders
    # train_dataset = processed_datasets["train"] # Old way
    # eval_split_name = "validation_matched" if args_param.task_name == "mnli" else "validation"
    # if eval_split_name not in processed_datasets:
    # print(f"Warning: '{eval_split_name}' not found in dataset splits. Available: {list(processed_datasets.keys())}. Using 'validation' if present.")
    # eval_split_name = "validation" # Fallback
    # if eval_split_name not in processed_datasets:
    # raise ValueError(f"Evaluation split '{eval_split_name}' not found.")
    # eval_dataset = processed_datasets[eval_split_name] # Old way
    
    # Set format for PyTorch - Handled by TensorDataset and get_tensor_data
    # columns_to_set_format = ["input_ids", "attention_mask", "label"]
    # if "token_type_ids" in processed_datasets["train"].column_names:
    # pass
    # train_dataset.set_format(type="torch", columns=columns_to_set_format) # Old way
    # eval_dataset.set_format(type="torch", columns=columns_to_set_format) # Old way
    
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=args_param.batch_size)
    
    eval_sampler = SequentialSampler(eval_data)
    valid_data_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args_param.batch_size)


    device_ids = [i for i in range(torch.cuda.device_count())]
    print(f"Using devices: {device_ids}")
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        model = model.to(device)

    output_dir_path = os.path.join(args_param.output_path, args_param.task_name.upper())
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    
    acc_list = []
    best_acc = 0.0

    for epoch in tqdm(range(args_param.epochs), desc="Epochs"):
        avg_loss = []
        model.train()
        for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch+1}/{args_param.epochs}", leave=False):
            # Batch from TensorDataset is a tuple: (input_ids, input_mask, segment_ids, label_ids, seq_lengths)
            # input_ids = batch['input_ids'].to(device)
            # input_mask_batch = batch['attention_mask'].to(device)
            # labels = batch['label'].to(device)
            
            input_ids = batch[0].to(device)
            # input_mask_batch = batch[1].to(device) # Not used by current new_spikformer_legacy model call
            # segment_ids_batch = batch[2].to(device) # Not used
            labels = batch[3].to(device)
            
            #_, outputs = model(input_ids, input_mask=input_mask_batch)
            _, outputs = model(input_ids)
            #logits = outputs
            logits = torch.mean(outputs, dim=1)
            
            loss = F.cross_entropy(logits, labels)
            avg_loss.append(loss.item())

            # Update train_loss in PredictiveLIFNode instances
            if hasattr(model, 'update_predictor_train_loss'): # Check if the model has the method (e.g. not DataParallel wrapper)
                model.update_predictor_train_loss(loss.item())
            elif isinstance(model, nn.DataParallel) and hasattr(model.module, 'update_predictor_train_loss'): # Check module inside DataParallel
                model.module.update_predictor_train_loss(loss.item())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            functional.reset_net(model)

        scheduler.step()
        print(f"Avg training loss at epoch {epoch+1}: {np.mean(avg_loss):.4f}")

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            correct = 0
            total_eval_samples = 0
            for batch in tqdm(valid_data_loader, desc=f"Evaluating Epoch {epoch+1}/{args_param.epochs}", leave=False):
                # input_ids = batch['input_ids'].to(device)
                # input_mask_batch = batch['attention_mask'].to(device)
                # labels = batch['label'].to(device)

                input_ids = batch[0].to(device)
                # input_mask_batch = batch[1].to(device) # Not used
                labels = batch[3].to(device)

                #_, outputs = model(input_ids, input_mask=input_mask_batch)
                _, outputs = model(input_ids)
                #logits = outputs
                logits = torch.mean(outputs, dim=1)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total_eval_samples += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                functional.reset_net(model)

        acc = float(correct) / total_eval_samples if total_eval_samples > 0 else 0
        acc_list.append(acc)
        print(f"Epoch {epoch+1} Validation Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            print(f"New best validation accuracy: {best_acc:.4f}")
            save_file_path = os.path.join(output_dir_path, 
                                     f"{args_param.task_name}_epoch{epoch+1}_acc{acc:.4f}" +
                                     f"_lr{args_param.fine_tune_lr}_seed{args_param.seed}" +
                                     f"_bs{args_param.batch_size}_depths{args_param.depths}_len{args_param.max_length}" +
                                     f"_tau{args_param.tau}_thr{args_param.common_thr}_T{args_param.num_step}.pt")
            torch.save(model.state_dict(), save_file_path)
            print(f"Model saved to {save_file_path}")

    print(f"Training finished. Best validation accuracy: {best_acc:.4f}")
    if acc_list:
        print(f"All validation accuracies: {acc_list}")


if __name__ == "__main__":
    _args = args_parser()
    print("Parsed arguments:")
    for arg_name, arg_val in vars(_args).items():
        print(f"  {arg_name}: {arg_val}")
    set_seed(_args.seed)
    train(_args)