import truecase
import re
import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] + [-1] * (max_len - len(f["labels"])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    output = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
    return output

# Preprocessing
def get_labels(path="labels.txt"):
    with open(path, 'r') as file:
        labels = file.read().splitlines()
    prefix = ["B-", "I-"]
    labels = [pre+lab for lab in labels for pre in prefix]
    labels.append("O")
    return labels


def true_case(tokens):
    word_lst = [(w, idx) for idx, w in enumerate(tokens) if all(c.isalpha() for c in w)]
    lst = [w for w, _ in word_lst if re.match(r'\b[A-Z\.\-]+\b', w)]

    if len(lst) and len(lst) == len(word_lst):
        parts = truecase.get_true_case(' '.join(lst)).split()
        if len(parts) != len(word_lst):
            return tokens
        for (w, idx), nw in zip(word_lst, parts):
            tokens[idx] = nw
    return tokens


def process_instance(words, labels, tokenizer, LABEL_TO_ID, max_seq_length=512):
    tokens, token_labels = [], []
    for word, label in zip(words, labels):
        tokenized = tokenizer.tokenize(word)
        if len(tokenized) > 0: # avoid errors
            if label not in LABEL_TO_ID:
                lab2id = LABEL_TO_ID["O"]
            else:
                lab2id = LABEL_TO_ID[label]
            token_label = [lab2id] + [-1] * (len(tokenized) - 1)
            tokens += tokenized
            token_labels += token_label
    assert len(tokens) == len(token_labels)
    tokens, token_labels = tokens[:max_seq_length - 2], token_labels[:max_seq_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    assert len(input_ids) <= max_seq_length
    token_labels = [-1] + token_labels + [-1]
    return {
        "tokens": tokens,
        "input_ids": input_ids,
        "labels": token_labels
    }


def read_conll(args, file_in, tokenizer, max_seq_length=512):
    label_list = get_labels(path=args.label_file)
    LABEL_TO_ID = {label: i for i, label in enumerate(label_list)}

    words, labels, sentences = [], [], []
    examples = []

    with open(file_in, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("</s>") or line.startswith("<s>") or line=="":
                if words:
                    assert len(words) == len(labels)
                    examples.append(process_instance(words, labels, tokenizer, LABEL_TO_ID, max_seq_length))
                    sentences.append(words[:max_seq_length - 2])
                    words, labels = [], []

            else:
                line = line.split("\t")
                word = line[0]
                label = line[1]
                words.append(word)
                labels.append(label)

        # pick up last sentence
        if words:
            examples.append(process_instance(words, labels, tokenizer, LABEL_TO_ID, max_seq_length))
            sentences.append(words[:max_seq_length - 2])

    return examples, sentences
