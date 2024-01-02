import spacy
from datasets import load_dataset, DatasetDict, load_from_disk, Audio
from transformers import AutoProcessor, BertTokenizer
from Ontonotes import Ontonotes5Features, get_non_empty_ner, get_iob
import numpy as np
import pandas as pd


def DatasetHelper(from_disc, percentage=5):
    if from_disc:
        return load_from_disk('./dataset/datasetVoxpopuliAfterMap')
    else:
        dataset = DatasetDict()
        dataset['train'] = load_dataset("asapp/slue", "voxpopuli", split=f"train[:{percentage}%]")
        dataset['test'] = load_dataset("asapp/slue", "voxpopuli", split=f"test[:{percentage}%]")
        dataset['validation'] = load_dataset("asapp/slue", "voxpopuli", split=f"validation[:{percentage}%]")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        return dataset


def check_input_audio(max_audio_length, processed_audio):
    input_audio = processed_audio.input_values[0]
    audio_length = len(input_audio)
    if audio_length > max_audio_length:
        input_audio = input_audio[:max_audio_length]  # Przycinanie do max_audio_length
    elif audio_length < max_audio_length:
        padding_length = max_audio_length - audio_length
        input_audio = np.pad(input_audio, (0, padding_length), 'constant')
    return input_audio


def get_ents(example):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(example['normalized_text'])
    ents = []
    ner_data = example['normalized_ner']
    for i in range(len(example['normalized_ner']['type'])):
        label = ner_data['type'][i]
        start = ner_data['start'][i]
        end = start + ner_data['length'][i]
        ent = doc.char_span(start, end, label=label, alignment_mode='contract')
        if ent is not None:
            ents.append(ent)
    doc.set_ents(ents)
    return doc


def preprocess_audio_text_voxpopuli_non_normalized(example):
    max_audio_length=16000
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    doc = get_ents(example)

    tokenized_text = text_tokenizer(
        example["normalized_text"],
        padding="max_length", truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    processed_audio = processor(
        example["audio"]["array"],
        sampling_rate=example["audio"]["sampling_rate"],
    )

    input_ids = tokenized_text["input_ids"][0].tolist()
    input_audio = check_input_audio(max_audio_length, processed_audio)

    df = pd.DataFrame([[t.text, get_iob(t), t.whitespace_ == ' '] for t in doc])
    df[2] = df[2].cumsum().shift(1).fillna(0)
    words = [''.join(grp[0]) for _, grp in df.groupby(2)]
    labels = [get_non_empty_ner(grp[1]) for _, grp in df.groupby(2)]
    label_ids = [Ontonotes5Features.label_to_id.get(label, Ontonotes5Features.label_to_id['O']) for label in labels]

    if isinstance(label_ids[0], list):
        label_ids = [id for sublist in label_ids for id in sublist]

    padded_labels = label_ids + [Ontonotes5Features.label_to_id['O']] * (512 - len(label_ids))
    # new_example = dict()
    # new_example['tokens'] = words
    # new_example['labels'] = padded_labels
    # new_example['labels_orig'] = [x.split('-')[-1] for x in labels]
    # new_example['tags'] = [Ontonotes5Features.label_to_id.get(label, 'O') for label in labels]
    # new_example["input_ids"] = input_ids
    # new_example["input_audio"] = input_audio
    a=[x.split('-')[-1] for x in labels]
    f=[Ontonotes5Features.label_to_id.get(label, 'O') for label in labels]
    return {
        'tokens': words,
        'labels': padded_labels,
        'labels_orig': [x.split('-')[-1] for x in labels],
        'tags': [Ontonotes5Features.label_to_id.get(label, 'O') for label in labels],
        "input_ids": input_ids,
        "input_audio": input_audio
    }


def preprocess_text_voxpopuli(example):
    max_length = 1024
    doc = get_ents(example)
    text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_text = text_tokenizer(
        example["normalized_text"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = tokenized_text["input_ids"][0].tolist()

    df = pd.DataFrame([[t.text, get_iob(t), t.whitespace_ == ' '] for t in doc])
    df[2] = df[2].cumsum().shift(1).fillna(0)
    words = [''.join(grp[0]) for _, grp in df.groupby(2)]
    labels = [get_non_empty_ner(grp[1]) for _, grp in df.groupby(2)]
    label_ids = [Ontonotes5Features.label_to_id.get(label, Ontonotes5Features.label_to_id['O'])
                 for label in labels]


    if isinstance(label_ids[0], list):
        label_ids = [id for sublist in label_ids for id in sublist]

    padded_labels = label_ids + [Ontonotes5Features.label_to_id['O']] * (max_length - len(label_ids))
    # new_example = dict()
    # new_example['tokens'] = words
    # new_example['labels'] = padded_labels
    # new_example['labels_orig'] = [x.split('-')[-1] for x in labels]
    # new_example['tags'] = [Ontonotes5Features.label_to_id.get(label, 'O') for label in labels]
    # new_example["input_ids"] = input_ids

    return {
        'tokens': words,
        'labels': padded_labels,
        'labels_orig': [x.split('-')[-1] for x in labels],
        'tags': [Ontonotes5Features.label_to_id.get(label, 'O') for label in labels],
        "input_ids": input_ids,
    }