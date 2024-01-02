from tqdm import tqdm
from transformers import BertPreTrainedModel, BertModel, AutoConfig, PreTrainedModel, AutoModel, \
    AutoModelForTokenClassification
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from Ontonotes import Ontonotes5Features



# class CustomTextModel1(BertPreTrainedModel):
#     def __init__(
#             self,
#             bert_model_name,
#             text_config,
#             num_labels
#     ):
#         super(CustomTextModel, self).__init__(config=text_config)
#         self.model_name = 'TextModel'
#         self.bert = BertModel.from_pretrained("bert-base-uncased")
#         # self.bert = BertModel.from_pretrained(bert_model_name)
#         self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
#         self.num_labels = num_labels
#         self.text_dim = text_config.hidden_size
#
#     def forward(
#             self,
#             input_ids,
#             attention_mask=None,
#             labels=None
#     ):
#         text_output = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         ).pooler_output
#
#         logits = self.classifier(text_output)
#         loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             logits = logits.view(-1, self.num_labels)
#             if len(labels) % logits.shape[0] == 0:
#                 labels = labels.view(logits.shape[0], -1)
#                 labels = labels[:, 0]
#                 loss = loss_fct(logits, labels)
#             else:
#                 raise ValueError("Liczba etykiet nie jest podzielna przez rozmiar wsadu")
#
#         return SequenceClassifierOutput(loss=loss, logits=logits)


class CustomTextModel(BertPreTrainedModel):
    config_class = AutoConfig
    def __init__(
            self,
            config,
            num_labels
    ):
        super(CustomTextModel, self).__init__(config=config)
        self.model_name = 'TextModel'
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        # self.bert = AutoModel.from_pretrained(model_name)
        self.bert = AutoModelForTokenClassification.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels
        self.text_dim = config.hidden_size

    def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None
    ):
        text_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )#.pooler_output

        # logits = self.classifier(text_output)
        logits = text_output.logits
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, self.num_labels)
            if len(labels) % logits.shape[0] == 0:
                labels = labels.view(logits.shape[0], -1)
                labels = labels[:, 0]
                loss = loss_fct(logits, labels)
            else:
                raise ValueError("Liczba etykiet nie jest podzielna przez rozmiar wsadu")

        return SequenceClassifierOutput(loss=loss, logits=logits)

def evaluate_text_model(dataset, tokenizer, model):
    true_y = []
    pred_y = []
    for item in tqdm(dataset):
        tokenized_item = tokenize_adjust_labels(item, tokenizer)
        gold = tokenized_item['labels']
        del tokenized_item['labels']

        with torch.no_grad():
            # Używamy modelu i przenosimy logity na odpowiednie urządzenie
            output = model(**tokenized_item)
            logits = output.logits.to('mps')



            # Zmiana wymiaru dla argmax
            predictions = torch.argmax(logits, dim=0)


        # Dostosowanie formatu prawdziwych etykiet i predykcji

        true_y.append(gold[1:-1])

        pred_y.append(predictions.tolist()[1:-1])
    print('po forze')
    return [true_y, pred_y]

def tokenize_adjust_labels(example, tokenizer):
    tokenized_samples = tokenizer.encode_plus(example['tokens'], is_split_into_words=True, return_tensors="pt").to(
        "mps")
    # tokenized_samples is not a datasets object so this alone won't work with Trainer API, hence map is used
    # so the new keys [input_ids, labels (after adjustment)]
    # can be added to the datasets dict for each train test validation split
    tokenized_samples.pop("token_type_ids", None)
    prev_wid = -1
    word_ids_list = tokenized_samples.word_ids()
    existing_label_ids = example["tags"]
    i = -1
    # print(tokenized_samples, example, existing_label_ids, word_ids_list)
    # print(len(example['tokens']))
    # print(max(x for x in word_ids_list if x is not None))
    adjusted_label_ids = []

    for wid in word_ids_list:
        if (wid is None):
            adjusted_label_ids.append(-100)
        elif (wid != prev_wid):
            i = i + 1
            adjusted_label_ids.append(existing_label_ids[i])
            prev_wid = wid
        else:
            adjusted_label_ids.append(existing_label_ids[i])

    tokenized_samples["labels"] = adjusted_label_ids
    return tokenized_samples
