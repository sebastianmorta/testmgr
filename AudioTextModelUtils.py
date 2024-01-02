from tqdm import tqdm
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, BertModel, Wav2Vec2ForCTC, AutoConfig, AutoModel, \
    AutoModelForTokenClassification
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel
import torch
from torch.nn import Linear


class CustomAudioTextModel(Wav2Vec2PreTrainedModel):
    config_class = AutoConfig

    def __init__(
            self,
            audio_config,
            text_config,
            num_labels
    ):
        super(CustomAudioTextModel, self).__init__(text_config, audio_config)
        self.model_name = 'AudioTextModel'
        self.wav2vec2 = AutoModel.from_config(audio_config)
        self.bert = AutoModelForTokenClassification.from_config(text_config)

        self.num_labels = num_labels

        self.audio_dim = audio_config.hidden_size
        self.text_dim = text_config.hidden_size
        self.combined_dim = self.audio_dim + self.text_dim

        self.classifier = nn.Linear(self.combined_dim, num_labels)

    def forward(
            self,
            input_audio,
            attention_mask,
            input_ids,
            labels=None
    ):
        audio_output = self.wav2vec2(input_values=input_audio).last_hidden_state.mean(dim=1)
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        combined_output = torch.cat((audio_output, text_output), dim=1)
        logits = self.classifier(combined_output)
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


def evaluate_audio_text_model(dataset, tokenizer, model):
    true_y = []
    pred_y = []
    model.to("cpu")

    for item in tqdm(dataset):
        # Przetwarzanie danych tekstowych
        lab=item['labels']
        l=item['tags']
        tokenized_text = tokenize_adjust_labels(item, tokenizer)
        gold = tokenized_text['labels']
        del tokenized_text['labels']

        # Przygotowanie danych audio
        input_audio = item['input_audio']  # Zakładam, że dane audio są już w odpowiednim formacie
        # tokenized_text['input_audio']=item['input_audio']
        if not isinstance(input_audio, torch.Tensor):
            # Konwersja na tensor PyTorch, jeśli to konieczne
            input_audio = torch.tensor(input_audio, dtype=torch.float32)

        if input_audio.ndim == 1:
            input_audio = input_audio.unsqueeze(0)
        with torch.no_grad():

            # print(tokenized_text['attention_mask'].shape)
            # print(tokenized_text['input_ids'][0].shape)
            # Przekazywanie danych audio i tekstowych do modelu
            # print(**tokenized_text)
            # output = model(input_audio=tokenized_text['input_audio'],
            # attention_mask=tokenized_text['attention_mask'][0],
            # input_ids=tokenized_text['input_ids'][0]
            # )
            output = model(input_audio=input_audio, **tokenized_text)
            logits = output.logits
            predictions = torch.argmax(logits, dim=1)

        true_y.append(gold[1:-1])
        pred_y.append(predictions.tolist()[1:-1])

    return [true_y, pred_y]


def tokenize_adjust_labels(example, tokenizer):
    tokenized_samples = tokenizer.encode_plus(
        example['tokens'],
        is_split_into_words=True,
        return_tensors="pt"
    )

    tokenized_samples.pop("token_type_ids", None)
    prev_wid = -1
    word_ids_list = tokenized_samples.word_ids()
    existing_label_ids = example["tags"]
    i = -1

    # print(word_ids_list)

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
    # print(tokenized_samples)
    return tokenized_samples

# class UnifiedAudioTextNERModel(torch.nn.Module):
#     def __init__(self, num_labels):
#         super(UnifiedAudioTextNERModel, self).__init__()
#         # Ścieżka audio
#         self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
#         self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
#
#         # Ścieżka tekstu
#         self.text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#         self.text_model = BertModel.from_pretrained("bert-base-uncased")
#
#         # Połączenie ścieżek i warstwa klasyfikująca
#         self.classifier = Linear(self.audio_model.config.hidden_size + self.text_model.config.hidden_size, num_labels)
#
#     def forward(self, input_audio, input_text):
#         # Przetwarzanie audio
#         input_values = self.audio_processor(input_audio, return_tensors="pt").input_values
#         audio_output = self.audio_model(input_values).last_hidden_state
#
#         # Przetwarzanie tekstu
#         input_ids = self.text_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids
#         text_output = self.text_model(input_ids).last_hidden_state
#
#         # Łączenie ścieżek
#         combined_output = torch.cat((audio_output, text_output), dim=-1)
#
#         # Klasyfikacja NER
#         predictions = self.classifier(combined_output)
#
#         return predictions
#
# class CustomAudioTextModel(Wav2Vec2PreTrainedModel):
#     config_class = AutoConfig
#     def __init__(
#             self,
#             audio_config,
#             text_config,
#             num_labels
#     ):
#         super(CustomAudioTextModel, self).__init__(text_config, audio_config)
#         self.model_name = 'AudioTextModel'
#         self.wav2vec2 = AutoModel.from_config(audio_config)
#         self.bert = AutoModel.from_config(text_config)
#
#         # self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
#         #
#         # self.audio_encoder = Wav2Vec2Model(audio_config)
#         # self.text_encoder = BertModel(text_config)
#         self.num_labels = num_labels
#         # print(audio_config.hidden_size, text_config.hidden_size)
#         self.audio_dim = audio_config.hidden_size
#         self.text_dim = text_config.hidden_size
#         self.combined_dim = self.audio_dim + self.text_dim
#         # self.combined_layer = nn.Linear(self.combined_dim, self.combined_dim)
#         self.classifier = nn.Linear(self.combined_dim, num_labels)
#         #
#
#     def forward(
#             self,
#             input_audio,
#             attention_mask,
#             input_ids,
#             labels=None
#     ):
#         audio_output = self.wav2vec2(input_values=input_audio).last_hidden_state.mean(dim=1)
#         text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
#         # print("-----------------uwaga----------------")
#         # print(audio_output)
#         # print(text_output)
#         combined_output = torch.cat((audio_output, text_output), dim=1)
#         logits = self.classifier(combined_output)
#         loss = None
#
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             logits = logits.view(-1, self.num_labels)
#
#             if len(labels) % logits.shape[0] == 0:
#                 labels = labels.view(logits.shape[0], -1)
#                 labels = labels[:, 0]
#                 loss = loss_fct(logits, labels)
#             else:
#                 raise ValueError("Liczba etykiet nie jest podzielna przez rozmiar wsadu")
#
#         return SequenceClassifierOutput(loss=loss, logits=logits)
#
# class CustomAudioTextModel1(Wav2Vec2PreTrainedModel):
#     # config_class = AutoConfig
#     def __init__(
#             self,
#             wav2vec2_model_name,
#             audio_config,
#             text_config,
#             bert_model_name,
#             num_labels
#     ):
#         super(CustomAudioTextModel, self).__init__(text_config)
#         self.model_name = 'AudioTextModel'
#         self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model_name)
#         self.bert = BertModel.from_pretrained(bert_model_name)
#
#         self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
#
#         self.audio_encoder = Wav2Vec2Model(audio_config)
#         self.text_encoder = BertModel(text_config)
#         self.num_labels = num_labels
#         self.audio_dim = audio_config.hidden_size
#         self.text_dim = text_config.hidden_size
#         self.combined_dim = self.audio_dim + self.text_dim
#         # self.combined_layer = nn.Linear(self.combined_dim, self.combined_dim)
#         self.classifier = nn.Linear(self.combined_dim, num_labels)
#         print(audio_config.hidden_size, text_config.hidden_size)
#
#     def forward(
#             self,
#             input_audio,
#             attention_mask,
#             input_ids,
#             labels=None
#     ):
#         audio_output = self.audio_encoder(input_values=input_audio).last_hidden_state.mean(dim=1)
#         text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
#         combined_output = torch.cat((audio_output, text_output), dim=1)
#         logits = self.classifier(combined_output)
#         loss = None
#
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             logits = logits.view(-1, self.num_labels)
#
#             if len(labels) % logits.shape[0] == 0:
#                 labels = labels.view(logits.shape[0], -1)
#                 labels = labels[:, 0]
#                 loss = loss_fct(logits, labels)
#             else:
#                 raise ValueError("Liczba etykiet nie jest podzielna przez rozmiar wsadu")
#
#         return SequenceClassifierOutput(loss=loss, logits=logits)
