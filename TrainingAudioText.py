from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoProcessor, BertModel, TrainingArguments, Trainer, \
    DataCollatorWithPadding, BertPreTrainedModel, AutoConfig, AutoTokenizer, AutoFeatureExtractor, \
    AutoProcessor, AutoModel, Wav2Vec2Processor, Wav2Vec2ForCTC, PreTrainedModel, PreTrainedTokenizer, Wav2Vec2Model, \
    AutoTokenizer, TrainingArguments, Trainer, Wav2Vec2Tokenizer, BertTokenizer, BertConfig, \
    DataCollatorForTokenClassification
import os

import AudioTextModelUtils
import TextModelUtils
from preprocessDataset import DatasetHelper, preprocess_text_voxpopuli, preprocess_audio_text_voxpopuli_non_normalized
from Ontonotes import Ontonotes5Features

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import torch.nn as nn
torch.device('cpu')
num_labels = len(Ontonotes5Features.ontonotes_labels_bio)
audio_config = AutoConfig.from_pretrained("facebook/wav2vec2-base-960h")
text_config = BertConfig.from_pretrained("bert-base-uncased")

text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_collator = DataCollatorForTokenClassification(tokenizer=text_tokenizer)

modelText = TextModelUtils.CustomTextModel(text_config, num_labels)
modelAudioText = AudioTextModelUtils.CustomAudioTextModel(audio_config, text_config, num_labels)

# models = [modelAudioText, modelText]
models = [modelText, modelAudioText]
# evaluators = [AudioTextModelUtils.evaluate_audio_text_model, TextModelUtils.evaluate_text_model]
evaluators = [TextModelUtils.evaluate_text_model, AudioTextModelUtils.evaluate_audio_text_model]

# dataset = DatasetHelper(False, 10).map(
#     lambda x: preprocess_audio_text_voxpopuli_non_normalized(x))
dataset = load_from_disk('./dataset/SmallDatasetVoxpopuliAfterMap')
# dataset.save_to_disk(f'./dataset/datasetVoxpopuliAfterMap10p')
if __name__ == "__main__":

    for model, eval in zip(models, evaluators):
        training_args = TrainingArguments(
            output_dir=f"./Resultssmall{model.model_name}",
            evaluation_strategy="steps",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            save_strategy='no',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=text_tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
        torch.save(model.state_dict(),f'./resultTorchsmall/{model.model_name}.pth')
        model.save_pretrained(f'./resultFromPretrainedsmall/{model.model_name}')
