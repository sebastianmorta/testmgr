import torch
from datasets import load_from_disk, load_metric
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, AutoModelForAudioClassification, \
    AutoModelForTokenClassification, BertConfig, AutoConfig, BertModel

import AudioTextModelUtils
import TextModelUtils
import modelsClass
from AudioTextModelUtils import CustomAudioTextModel
from TextModelUtils import CustomTextModel
from modelsClass import DatasetHelper, process_voxpopuli_non_normalized, evaluate_model, Ontonotes5Features

num_labels = len(Ontonotes5Features.ontonotes_labels_bio)
dataset = load_from_disk('./dataset/DatasetVoxpopuliAfterMap10p')
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

textConfig = AutoConfig.from_pretrained('./resultPretrained/TextModelsmall1')

modelText = CustomTextModel(textConfig, num_labels=num_labels).to("mps")
modelText.load_state_dict(torch.load('./resulttorch/TextModel.pth'))

audio_text_config_text = BertConfig.from_pretrained("bert-base-uncased")
audio_text_config_audio = AutoConfig.from_pretrained("facebook/wav2vec2-base-960h")

modelAudioText = CustomAudioTextModel(audio_config=audio_text_config_audio, text_config=audio_text_config_text,
                                      num_labels=num_labels)
modelAudioText.load_state_dict(torch.load('./resulttorch/AudioTextModel.pth'))

models = [modelAudioText, modelText]
evaluators = [AudioTextModelUtils.evaluate_audio_text_model, TextModelUtils.evaluate_text_model]

if __name__ == "__main__":
    for model, evaluator in zip(models, evaluators):

        model.eval()
        result = evaluator(dataset['validation'], text_tokenizer, model)
        print("RESULTSS", result)
        a=[]
        for i in result[0]:
            a+=i
        print(sum(a))
        b=[]
        for i in result[1]:
            b+=i
        print(sum(b))