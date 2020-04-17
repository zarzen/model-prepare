import torch
from common import get_layers, get_layers_size
from transformers import *
import os
import json
import time

torch_vision_models = [
    'resnet50',
    'resnet101',
    'resnet152',
    'wide_resnet101_2',
    'wide_resnet50_2',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'inception_v3',
    'googlenet',
    'alexnet',
    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0',
    'squeezenet1_0',
    'squeezenet1_1',
]


def process_vision_models():
    data_dir = "../base-models/vision"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for name in torch_vision_models:
        model = torch.hub.load("pytorch/vision:v0.4.2", name, pretrained=True)
        layers = []
        get_layers(model, layers)
        sizes = get_layers_size(layers)

        model_dir = os.path.join(data_dir, name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
        meta_obj = {
            "name": name,
            "total_size": sum(sizes),
            "layer_sizes": sizes
        }
        with open(os.path.join(model_dir, "meta.json"), "w") as ofile:
            json.dump(meta_obj, ofile, indent=2)

        print("processed model", name, "total size",
              sum(sizes) / (1024.0*1024), "MB")


# using huggingface api
nlp_models = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
              (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
              (GPT2Model,       GPT2Tokenizer,       'gpt2'),
              (CTRLModel,       CTRLTokenizer,       'ctrl'),
              (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
              (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
              (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
              (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
              (RobertaModel,    RobertaTokenizer,    'roberta-base'),
              (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
              ]

def process_nlp_models():
    data_dir = "../base-models/nlp"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for model_class, tokenizer_class, pretrained_weights in nlp_models:
        t1 = time.time()
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, cache_dir="./huggingface-cache")
        t2 = time.time()
        model = model_class.from_pretrained(pretrained_weights, cache_dir="./huggingface-cache")
        t3 = time.time()
        print("loading tokenizer cost", t2 - t1, "loading model cost: ", t3 - t2)
        # tokenizer_layers = []
        model_layers = []
        # get_layers(tokenizer, tokenizer_layers)
        get_layers(model, model_layers)

        # tokenizer_sizes = get_layers_size(tokenizer_layers)
        model_sizes = get_layers_size(model_layers)

        model_dir = os.path.join(data_dir, pretrained_weights)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # tokenizer.save_pretrained(model_dir)
        torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
        meta_obj = {
            "model": model.__class__.__name__,
            "tokenizer": tokenizer.__class__.__name__,
            "total_size": sum(model_sizes),
            "layer_sizes": model_sizes
        }
        with open(os.path.join(model_dir, "meta.json"), "w") as ofile:
            json.dump(meta_obj, ofile, indent=2)

        print("processed model", pretrained_weights, "total size",
              sum(model_sizes) / (1024.0*1024), "MB")

def main():
    """"""
    process_vision_models()

    process_nlp_models()

if __name__ == "__main__":
    main()
