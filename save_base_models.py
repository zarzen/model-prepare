import torch
from common import get_layers, get_layers_size, torch_vision_models, nlp_models
from transformers import *
import os
import json
import time


def process_vision_models():
    data_dir = "../base-models/vision"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for name, _ in torch_vision_models:
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



def process_nlp_models():
    data_dir = "../base-models/nlp"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for model_class, tokenizer_class, pretrained_weights, _ in nlp_models:
        t1 = time.time()
        tokenizer = tokenizer_class.from_pretrained(
            pretrained_weights, cache_dir="./huggingface-cache")
        t2 = time.time()
        model = model_class.from_pretrained(
            pretrained_weights, cache_dir="./huggingface-cache")
        t3 = time.time()
        print("loading tokenizer cost", t2 - t1,
              "loading model cost: ", t3 - t2)
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
