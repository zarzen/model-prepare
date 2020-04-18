import torch
import os
from save_base_models import torch_vision_models, nlp_models
from common import get_layers, get_layers_size
from tqdm import tqdm

nrepeat = 100


def perturb_model(module, seed):
    torch.manual_seed(seed)
    childs = list(module.children())
    if len(childs) == 0:
        # change current module parameters
        for p in module.parameters():
            p_size = p.size()
            noise = (torch.rand(p_size) - 0.5) / 1e4
            new_param = p.data + noise
            p.data = new_param
        return
    else:
        for c in childs:
            perturb_model(c, seed)


def perturb_vision_models():
    """"""
    save_to = "../perturbed-models/vision"
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    rand_data = torch.rand((8, 3, 299, 299))

    for name in tqdm(torch_vision_models):
        print("===== working on model name", name)
        for seed in range(nrepeat):
            print("working on seed", seed)
            torch.manual_seed(seed)
            model = torch.hub.load(
                "pytorch/vision:v0.4.2", name, pretrained=True)
            perturb_model(model, seed)

            outputs = model(rand_data)
            if name == "inception_v3":
                print("model {}, seed {}".format(name, seed),
                      outputs[0].sum().item())
            else:
                print("model {}, seed {}".format(name, seed),
                      outputs.sum().item())

            model_dir = os.path.join(save_to, name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_filename = "{}-{}.pt".format(name, seed)
            torch.save(model.state_dict(), os.path.join(
                model_dir, model_filename))
            del model


def perturb_nlp_models():
    """"""
    save_to = "../perturbed-models/nlp"
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    
    test_inputs = ["This document provides solutions to a variety of use cases regarding the saving and loading of PyTorch models.",
                   " Feel free to read the whole document, or just skip to the code you need for a desired use case.",
                   "When it comes to saving and loading models, there are three core functions to be familiar with:",
                   " Saves a serialized object to disk. ",
                   "This function uses Python’s pickle utility for serialization.",
                   "Models, tensors, and dictionaries of all kinds of objects can be saved using this function.",
                   "Uses pickle’s unpickling facilities to deserialize pickled object files to memory.",
                   "This function also facilitates the device to load the data into."]

    for model_class, tokenizer_class, pretrained_weights, conf in nlp_models:
        tokenizer = tokenizer_class.from_pretrained(
            pretrained_weights, cache_dir="./huggingface-cache")
        tokenizer.pad_token = "<END>"
        input_ids = []
        for sent in test_inputs:
            _ids = tokenizer.encode(
                sent, max_length=50, pad_to_max_length=True, add_space_before_punct_symbol=True)
            input_ids.append(_ids)
        input_ids = torch.tensor(input_ids)

        print("===== working on model name", model_class.__name__)
        print("----- input shape", input_ids.size())
        for seed in range(nrepeat):
            print("working on seed", seed)
            model = model_class.from_pretrained(
                pretrained_weights, cache_dir="./huggingface-cache")

            perturb_model(model, seed)
            outputs = model(input_ids)

            print("model {}, seed {}".format(model_class.__name__, seed),
                  outputs[0].sum().item())

            model_dir = os.path.join(save_to, model_class.__name__)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_filename = "{}-{}.pt".format(model_class.__name__, seed)
            torch.save(model.state_dict(), os.path.join(
                model_dir, model_filename))
            
            del model

        del input_ids


def main():
    """"""
    perturb_vision_models()

    perturb_nlp_models()


if __name__ == "__main__":
    main()
