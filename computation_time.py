import torch
import time
from transformers import *
from save_base_models import nlp_models, torch_vision_models
from common import get_layers, get_layers_size
bw = 80e9  # 80Gbps


def vision_model_time():
    """"""

    for name, _ in torch_vision_models:
        _t = time.time()
        model = torch.hub.load("pytorch/vision:v0.4.2", name, pretrained=True)
        print("load ", name, 'cost', time.time() - _t, 's')
        _t = time.time()
        model = model.to("cuda")
        print('to cuda cost', time.time() - _t)
        model.eval()
        layers = []
        get_layers(model, layers)
        sizes = get_layers_size(layers)

        # random data
        for i in range(10):
            # pylint: disable=no-member
            data = torch.rand((1, 3, 224, 224)).to("cuda")
            torch.cuda.synchronize()
            t1 = time.time()
            outputs = model(data)
            torch.cuda.synchronize()
            t2 = time.time()

            del data
            del outputs
            torch.cuda.empty_cache()

        trans_time = sum(sizes) * 8.0 / bw
        print(name, ":: foward cost ", t2 - t1,
              "est. transition time", trans_time)

        del model
        torch.cuda.empty_cache()


def _nlp_exp(model_class, tokenizer_class, pretrained_weights, config):
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(
        pretrained_weights, cache_dir="./huggingface-cache")
    model = model_class.from_pretrained(
        pretrained_weights, cache_dir="./huggingface-cache")
    layers = []
    get_layers(model, layers)
    sizes = get_layers_size(layers)

    model = model.to('cuda')
    model.eval()
    for i in range(10):
        # pylint: disable=not-callable
        input_ids = torch.tensor([tokenizer.encode(
            "Let's see all hidden-states and attentions on this text", add_space_before_punct_symbol=True)] * 8).to('cuda')
        torch.cuda.synchronize()
        t1 = time.time()
        outputs = model(input_ids)
        torch.cuda.synchronize()
        t2 = time.time()
        del outputs
        del input_ids
        torch.cuda.empty_cache()

    trans_time = sum(sizes) * 8.0 / bw
    print(model_class.__name__, ":: foward cost ", t2 - t1,
          "est. transition time", trans_time)

    del model
    del tokenizer
    del layers
    del sizes
    torch.cuda.empty_cache()


def nlp_model_inf_time():
    for model_class, tokenizer_class, pretrained_weights, conf in nlp_models:
        _nlp_exp(model_class, tokenizer_class, pretrained_weights, conf)
        torch.cuda.empty_cache()
        time.sleep(5)

def main():
    """"""
    vision_model_time()

    nlp_model_inf_time()


if __name__ == "__main__":
    main()
