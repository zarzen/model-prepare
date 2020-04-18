from transformers import *


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

# using huggingface api
nlp_models = [(BertModel,       BertTokenizer,       'bert-base-uncased', BertConfig),
              (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt', OpenAIGPTConfig),
              (GPT2Model,       GPT2Tokenizer,       'gpt2', GPT2Config),
              (TransfoXLModel,  TransfoXLTokenizer,
               'transfo-xl-wt103', TransfoXLConfig),
              (XLNetModel,      XLNetTokenizer,
               'xlnet-base-cased', XLNetConfig),
              (XLMModel,        XLMTokenizer,
               'xlm-mlm-enfr-1024', XLMConfig),
              (DistilBertModel, DistilBertTokenizer,
               'distilbert-base-cased', DistilBertConfig),
              (RobertaModel,    RobertaTokenizer,
               'roberta-base', RobertaConfig),
              (XLMRobertaModel, XLMRobertaTokenizer,
               'xlm-roberta-base', XLMRobertaConfig),
              ]


def get_layers(module, layers):
    childs = list(module.children())
    if len(childs) == 0:
        layers.append(module)
    else:
        for c in childs:
            get_layers(c, layers)


def get_layers_size(layers):
    sizes = []

    for l in layers:
        n = 0
        for p in l.parameters():
            t = 1
            for s in p.size():
                t *= s
            n += t
        sizes.append(n * 4)  # assume 4 bytes for each parameter
    return sizes
