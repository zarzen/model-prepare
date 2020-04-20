""" 
partition perturbed models
"""

from common import *
import os
import struct
from tqdm import tqdm


def strategy1(sizes):
    """
    """
    partitions = []
    # first batch 8MB
    # the rest 20MB each
    first_batch = 8 * 1024 * 1024

    batch = []
    cidx = 0
    acc = 0
    while acc < first_batch and cidx < len(sizes):
        batch.append(cidx)

        acc += sizes[cidx]
        cidx += 1
    partitions.append(batch)

    other_batch = 20 * 1024 * 1024

    batch = []
    acc = 0
    for i in range(cidx, len(sizes)):
        if acc < other_batch:
            pass
        else:
            partitions.append(batch)
            batch = []
            acc = 0

        batch.append(i)
        acc += sizes[i]

    if len(batch) != 0:
        partitions.append(batch)

    return partitions


def get_model_dict():
    d = {}
    for name, _ in torch_vision_models:
        d[name] = [name, _]
    for entry in nlp_models:
        d[entry[0].__name__] = entry

    return d


def save_partitions(save_to, layers, partitions, p_name):
    """"""
    for bidx in range(len(partitions)):
        batch = partitions[bidx]

        with open(os.path.join(save_to, "{}-{}.bin".format(p_name, bidx)), "wb") as ofile:
            for idx in batch:
                for param in layers[idx].parameters():
                    n_param = param.cpu().detach().numpy()
                    param_arr = n_param.flatten().tolist()
                    # append to file
                    d = struct.pack("%sf" % len(param_arr), *param_arr)
                    ofile.write(d)


def main():
    """"""
    part_strategy_fn = strategy1

    partition_dir = "../partitioned"
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)

    models_dict = get_model_dict()
    base_dir = "../perturbed-models"
    domains = os.listdir(base_dir)
    for d in domains:
        model_names = os.listdir(os.path.join(base_dir, d))
        for m_name in model_names:
            if m_name not in models_dict:
                # bypass googlenet
                continue
            print("working on model", m_name)
            perturbs = os.listdir(os.path.join(base_dir, d, m_name))
            for p_name in tqdm(perturbs):
                param_path = os.path.join(base_dir, d, m_name, p_name)
                try:
                    if d == "nlp":
                        model_class, _, pretrain_name, config_class = models_dict[m_name]
                        model = load_nlp_model(
                            param_path, model_class, config_class, pretrain_name)
                    elif d == "vision":
                        _, model_class = models_dict[m_name]
                        model = load_vision_model(param_path, model_class)

                    # unroll model layers
                    layers = []
                    get_layers(model, layers)
                    sizes = get_layers_size(layers)
                    parts = part_strategy_fn(sizes)
                    print("{} total-size {}, num-layers {} partitions:".format(p_name,
                                                                               sum(sizes), len(layers)), parts)
                    save_to = os.path.join(partition_dir, d, m_name, p_name)
                    if not os.path.exists(save_to):
                        os.makedirs(save_to)
                    save_partitions(save_to, layers, parts, p_name)
                except Exception as e:
                    with open("exceptions.txt", "a+") as ofile:
                        ofile.write("=" * 10)
                        ofile.write("{}::{}\n".format(m_name, p_name))
                        ofile.write("{}\n".format(str(e)))
                        ofile.write("-" * 10)


if __name__ == "__main__":
    main()
