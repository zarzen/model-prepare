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