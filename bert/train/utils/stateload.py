def stateLoading(model, pretrained_path):
    # Since issue: KeyError: 'unexpected key ...'
    # See https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    # Build a new dict that contains no prefix 'module.', the length of the prefix is 7
    # original saved file with DataParallel
    from collections import OrderedDict
    import torch
    state_dict = torch.load(pretrained_path, map_location='cpu')['state_dict']
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[6:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model
