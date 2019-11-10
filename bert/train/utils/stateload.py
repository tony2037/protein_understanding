from collections import OrderedDict
import torch

def stateLoading(model, pretrained_path):
    # Since issue: KeyError: 'unexpected key ...'
    # See https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    # Build a new dict that contains no prefix 'module.', the length of the prefix is 7
    # original saved file with DataParallel

    state_dict = torch.load(pretrained_path, map_location='cpu')['state_dict']
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('model.', '') # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model

def remove_state(state_dict, to_removes):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k not in to_removes:
            name = k.replace('model.', '')
            new_state_dict[name] = v
    return new_state_dict

def remove_prefix(state_dict, prefix):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('%s' % prefix, '')
        new_state_dict[name] = v
    return new_state_dict
