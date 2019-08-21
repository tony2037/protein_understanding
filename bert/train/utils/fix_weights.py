def trigger_grad(model):
    """
    Trigger the trainable of a model:

    @model(inherit nn.Module): The model that all of trainable = !trainable
    """
    for para in model.parameters():
        para.requires_grad = (True, False)[para.requires_grad]
    return model


def disable_grad(model):
    """
    Disable the trainable of a model:

    @model(inherit nn.Module): The model that all of trainable = False
    """
    for para in model.parameters():
        para.requires_grad = False
    return model


def Able_grad(model):
    """
    Activate the trainable of a model:

    @model(inherit nn.Module): The model that all of trainable = True
    """
    for para in model.parameters():
        para.requires_grad = True
    return model
