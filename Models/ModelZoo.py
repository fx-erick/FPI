import Models.ModelsClassification as cModels
import Models.ModelsSegmentation as sModels
import Models.TernausNet as tNet
import Models.FullyConvNetworks as fcn


def get_model_from_zoo(model_info):
    """
    register your model here to use it with run_hyperparameters().
    allows to specify a model via the model_info argument and build it later... this was helpful to
    allow automatic grid searches etc.

    Args:
        model_info: list of type ["<model name>", { "<model args>": <args values>, ... }]

    Returns: model instance

    """
    model_name = model_info[0]
    model_args = model_info[1]
    if model_name == "ResCNN2Dv1":
        return cModels.ResCNN2Dv1(**model_args)
    elif model_name == "TernausUNet11":
        return tNet.TernausUNet11(**model_args)
    elif model_name == "TernausUNet16":
        return tNet.TernausUNet16(**model_args)
    elif model_name == "miniVGGUNet":
        return sModels.miniVGGUNet(**model_args)
    elif model_name == "UNetVGGbase":
        return sModels.UNetVGGbase(**model_args)
    elif model_name == "VGGbase":
        return cModels.VGGbase(**model_args)
    elif model_name == "UNetVGGwcc":
        return sModels.UNetVGGwcc(**model_args)
    elif model_name == "UNetVGGGroupConvs":
        return sModels.UNetVGGGroupConvs(**model_args)
    elif model_name == "dummyModel":
        return sModels.dummyModel()
    elif model_name == "UNetClassic":
        return sModels.UNetClassic(**model_args)
    elif model_name == "FCN8s":
        return fcn.FCN8s(**model_args)
    elif model_name == "FCN16s":
        return fcn.FCN16s(**model_args)
    elif model_name == "EESPNet":
        return sModels.EESPNet_Seg(**model_args)

    else:
        raise ValueError("Model not implemented in ModelZoo.py")
