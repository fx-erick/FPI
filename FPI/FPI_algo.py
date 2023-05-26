import numpy as np
import torch.nn.utils.prune as prune
import torch
import torch.optim as optim
import time
import json

import Development.DevUtils as devU

def get_layers_for_pruning(model, n_classes):
    """
    assumes only filters of conv. layers are pruned.
    if a conv. layer only has one output filter/channel or n_classes output filters/channels, it is ignored
    """
    layers = []
    n_filters = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.weight.shape[0] != n_classes and m.weight.shape[0] != 1:
                layers.append(m)
                n_filters += m.weight.shape[0]

    return layers, n_filters


def FPI_prune(layers, n_filters, prune_amount, method, device):
    """
    The main algorithm for FPI Pruning Method

    Args:
        layers: (torch.nn.conv2d) CNN layers
        n_filters : (int) number of filters
        prune_amout: (int) number of filters to be pruned
        method : (str) methods to measure similarity between filters "l1" , "cc", "cosine",
        device: (torch.device) where to map the data

    Returns: None

    """

    N_MAX = n_filters - int(n_filters*prune_amount)

    # Initialize selected filter list and filter array of all filters
    phi_s = []
    filters_array = []

    if not isinstance(layers, list):
        layers = [layers]

    # add filters from all layers to filters_array
    for _, layer in enumerate(layers):
        for _, filter in enumerate(layer.weight):
            filter = filter.cpu().detach().numpy()
            if filter.shape[0] > 1:
                filter = np.average(filter, axis=0)
                filter = np.expand_dims(filter, axis=0)
            filters_array.append(filter)


    filters_array = np.array(filters_array)

    # calculate correlation matrix between filters by either l1, cc or cross correlation
    if method == "l1":
        array = np.abs(filters_array[:, np.newaxis, :, :, :] - filters_array[np.newaxis, :, :, :, :])
        corr_mat = np.sum(np.sum(np.sum(array, axis=-1), axis=-1), axis=-1)
    elif method == "cc":
        array = filters_array.squeeze(axis=1)
        mean = array.mean(axis=(1, 2))
        array = array - mean[:, None, None]

        nominator = np.multiply(array[:, None, :], array[None, :, :])
        nominator = np.sum(np.sum(nominator, axis=-1), axis=-1)

        squared = array ** 2

        denom = np.sum(np.sum(squared, axis=-1), axis=-1)
        denom = np.sqrt(np.matmul(denom[:, None], denom[None, :]))

        corr_mat = np.divide(nominator, denom)
    elif method == "cosine":
        array = filters_array.squeeze(axis=1)

        nominator = np.multiply(array[:, None, :], array[None, :, :])
        nominator = np.sum(np.sum(nominator, axis=-1), axis=-1)

        squared = array ** 2

        denom = np.sum(np.sum(squared, axis=-1), axis=-1)
        denom = np.sqrt(np.matmul(denom[:, None], denom[None, :]))

        corr_mat = np.divide(nominator, denom)
    else:
        print("Method not implemented!")

    N = filters_array.shape[0]

    #initialize parameters for calculating relative correlation and redundancy
    # randomly select a filter for initializations
    i_s = np.random.randint(filters_array.shape[0])
    phi_s.append(i_s)
    n_s = len(phi_s)
    phi_u = [i for i in range(N) if i != i_s]

    corr_u = corr_mat
    corr_s = corr_u[:, i_s]

    # repeat until a desired number of selected filter is reached
    while n_s < N_MAX:
        # reshape selected and unselected correlation matrices
        corr_u = np.delete(corr_u, i_s, axis=1)
        corr_u = np.delete(corr_u, i_s, axis=0)
        corr_s = np.delete(corr_s, i_s, axis=0)

        # calculate relative correlation between unselected filters index I_u and all filters
        # calculate redundancy between selected filters and all filters
        I_u = np.sum(corr_u, axis=1) / (N - n_s + 1)
        if n_s > 1:
            I_s = np.sum(corr_s, axis=1) / n_s
        else:
            I_s = corr_s / n_s

        # calculate priority index for all filters
        I_i = I_u / I_s

        # find index of maximum I_i. if it is L1 then it is the minimum I_i value
        if method == "l1":
            i_s = np.argmin(I_i)
        else:
            i_s = np.argmax(I_i)

        # add the selected filter from the index
        phi_s.append(phi_u[i_s])
        n_s = len(phi_s)
        phi_u = [x for i, x in enumerate(phi_u) if i != i_s]

        # reshape selected filters correlation array
        corr_s = np.column_stack((corr_s, corr_u[:, i_s]))

    index_counter = 0

    # prune filters that are selected in the phi_s set
    for _, layer in enumerate(layers):
        mask = torch.zeros(layer.weight.shape).to(device)
        layer = layer.to(device)
        for idx, weight in enumerate(layer.weight):
            if index_counter in phi_s:
                mask[idx, :, :, :] += 1
            index_counter += 1
        prune.CustomFromMask(mask).apply(layer, "weight", mask)
    '''
        #global pruning implementation
        for _, layer in enumerate(layers):
        mask = torch.zeros(layer.weight.shape).to(device)
        layer = layer.to(device)
        for idx, weight in enumerate(layer.weight):
            if index_counter in phi_s:
                mask[idx, :, :, :] += 1
            index_counter += 1
        prune.CustomFromMask(mask).apply(layer, "weight", mask)
    '''

def default_prune(layer,prune_amount):
    prune.ln_structured(layer, name="weight", amount=prune_amount, n=1, dim=0)


def validate_model(model, valloader, criterion, metrics, device):
    """
    computation is performed on device.
    python float on cpu is returned

    Args:
        model: (torch.nn.Module) model instance
        valloader: (torch.utils.data.DataLoader) validation data, ideally batch_size=1
        criterion: (torch.nn.Module) criterion for performance assessment
        device: (torch.device) where to map the data

    Returns: mean performance, mean
    """
    model.to(device)
    run_crit = 0
    model.eval()
    run_metrics = {
        "vl loss": 0.0
    }

    for m in metrics:
        run_metrics["vl " + m] = 0.0

    print("-" * 150 +
          "\nVALIDATING MODEL")
    for x, y in valloader:
        x, y = x.to(device), y.to(device)

        with torch.set_grad_enabled(False):
            y_hat = model(x)
            run_crit += criterion(y_hat, y).item()

            metrics_val = compute_metrics(preds=y_hat, targets=y, metrics=metrics)
            for m in metrics:
                run_metrics["vl " + m] += metrics_val[m].item()

    for m in metrics:
        run_metrics["vl " + m] = run_metrics["vl " + m] / len(valloader)



    model.train()
    return run_metrics, run_crit / len(valloader)

def compute_metrics(preds, targets, metrics):
    """
    Wrapper to flexible compute metrics directly on gpu
    See Development.DevUtils.py for implementations
    Note that some classification metrics may require to use torch.max() or something ...

    Args:
        preds: (torch.Tensor) predictions, passed to metric funtion(s)
        targets: (torch.Tensor) targets, passed to metric funtion(s)
        metrics: (list of str) list of metric keywords e.g. 'acc.', 'bce', 'jaccard score', ...

    Returns: dictionary with metric keywords as keys and metric values as values

    """
    metrics_funcs = {
        "jaccard score": devU.JaccardScore(threshold=0.5, reduction="mean"),
        "acc. seg.": devU.AccuracySegmentation(threshold=0.5),
    }

    res_dict = {}
    for key in metrics:
        try:
            res_dict[key] = metrics_funcs[key](preds, targets)
        except:
            raise NotImplementedError(f"Error in compute_classification_metrics() for {key} " +
                                      "- make sure a metric function is implemented ... ")
    return res_dict


def calculate_pruned_flops(model, relevant_layers, shape, device):
    """
    this is a very bare bone way of calculationg the flops,
    based on the assumptions, that:
    - bias parameters are not pruned, therefore inputs to layers are not reduced
    - only convolutional layers are pruned (those in relevant_layers)
    - a pruned model with module.weight_mask is passed
    """

    FLOPs = []

    def hook(module, input, output):
        if isinstance(module, torch.nn.Conv2d):

            rem_input_filters = input[0].shape[1]
            rem_layer_filters = 0
            for i in range(0, module.weight_mask.shape[0]):
                if 1.0 in module.weight_mask[i]:
                    rem_layer_filters += 1

            # since bias terms are not pruned, no input filters can be empty
            # if inputs should be checked as well, this can be done via:
            # rem_input_filters = 0
            # for i in range(0, input[0].shape[1]):
            #    if torch.eq(torch.abs(input[0][0, i, :, :]).sum(), 0) is False:
            #         rem_input_filters += 1

            kernel_size = module.weight.shape[-1] * module.weight.shape[-2]
            n_steps = output.shape[-1] * output.shape[-2]

            # following: Liu et al: Compressing CNNs using Multi-level Filter Pruning
            #                       for the Edge Nodes of Multimedia Internet of Things
            f = n_steps * (kernel_size * rem_input_filters + 1) * rem_layer_filters
            FLOPs.append(f)

        return output.abs()

    for l in relevant_layers:
        l.register_forward_hook(hook)

    x = torch.ones(shape, device=device, requires_grad=False)
    _ = model(x)
    return sum(FLOPs)

def calculate_pruned_params(layers):
    """
    Function to calculate pruned parameters from layers by calculating number of zeros

    """
    n_pruned_layer = []
    n_params_layer = []

    for layer in layers:
        n_pruned_filters = 0
        pruned_params = 0
        for i in range(len(layer.weight)):
            weight = layer.weight[i]
            if torch.count_nonzero(weight.detach()) == 0:
                n_pruned_filters += 1
                pruned_params += np.prod(weight.shape)

        n_pruned_layer.append(n_pruned_filters)
        n_params_layer.append(pruned_params)

    return sum(n_params_layer),sum(n_pruned_layer)

def train_model(model,trainloader, valloader, max_epochs, metrics):
    """
    the core function that handles training and validation of the model.
    By default this function saves the model parameters (also called its state dict)
    of the epoch with the lowest validation set loss at the specified model_path (XY.pt).
    It additionally saves the parameters from the last training epoch as XY_last-epoch.pt.
    This may differ a bit if a lr_scheduler is used. Additional info from model.get_model_dict() is saved as XY.json.
    Based on these files the model can be rebuilt (according to XY.json) and
    its preferred state dict can be loaded (e.g. XY.pt).

    Args:
        model: (torch.nn.Module) the model that should be trained
        model_path: (str) path to save model state dict, should end with .pt
        trainloader: (torch.utils.data.DataLoader) used to update parameters
        valloader: (torch.utils.data.DataLoader) used to validate after each epoch
        max_epochs: (int) maximum number of training epochs
        optimizer_def: (list) e.g. ['adam', { 'lr': 0.001 }], passed to get_optimizer()
        criterion: (str) e.g. 'crossentropy', passed to get_criterion()
        metrics: (list) e.g. ['acc.', 'bce'], passed as arg. to compute_metrics()
        lr_scheduler: (None or str) a keyword that specifies a lr schedule. If None no schedule is used. Please mind
                      the specifications in configurations.py
        early_stopping: (bool) if True the training is aborted if vl loss is not decreasing for as long as stated in
                        configurations.py

    Returns: model_dict (contains model info), res_dict (contains train logs)

    """

    # setup return variables
    res_dict = {
        "epoch": [],
        "tr loss": [],
        "vl loss": []
    }
    for m in metrics:
        res_dict["tr " + m] = []
        res_dict["vl " + m] = []

    # get available devices for parallelization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("We are using", torch.cuda.device_count(), "GPUs!")
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()


    # build optimizer after model is on gpu (some need it to be ... )
    optimizer = optim.Adam(model.parameters(),lr = 0.001)
    criterion = torch.nn.BCELoss(weight=None, reduction="mean").to(device)

    print("-"*150 +
          "\nTRAINING STARTED")
    # ============================================================================================================

    for epoch in range(max_epochs):
        # setup some more variables to save train stats
        run_loss_tr = 0.0
        run_loss_val = 0.0
        run_metrics = {}
        for m in metrics:
            run_metrics["tr " + m] = 0.0
            run_metrics["vl " + m] = 0.0

        eptime = time.time()
        # ========================================================================================================
        # train on trainloader
        model.train()
        for data, labels in trainloader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(data).to(device)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            metrics_tr = compute_metrics(preds=outputs, targets=labels, metrics=metrics)
            run_loss_tr += loss.item()
            for m in metrics:
                run_metrics["tr " + m] += metrics_tr[m].item()

        # ========================================================================================================
        # evaluate on valloader
        model.eval()
        for data, labels in valloader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = model(data).to(device)

                loss = criterion(outputs, labels)

            metrics_val = compute_metrics(preds=outputs, targets=labels, metrics=metrics)
            run_loss_val += loss.item()
            for m in metrics:
                run_metrics["vl " + m] += metrics_val[m].item()

        res_dict["epoch"].append(epoch + 1)
        res_dict["tr loss"].append(run_loss_tr / len(trainloader))
        res_dict["vl loss"].append(run_loss_val / len(valloader))
        for m in metrics:
            res_dict["tr " + m].append(run_metrics["tr " + m] / len(trainloader))
            res_dict["vl " + m].append(run_metrics["vl " + m] / len(valloader))

        for key in res_dict.keys():
            value = str(round(res_dict[key][-1], 3))
            pad = 5 - len(value)
            value += pad*" "
            print("{}: {} | ".format(key, value), end="")
        print(f" {round(time.time()-eptime)} seconds")

    return model,res_dict, run_loss_val / len(valloader)


def save_model(model, layers, model_save_path):

    for layer in layers:
        prune.remove(layer, "weight")

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save(model.state_dict(), model_save_path)

    model_dict = model.get_model_dict()
    with open(model_save_path.replace(".pt", ".json"), "w") as mf:
        json.dump(model_dict, mf)


