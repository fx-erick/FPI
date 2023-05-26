"""
Implements functions for model training and pruning experiments

content:
- wrapper functions that return metrics, optimizers, criterions
- training function
- workflow stuff that handles hyperparameter search and export of results
"""

import torch
import sklearn.metrics
import json
import time
import os
import sklearn.model_selection
import datetime
import torch.nn.utils.prune as prune
import copy

from configurations import get_train_config
from Models.ModelZoo import get_model_from_zoo
import Development.DevUtils as devU

from Models.TernausNet import TernausUNet11

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
        "acc.": devU.Accuracy(),
        "jaccard sim.": devU.JaccardSimilarity(reduction="mean"),
        "jaccard score": devU.JaccardScore(threshold=0.5, reduction="mean"),
        "crossentropy": torch.nn.CrossEntropyLoss(weight=None, reduction="mean"),
        "bce": torch.nn.BCELoss(weight=None, reduction="mean"),
        "acc. seg.": devU.AccuracySegmentation(threshold=0.5),
        "ruzicka": devU.RuzickaSimilarity(reduction="mean")
    }

    res_dict = {}
    for key in metrics:
        try:
            res_dict[key] = metrics_funcs[key](preds, targets)
        except:
            raise NotImplementedError(f"Error in compute_classification_metrics() for {key} " +
                                      "- make sure a metric function is implemented ... ")
    return res_dict


def get_optimizer(model_parameters, optim_info):
    """
    wrapper function to be able to specifiy the optimizer before the model is constructed.
    convenient for grid searches etc.

    Args:
        model_parameters: model parameters, that optimizer should optimize
        optim_info: list that looks like ['adam', { 'lr': 0.001 }], dict can be empty

    Returns: (torch.optim) optimizer instance

    """
    if optim_info[0] == "adam":
        return torch.optim.Adam(model_parameters, **optim_info[1])
    elif optim_info[0] == "sgd":
        return torch.optim.SGD(model_parameters, **optim_info[1])
    else:
        raise ValueError("Optimizer not implemented in get_optimizer()")


def get_criterion(criterion_name):
    """
    wrapper class to conveniently specify a criterion and get it later via this function

    Args:
        criterion_name: (str) a name which tells this function what criterion to return e.g. 'crossentropy'

    Returns: instance of the specified criterion

    """
    if criterion_name == "crossentropy":
        return torch.nn.CrossEntropyLoss(weight=None, reduction="mean")
    elif criterion_name == "mse":
        return torch.nn.MSELoss(reduction="mean")
    elif criterion_name == "jaccard loss":
        return devU.JaccardSimilarity(reduction="mean")
    elif criterion_name == "bce-jaccard loss":
        return devU.BCEJaccardSim(reduction="mean")
    elif criterion_name == "bce":
        return torch.nn.BCELoss(weight=None, reduction="mean")
    elif criterion_name == "ruzicka":
        return devU.RuzickaSimilarity(reduction="mean")
    elif criterion_name == "bce-ruzicka":
        return devU.BCERuzickaSim(reduction="mean")
    else:
        raise ValueError("Criterion not implemented in get_criterion()")


def train_model(model, model_path, trainloader, valloader, max_epochs,
                optimizer_def, criterion, metrics, lr_scheduler,
                early_stopping):
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

    model_dict = model.get_model_dict()
    model_dict["lr_scheduler"] = lr_scheduler
    min_val_loss = 100.0
    start_time = time.time()

    # only used if early_stopping == True
    n_early_stopping_epochs = get_train_config()["early_stopping_n_epochs"]
    early_stopping_ctr = 0

    # get available devices for parallelization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("We are using", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # build optimizer after model is on gpu (some need it to be ... )
    optimizer = get_optimizer(model.parameters(), optimizer_def)

    # setup lr_schedulers if specified
    if lr_scheduler == "anneal":
        anneal_lr_min = get_train_config()["anneal_lr_min"]
        anneal_lr_T0 = get_train_config()["anneal_lr_T0"]
        anneal_lr_Tmult = get_train_config()["anneal_lr_Tmult"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0=anneal_lr_T0,
                                                                         T_mult=anneal_lr_Tmult,
                                                                         eta_min=anneal_lr_min, verbose=False)
        # use early stopping variables to check if a cycle is completed
        n_early_stopping_epochs = anneal_lr_T0
        # make sure a min. of one cycle is run
        if max_epochs < anneal_lr_T0:
            max_epochs = anneal_lr_T0
            print("> increased max_epochs to", anneal_lr_T0, "since lr_scheduler == 'anneal'")
    elif lr_scheduler == "stepwise":
        patience = get_train_config()["stepwise_lr_patience"]
        stepwise_lr_min = get_train_config()["stepwise_lr_min"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode="min",
                                                               factor=0.1,
                                                               patience=patience,
                                                               min_lr=stepwise_lr_min, verbose=False)

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

                if model_dict["type"] == "EESPNet":
                    output1,output2 = model(data)
                    loss1 = criterion(output1, labels)
                    loss2 = criterion(output2, labels)
                    loss = loss1 + loss2
                    outputs = output1
                else:
                    outputs = model(data)
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
                    outputs = model(data)
                    loss = criterion(outputs, labels)

            metrics_val = compute_metrics(preds=outputs, targets=labels, metrics=metrics)
            run_loss_val += loss.item()
            for m in metrics:
                run_metrics["vl " + m] += metrics_val[m].item()

        # ========================================================================================================
        # step lr schedulers if specified
        if lr_scheduler == "anneal":
            scheduler.step()
            early_stopping_ctr += 1
        elif lr_scheduler == "stepwise":
            scheduler.step(run_loss_val)

        # ========================================================================================================
        # compute metrics for epoch, append to res_dict and print epoch stats
        res_dict["epoch"].append(epoch+1)
        res_dict["tr loss"].append(run_loss_tr/len(trainloader))
        res_dict["vl loss"].append(run_loss_val/len(valloader))
        for m in metrics:
            res_dict["tr " + m].append(run_metrics["tr " + m]/len(trainloader))
            res_dict["vl " + m].append(run_metrics["vl " + m]/len(valloader))

        for key in res_dict.keys():
            value = str(round(res_dict[key][-1], 3))
            pad = 5 - len(value)
            value += pad*" "
            print("{}: {} | ".format(key, value), end="")
        print(f" {round(time.time()-eptime)} seconds")

        # ========================================================================================================
        # saving best model and abort if early stopping is met. if cosine annealing with warm
        # restarts is used, this is only done after a cycle is finished.
        if lr_scheduler != "anneal":
            # save best performing model and perform early stopping if specified
            if res_dict["vl loss"][-1] < min_val_loss:
                min_val_loss = res_dict["vl loss"][-1]

                torch.save(model.state_dict(), model_path)
                model_dict["state_dict_path"] = model_path
                model_dict["best_epoch"] = epoch+1
                with open(model_path.replace(".pt", ".json"), "w") as mf:
                    mf.write(json.dumps(model_dict, indent=4))
                early_stopping_ctr = 0
            else:
                early_stopping_ctr += 1

            torch.save(model.state_dict(), model_path.replace(".pt", "_last_epoch.pt"))  # to also obtain the parameters from the last epoch

            # abort if early stopping criteria met
            if early_stopping and (early_stopping_ctr >= n_early_stopping_epochs):
                break
        elif (lr_scheduler == "anneal") and (early_stopping_ctr == n_early_stopping_epochs):
            # only save model after an annealing cycle is finished to avoid noise in performance
            # and allow to use the epoch for re-training of the model
            # if early stopping is used, training is stopped after a cycle without improvement

            torch.save(model.state_dict(), model_path.replace(".pt", "_last_epoch.pt"))  # to also obtain the parameters from the last epoch

            if res_dict["vl loss"][-1] < min_val_loss:
                min_val_loss = res_dict["vl loss"][-1]
                torch.save(model.state_dict(), model_path)
                model_dict["state_dict_path"] = model_path
                model_dict["best_epoch"] = epoch + 1
                early_stopping_ctr = 0
                with open(model_path.replace(".pt", ".json"), "w") as mf:
                    json.dump(model_dict, mf)
            else:
                if early_stopping:
                    break
        else:
            pass

    print(f"TRAINING FINISHED after {round((time.time()-start_time)/60, 2)} minutes")
    print("-"*150)
    del optimizer  # paranoia of memory leakage
    return model_dict, res_dict


def run_single_hyperparameter_combination_from_scratch(hps_name, params, trainloader, valloader):
    """
    called by run_hyperparameters() handles the export of the results, creates timestamps, prints stuff etc.
    only handles a single hyperparameter combination.

    Args:
        hps_name: (str) a name of the experiment run, will be added before the timestamp of exported files
        params: (dict) one item in the list returned by build_dataloaders_and_param_grid()
        trainloader: (torch.utils.data.DataLoader) used to update parameters
        valloader: (torch.utils.data.DataLoader) used to validate after each epoch

    Returns: dict with model info, dict with train logs

    """
    # set up administrative variables
    idx = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    r_name = "{}_{}".format(hps_name, str(idx))
    model_path = get_train_config()["export_path"] + r_name + ".pt"

    overview_dict = {
        "run id": r_name,
        "mode": params["mode"],
        "model name": params["model"][0],
        "model args": str(params["model"][1]),
        "model path": model_path,
        "optimizer": params["optimizer"][0],
        "optimizer args": str(params["optimizer"][1]),
        "criterion": params["criterion"],
        "batchsize": params["batchsize"],
        "lr scheduler": params["lr_scheduler"],
        "max. epochs": params["max_epochs"],
        "early stopping": params["early_stopping"]
    }

    print("=" * 150)
    print("CURRENT HYPERPARAMETERS - TRAINING FROM SCRATCH")
    for k in overview_dict.keys():
        print("\t{}: {}".format(k, overview_dict[k]))

    # get fundamentals
    model = get_model_from_zoo(params["model"])
    criterion = get_criterion(params["criterion"])

    # perform a train run
    model_d, res_d = train_model(model=model, model_path=model_path,
                                 trainloader=trainloader, valloader=valloader,
                                 max_epochs=params["max_epochs"],
                                 optimizer_def=params["optimizer"],
                                 criterion=criterion,
                                 metrics=get_train_config()[params["mode"]]["train_metrics"],
                                 lr_scheduler=params["lr_scheduler"],
                                 early_stopping=params["early_stopping"])

    # avoid risk of memory overflow
    del model, criterion

    # export results
    overview_dict["best epoch"] = model_d["best_epoch"]
    with open(model_path.replace(".pt", "_hp-overview.json"), "w") as mf:
        mf.write(json.dumps(overview_dict, indent=4))

    with open(model_path.replace(".pt", "_train-logs.json"), "w") as mf:
        mf.write(json.dumps(res_d, indent=4))

    print("EXPORTED RESULTS")
    return model_d, res_d


def run_existing_model(model, hps_name, params, trainloader, valloader, save_init_model=False):
    """
    similar to run_single_hyperparameter_combination_from_scratch, handles all the export and printing stuff,
    but allows to retrain a model by passing a complete model instance.

    Args:
        model: (torch.nn.Module) a model instance
        hps_name: (str) a name of the experiment run, will be added before the timestamp of exported files
        params: (dict) one item in the list returned by build_dataloaders_and_param_grid()
        trainloader: (torch.utils.data.DataLoader) used to update parameters
        valloader: (torch.utils.data.DataLoader) used to validate after each epoch
        save_init_model: (bool) optional, if True the state dict of the model before training is also saved

    Returns: dict with model info, dict with train logs

    """
    # set up administrative variables
    idx = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    r_name = "{}_{}".format(hps_name, str(idx))
    model_path = get_train_config()["export_path"] + r_name + ".pt"

    model_info = model.get_model_dict()
    model_args = {}
    for key in model_info.keys():
        if key != "type":
            model_args[key] = model_info[key]

    if save_init_model:
        torch.save(model.state_dict(), model_path.replace(".pt", "_init.pt"))

    overview_dict = {
        "run id": r_name,
        "mode": params["mode"],
        "model name": model_info["type"],
        "model args": model_args,
        "model path": model_path,
        "optimizer": params["optimizer"][0],
        "optimizer args": str(params["optimizer"][1]),
        "criterion": params["criterion"],
        "batchsize": params["batchsize"],
        "lr scheduler": params["lr_scheduler"],
        "max. epochs": params["max_epochs"],
        "early stopping": params["early_stopping"]
    }

    if list(params.keys()).__contains__("pruning method"):
        overview_dict["pruning method"] = params["pruning method"]
        overview_dict["pruning amount"] = params["pruning amount"]
        overview_dict["pruning iter"] = params["pruning iter"]

    print("=" * 150)
    print("CURRENT HYPERPARAMETERS - TRAINING EXISTING MODEL")
    for k in overview_dict.keys():
        print("\t{}: {}".format(k, overview_dict[k]))

    # get fundamentals
    criterion = get_criterion(params["criterion"])

    # perform a train run
    model_d, res_d = train_model(model=model, model_path=model_path,
                                 trainloader=trainloader, valloader=valloader,
                                 max_epochs=params["max_epochs"],
                                 optimizer_def=params["optimizer"],
                                 criterion=criterion,
                                 metrics=get_train_config()[params["mode"]]["train_metrics"],
                                 lr_scheduler=params["lr_scheduler"],
                                 early_stopping=params["early_stopping"])

    # avoid risk of memory overflow
    del model, criterion

    # export results
    overview_dict["best epoch"] = model_d["best_epoch"]
    with open(model_path.replace(".pt", "_hp-overview.json"), "w") as mf:
        json.dump(overview_dict, mf)

    with open(model_path.replace(".pt", "_train-logs.json"), "w") as mf:
        json.dump(res_d, mf)

    print("EXPORTED RESULTS")
    return model_d, res_d


def build_dataloaders_and_param_grid(grid_dict, trainset, valset):
    """
    builds dataloaders from datasets and a list of hyperparameter combinations for multiple runs

    Args:
        grid_dict: (dict) information on the hyperparameters you want to use for your experiment(s)
        trainset: (torch.utils.data.Dataset) your dataset with training data
        valset: (torch.utils.data.Dataset) your dataset with validation data

    Returns: list of hyperparameter combinations, trainloader, valloader

    """
    # before making parameter grid, catch undesired behaviour of sklearn if one parameter is not in a list
    for v in grid_dict.values():
        if type(v) != list:
            raise ValueError("All values of grid_dict should be lists," +
                             "to avoid unintended behaviour of sklearn ParameterGrid()")
    run_params = list(sklearn.model_selection.ParameterGrid(grid_dict))
    print("=" * 150)
    print("RUNNING {} HYPERPARAMETER COMBINATIONS".format(len(run_params)))

    # check if dataloaders can be declared once or have to be declared in every run
    if len(grid_dict["batchsize"]) != 1:
        raise ValueError("Sorry, only a single batchsize per hyperparameter search is allowed currently")

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=grid_dict["batchsize"][0],
                                              shuffle=True, num_workers=8, pin_memory = True)
    valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=64, shuffle=False, num_workers=8, pin_memory = True)

    return run_params, trainloader, valloader


def run_hyperparameters(hps_name, grid_dict, trainset, valset):
    """
    passes the grid_dict and datasets to build_dataloaders_and_param_grid()
    then run_single_hyperparameter_combination_from_scratch() is called for all hyperparameter combinations
    contained in grid_dict

    exports quite a lot to export path of configurations.py:
        -  [hps_name]_[timestamp].json: contains model dict and path to state dict etc
        -  [hps_name]_[timestamp].pt: state dict of epoch with lowest vl loss
        -  [hps_name]_[timestamp]_hp-overview.json: used hyperprameters for this run
        -  [hps_name]_[timestamp]_last-epoch.pt: state dict after last training epoch
        -  [hps_name]_[timestamp]_train-logs.json: training logs of run

    Args:
        hps_name: (str) name for your experiment runs, a timestamp will be added to files
        grid_dict: (dict) hyperparameter combination(s)
        trainset: (torch.utils.data.Dataset) your dataset with training data
        valset: (torch.utils.data.Dataset) your dataset with validation data

    Returns: void

    """
    # get parameter grid and datalaoders
    run_params, trainloader, valloader = build_dataloaders_and_param_grid(grid_dict=grid_dict,
                                                                          trainset=trainset,
                                                                          valset=valset)

    # loop over hyperparameter combinations
    for params in run_params:
        _, _ = run_single_hyperparameter_combination_from_scratch(hps_name=hps_name, params=params,
                                                                  trainloader=trainloader, valloader=valloader)


def run_hyperparameters_with_pruning_steps(hps_name,
                                           grid_dict,
                                           trainset,
                                           valset,
                                           pruning_method,
                                           n_pruning_iters,
                                           amount,
                                           shrutika_prune_args=None):
    """
    DEPRECIATED! probably should not be used, is here for backwards compatibility
    """
    # get parameter grid and datalaoders
    run_params, trainloader, valloader = build_dataloaders_and_param_grid(grid_dict=grid_dict,
                                                                          trainset=trainset,
                                                                          valset=valset)

    if len(run_params) > 1:
        raise ValueError("more than one hyperparameter combination is currently not supported for pruning")
    params = run_params[0]

    if grid_dict["model"][0][0] != "UNetVGGbase" and pruning_method != "theresa":
        raise ValueError("run_hyperparameters_with_pruning_steps is only intended for UNetVGGbase models ..." +
                         "you should probably check/adapt its implementation in Workflow.py")

    if shrutika_prune_args is not None:
        shrutika_prune_args["prune"] = amount

    # run model with newly initialized weights
    model_d, _ = run_single_hyperparameter_combination_from_scratch(hps_name=hps_name + "_pr-Base", params=params,
                                                                    trainloader=trainloader, valloader=valloader)

    for i in range(0, n_pruning_iters):
        model = devU.load_model_from_json_dict(model_d)

        conv_layers = []
        # FCN8s dos not fit into the UNet base class, therefore must be handled a little different
        if model.get_model_dict()["type"].__contains__("FCN"):
            for m in model.modules():
                if isinstance(m, torch.nn.Conv2d):
                    if m.weight.shape[0] > 1:
                        conv_layers.append(m)
        else:
            for m in model.down_blocks.modules():
                if isinstance(m, torch.nn.Conv2d):
                    conv_layers.append(m)
            for m in model.latent.modules():
                if isinstance(m, torch.nn.Conv2d):
                    conv_layers.append(m)
            for m in model.up_blocks.modules():
                if isinstance(m, torch.nn.Conv2d):
                    conv_layers.append(m)

        if pruning_method == "theresa":
            model.set_own_weights_below_threshold_to_zero(threshold=amount)

        elif pruning_method == "l1_unstructured":
            print(f"\n>pruning with pytorch l1 unstructured, {amount*100} % of parameters are pruned\n")
            for layer in conv_layers:
                prune.l1_unstructured(layer, name="weight", amount=amount)
                #prune.remove(layer, "weight")

        elif pruning_method == "random_unstructured":
            print(f"\n>pruning with pytorch random unstructured, {amount*100} % of parameters are pruned\n")
            for layer in conv_layers:
                prune.random_unstructured(layer, name="weight", amount=amount)
                #prune.remove(layer, "weight")

        elif pruning_method == "l1_structured":
            print(f"\n>pruning with pytorch l1 structured, {amount*100} % of parameters are pruned\n")
            for layer in conv_layers:
                prune.ln_structured(layer, name="weight", amount=amount, n=1, dim=0)
                #prune.remove(layer, "weight")

        elif pruning_method == "random_structured":
            print(f"\n>pruning with pytorch random structured, {amount*100} % of parameters are pruned\n")
            for layer in conv_layers:
                prune.random_structured(layer, name="weight", amount=amount, dim=0)
                #prune.remove(layer, "weight")

        elif pruning_method == "shrutika":
            print(f"\n>pruning with shrutika pruning, {amount * 100} % of filters are pruned\n")
            for layer in conv_layers:
                devU.shrutika_prune(layer, shrutika_prune_args)

        params["pruning method"] = pruning_method
        params["pruning amount"] = amount
        params["pruning iter"] = i+1

        model_d, _ = run_existing_model(model=model, hps_name=hps_name + f"_pr-{i+1}", params=params,
                                        trainloader=trainloader, valloader=valloader)


def prune_conv_layers_of_model(model, pruning_method, amount, shrutika_prune_args=None):
    """
    takes a model and prunes its conv layers with the specified method and amount.
    !! The selection of conv layers is highly dependent on model structure, maybe adapt this function for new models

    Args:
        model: (torch.nn.Module) model instance
        pruning_method: (str) e.g. 'l1_unstructured' or 'shrutika'
        amount: (float) between 0 and 1, this share will be pruned per layer
        shrutika_prune_args: (dict or None) additional arguments required by shrutika filter pruning method

    Returns: void (pruning happens inplace)

    """
    conv_layers = []
    # FCN8s dos not fit into the UNet base class,
    # therefore conv layers for pruning must be selected a little different
    if model.get_model_dict()["type"].__contains__("FCN") or model.get_model_dict()["type"].__contains__("TernausUNet11") \
            or model.get_model_dict()["type"].__contains__("EESPNet"):
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                if m.weight.shape[0] > 1:
                    conv_layers.append(m)
    else:
        for m in model.down_blocks.modules():
            if isinstance(m, torch.nn.Conv2d):
                conv_layers.append(m)
        for m in model.latent.modules():
            if isinstance(m, torch.nn.Conv2d):
                conv_layers.append(m)
        for m in model.up_blocks.modules():
            if isinstance(m, torch.nn.Conv2d):
                conv_layers.append(m)

    if pruning_method == "theresa":
        """
        please do not use without inspection... 
        """
        model.set_own_weights_below_threshold_to_zero(threshold=amount)

    elif pruning_method == "l1_unstructured":
        print(f"\n>pruning with pytorch l1 unstructured, {amount * 100} % of parameters are pruned\n")
        for layer in conv_layers:
            prune.l1_unstructured(layer, name="weight", amount=amount)

    elif pruning_method == "random_unstructured":
        print(f"\n>pruning with pytorch random unstructured, {amount * 100} % of parameters are pruned\n")
        for layer in conv_layers:
            prune.random_unstructured(layer, name="weight", amount=amount)

    elif pruning_method == "l1_structured":
        print(f"\n>pruning with pytorch l1 structured, {amount * 100} % of parameters are pruned\n")
        for layer in conv_layers:
            prune.ln_structured(layer, name="weight", amount=amount, n=1, dim=0)

    elif pruning_method == "random_structured":
        print(f"\n>pruning with pytorch random structured, {amount * 100} % of parameters are pruned\n")
        for layer in conv_layers:
            prune.random_structured(layer, name="weight", amount=amount, dim=0)

    elif pruning_method == "shrutika":
        print(f"\n>pruning with shrutika pruning, {amount * 100} % of filters are pruned\n")
        for layer in conv_layers:
            devU.shrutika_prune(layer, amount, shrutika_prune_args)


def validate_model(model, params, valloader):
    """
    validate a passed model with the passed hyperparameters on the passed validation dataloader once.
    Args:
        model: model instance
        params: dict with hyperparameters (criterion, mode)
        valloader: instance of torch.utils.data.DataLoader

    Returns: dict with logs
    """

    # get available devices for parallelization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("We are using", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # get and declare stuff
    criterion = get_criterion(params["criterion"])
    metrics = get_train_config()[params["mode"]]["inference_metrics"]
    run_loss_val = 0.0
    run_metrics = {
        "vl loss": 0.0
    }
    for m in metrics:
        run_metrics["vl " + m] = 0.0

    res_dict = {}

    # ========================================================================================================
    # evaluate on valloader
    model.eval()
    for data, labels in valloader:
        data, labels = data.to(device), labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(data).to(device)
            loss = criterion(outputs, labels)

        metrics_val = compute_metrics(preds=outputs, targets=labels, metrics=metrics)
        run_loss_val += loss.item()
        for m in metrics:
            run_metrics["vl " + m] += metrics_val[m].item()

    # ========================================================================================================
    # compute metrics, append to res_dict and print stats
    res_dict["vl loss"] = run_loss_val / len(valloader)
    for m in metrics:
        res_dict["vl " + m] = run_metrics["vl " + m] / len(valloader)

    for key in res_dict.keys():
        value = str(round(res_dict[key], 3))
        pad = 5 - len(value)
        value += pad * " "
        print("{}: {} | ".format(key, value), end="")
    print()
    return res_dict


def run_single_pruning_experiment(hps_name, retrain_grid_dict, model_json_path, pruning_spec, trainset, valset, imagenet_pretrained = False ):
    """
    used to perform pruning experiments.
    1. initial validation of model from model_json_path
    2. pruning acc. to pruning_spec
    3. validation of pruned model
    4. finetuning
    5. exporting a lot of results and state dicts to export_path in configurations.py
        - [hps_name]_results_[timestamp].json: results on validation set of different phases
        - [hps_name]_retrained_[timestamp].json: contains model dict and path to state dict etc
        - [hps_name]_retrained_[timestamp].pt: state dict epoch with lowest vl loss during finetuning
        - [hps_name]_retrained_[timestamp]_hp-overview.json: hyperparameter for finetuning
        - [hps_name]_retrained_[timestamp]_init.pt: state dict after pruning before finetuning
        - [hps_name]_retrained_[timestamp]_last-epoch.pt: state dict of last epoch of finetuning
        - [hps_name]_retrained_[timestamp]_train-logs.json: train logs of finetuning stage

    Args:
        hps_name: (str) name for your experiment runs, a timestamp will be added to files
        retrain_grid_dict: (dict) hyperparameters used for finetuning of pruned model
        model_json_path: (str) path to XY.json file of model, the state dict XY.pt will be loaded
        pruning_spec: (dict) e.g. {"method": "l1_structured", "amount": 0.1, "shrutika_prune_args": None}
        rainset: (torch.utils.data.Dataset) your dataset with training data
        valset: (torch.utils.data.Dataset) your dataset with validation data

    Returns: void

    """

    res_dict = {}

    run_params, trainloader, valloader = build_dataloaders_and_param_grid(grid_dict=retrain_grid_dict,
                                                                          trainset=trainset,
                                                                          valset=valset)

    if len(run_params) > 1:
        raise ValueError("Sorry currently only a single hp-combination is supported," +
                         " please adapt run_single_pruning_experiment()")

    # get model and perform validation to check
    print("-" * 150 + "\nINITIAL VALIDATION")
    if imagenet_pretrained:
        model = TernausUNet11(pretrained=True)
    else:
        model = devU.load_model_from_json_path(model_json_path=model_json_path, load_state_dict=True)

    model.eval()
    res_dict["base_vl"] = validate_model(model=model, params=run_params[0], valloader=valloader)
    print("-"*150)

    # prune model and perform initial validation
    prune_conv_layers_of_model(model=model,
                               pruning_method=pruning_spec["method"],
                               amount=pruning_spec["amount"],
                               shrutika_prune_args=pruning_spec["shrutika_prune_args"])

    print("-" * 150 + "\nPRUNED VALIDATION")
    res_dict["pruned_vl"] = validate_model(model=model, params=run_params[0], valloader=valloader)
    print("-" * 150)

    # retrain model and validate again
    model.train()
    model_d, run_logs = run_existing_model(model=model, hps_name=hps_name + "_retrained", params=run_params[0],
                                           trainloader=trainloader, valloader=valloader, save_init_model=True)

    res_dict["pruned_retrained_vl_min_loss_epoch"] = {}
    res_dict["pruned_retrained_vl_last_epoch"] = {}
    for k, v in run_logs.items():
        res_dict["pruned_retrained_vl_min_loss_epoch"][k] = v[model_d["best_epoch"]-1]
        res_dict["pruned_retrained_vl_last_epoch"][k] = v[-1]

    res_dict_path = model_d["state_dict_path"].replace("_retrained_", "_results_").replace(".pt", ".json")
    with open(res_dict_path, "w") as file:
        file.write(json.dumps(res_dict, indent=4))


if __name__ == "__main__":
    pass
