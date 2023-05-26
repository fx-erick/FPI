import torch
import json
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.utils.prune as prune
#import matplotlib.pyplot as plt

from Models.ModelZoo import get_model_from_zoo


def check_cuda():
    print("Device 0:", torch.cuda.get_device_name(0))
    print("Cuda available:", torch.cuda.is_available())

    x = torch.rand((3, 2))
    y = torch.rand((2, 3))
    x, y = x.cuda(), y.cuda()
    res = torch.matmul(x, y)

    print("Test Tensor Device:", res.device)


def set_manual_seed(pytorch=False):
    """
    Sets the random/manual seed for random, pytorch and numpy according to configurations.py
    Since pandas relies on numpy, this also makes pandas deterministic.
    ! setting torch deterministic may substantially decrease model performance !
    """
    import numpy
    import random
    from configurations import get_train_config

    seed = get_train_config()["random_seed"]
    random.seed(seed)
    numpy.random.seed(seed)
    if pytorch:
        import torch
        torch.manual_seed(seed)


class BCEJaccardSim(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(BCEJaccardSim, self).__init__()
        self.reduction = reduction
        self.BCELoss = torch.nn.BCELoss(weight=None, reduction=reduction)
        self.Jaccard = JaccardSimilarity(reduction=reduction)
        if self.reduction != "mean":
            raise NotImplementedError(
                "BCEJaccardSim has no reduction {}, use 'mean' instead".format(self.reduction))

    def forward(self, outputs, targets):
        bce = self.BCELoss(outputs, targets)
        jac = self.Jaccard(outputs, targets)
        return bce + jac


class BCERuzickaSim(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(BCERuzickaSim, self).__init__()
        self.BCELoss = torch.nn.BCELoss(weight=None, reduction=reduction)
        self.Ruzicka = RuzickaSimilarity(reduction=reduction)
        if reduction != "mean":
            raise NotImplementedError(
                "BCERuzickaSim has no reduction {}, use 'mean' instead".format(self.reduction))

    def forward(self, outputs, targets):
        bce = self.BCELoss(outputs, targets)
        ruz = self.Ruzicka(outputs, targets)
        return bce + ruz


class JaccardSimilarity(torch.nn.Module):
    """
    jaccard similarity acc. to ternausNet paper
    """
    def __init__(self, reduction):
        super(JaccardSimilarity, self).__init__()
        self.reduction = reduction
        self.epsilon = 0.00001
        if self.reduction != "mean":
            raise NotImplementedError(
                "JaccardSimilarity has no reduction {}, use 'mean' instead".format(self.reduction))

    def forward(self, outputs, targets):
        targets = targets
        outputs = outputs
        nom = targets * outputs
        denom = targets + outputs - nom + self.epsilon
        return ((nom / denom) + self.epsilon).mean().log() * -1.


class RuzickaSimilarity(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(RuzickaSimilarity, self).__init__()
        self.epsilon = 0.5
        if reduction != "mean":
            raise NotImplementedError("RuzickSimilarity only implemented for reduction 'mean'")

    def forward(self, outputs, targets):
        outputs = outputs + self.epsilon
        targets = targets + self.epsilon
        r = torch.min(outputs[0], targets[0]).sum() / torch.max(outputs[0], targets[0]).sum()
        for i in range(1, targets.shape[0]):
            r = r + torch.min(outputs[i], targets[i]).sum() / torch.max(outputs[i], targets[i]).sum()
        r = r / targets.shape[0]
        return r.log() * -1


class JaccardScore(torch.nn.Module):
    def __init__(self, threshold, reduction):
        super(JaccardScore, self).__init__()
        self.t = threshold
        if reduction != "mean":
            raise NotImplementedError("JaccardScore only implemented for reduction 'mean'")

    def calc_jac(self, preds, targets):
        """
        partly derived from this implementation:
        https://gitlab.com/theICTlab/UrbanReconstruction/ictnet/-/blob/master/code/compute_accuracy.py
        """
        inters = (targets & preds).float().sum()
        union = (targets | preds).float().sum()
        # catch union of 0 to avoid div by 0 (min. union = 1.0)
        if union < 0.1:
            return inters
        else:
            return inters / union

    def forward(self, outputs, targets):
        if outputs.shape[0] != targets.shape[0]:
            raise ValueError("Error in JaccardScore: output batch dim must equal target batch dim")

        preds = torch.ge(outputs, self.t).int()
        targets = targets.int()

        accum_jac = 0.0
        for i in range(0, outputs.shape[0]):
            accum_jac += self.calc_jac(preds=preds[i], targets=targets[i])

        return accum_jac / float(outputs.shape[0])


class Accuracy(torch.nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, outputs, targets):
        _, preds = torch.max(outputs, 1)

        if targets.shape[0] != preds.shape[0]:
            raise ValueError("targets shape batch dim {} must equal preds batch dim {}"
                             .format(targets.shape, preds.shape))

        n = targets.shape[0]
        correct = (preds == targets).float().sum()
        return correct / n


class AccuracySegmentation(torch.nn.Module):
    def __init__(self, threshold):
        super(AccuracySegmentation, self).__init__()
        self.t = threshold

    def forward(self, outputs, targets):
        if outputs.shape[0] != targets.shape[0]:
            raise ValueError("Error in JaccardScore: output batch dim must equal target batch dim")

        n = 1
        for ns in targets.shape:
            n *= ns

        preds = torch.ge(outputs, self.t).int()
        targets = targets.int()
        correct = (preds == targets).float().sum()
        return correct / n


def get_model_parameters(model_conf, input_size, full_model=None):
    from torchsummary import summary
    from Models.ModelZoo import get_model_from_zoo

    if full_model == None:
        model = get_model_from_zoo(model_info=model_conf)
    else:
        model = full_model

    print("PYTORCH PRINT:\n" + "="*150)
    print(model)
    print("=" * 150)
    summary(model=model, input_size=input_size)


def load_model_from_json_path(model_json_path):
    with open(model_json_path) as file0:
        model_json = json.load(file0)

    model_type = model_json["type"]
    state_dict_path = model_json_path.replace(".json", ".pt")
    model_args = {}
    for key in model_json:
        if key != "type" and \
                key != "state_dict_path" and \
                key != "best_epoch" and \
                key != "lr_scheduler":
            model_args[key] = model_json[key]

    model = get_model_from_zoo((model_type, model_args))
    model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
    return model


def load_model_from_json_dict(model_json):
    model_type = model_json["type"]
    state_dict_path = model_json["state_dict_path"]
    model_args = {}
    for key in model_json:
        if key != "type" and \
                key != "state_dict_path" and \
                key != "best_epoch" and \
                key != "lr_scheduler":
            model_args[key] = model_json[key]

    model = get_model_from_zoo((model_type, model_args))
    model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
    return model


def shrutika_prune_compute_mask(m, prune=0.5, n_clusters=10, n_iters=5, alpha=0.5, verbose=None, scoring_fn="l1"):
    """
    runtime for vgg11 on hp notebook (python by default does not utilize more than on core): 105 minutes!
    - corr_coef distance/similarity measure is currently faulty: its use in the update term has to be adapted as
      well as the calculation (e.g. like https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739)

    """
    if isinstance(m, torch.nn.Conv2d):
        print(f">> Pruning Module {m} with weights of shape", m.weight.shape)
        n_in_filters_for_layer = m.weight.shape[1]  # in theory this should consider all dims of each filter
        # getting weights and abs. means of all filters
        filter_weights = []
        filter_means = []
        for i in range(0, len(m.weight)):
            i_filter = m.weight[i]
            i_filter = i_filter.flatten()
            filter_weights.append(i_filter.detach().cpu().numpy())

        min_weight = min(np.array(filter_weights).flatten())
        max_weight = max(np.array(filter_weights).flatten())

        # calculation of filter means
        for i in range(0, len(filter_weights)):
            scaled_weights = ((filter_weights[i] - min_weight) / (max_weight - min_weight))
            mean = np.abs(scaled_weights).mean()
            filter_means.append(mean)

        # clustering of filters acc. to mean abs. value
        filter_means = np.array(filter_means)
        filter_means = filter_means.reshape((-1, 1))
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=300, random_state=123)
        kmeans.fit(filter_means)
        pred = kmeans.predict(filter_means)

        # aggregate variables acc. to cluster nr
        clustered = {}
        for nc in range(0, n_clusters):
            clustered[nc] = {
                "means": [],
                "weights": [],
                "scores_last_iter": [],
                "final_scores": [],
                "n_items": 0,
                "idx": []
            }

        for i in range(0, len(filter_means)):
            clustered[pred[i]]["means"].append(filter_means[i][0])
            clustered[pred[i]]["weights"].append(filter_weights[i])
            clustered[pred[i]]["scores_last_iter"].append(filter_means[i][0])  # initial value for score calc.
            clustered[pred[i]]["n_items"] += 1
            clustered[pred[i]]["idx"].append(i)

        # perform first score calculation run for each cluster
        for kkey in clustered.keys():
            for i in range(0, clustered[kkey]["n_items"]):
                sum_term = 0.0
                for n in range(0, clustered[kkey]["n_items"]):
                    if i != n:
                        if scoring_fn == "l1":
                            diff_in = clustered[kkey]["weights"][i] - clustered[kkey]["weights"][n]
                            l1_in = np.abs(diff_in).sum() / n_in_filters_for_layer
                            sum_term += l1_in * clustered[kkey]["scores_last_iter"][n]
                        elif scoring_fn == "l2":
                            diff_in = clustered[kkey]["weights"][i] - clustered[kkey]["weights"][n]
                            l2_in = np.linalg.norm(diff_in, 2) / n_in_filters_for_layer
                            sum_term += l2_in * clustered[kkey]["scores_last_iter"][n]
                        else:
                            raise NotImplementedError("this scoring function is unknown")

                        # elif scoring_fn == "cc":
                        #    corr_coeff = np.corrcoef(clustered[kkey]["weights"][i], clustered[kkey]["weights"][n])
                        #    #print(corr_coeff)
                        #    cc = np.abs(corr_coeff).mean()
                        #    sum_term += cc * clustered[kkey]["scores_last_iter"][n]

                score = clustered[kkey]["means"][i] + sum_term
                score = score * alpha
                clustered[kkey]["final_scores"].append([score])
                if verbose is not None and (i == verbose and kkey == verbose):
                    print(f">> cluster {kkey}, filter {i}, init. score: {score}")

        # update scores of last iter
        for kkey in clustered.keys():
            for i in range(0, clustered[kkey]["n_items"]):
                clustered[kkey]["scores_last_iter"][i] = clustered[kkey]["final_scores"][i][-1]

        # perform remaining score calculation runs
        for t in range(2, n_iters+1):
            for kkey in clustered.keys():
                # term 1/2
                for ik in range(0, clustered[kkey]["n_items"]):
                    sum_term = 0.0
                    for nk in range(0, clustered[kkey]["n_items"]):
                        if ik != nk:
                            if scoring_fn == "l1":
                                diff_in = clustered[kkey]["weights"][ik] - clustered[kkey]["weights"][nk]
                                l1_in = np.abs(diff_in).sum() / n_in_filters_for_layer
                                sum_term += l1_in * clustered[kkey]["scores_last_iter"][nk]

                            elif scoring_fn == "l2":
                                diff_in = clustered[kkey]["weights"][ik] - clustered[kkey]["weights"][nk]
                                l2_in = np.linalg.norm(diff_in, 2) / n_in_filters_for_layer
                                sum_term += l2_in * clustered[kkey]["scores_last_iter"][nk]

                            # elif scoring_fn == "cc":
                            #    corr_coeff = np.corrcoef(clustered[kkey]["weights"][ik], clustered[kkey]["weights"][nk])
                            #    #print(corr_coeff)
                            #    cc = np.abs(corr_coeff).mean()
                            #    sum_term += cc * clustered[kkey]["scores_last_iter"][nk]

                    score_1 = sum_term * alpha

                    # term 2/2
                    sum_of_sum_term = 0.0
                    for jkey in clustered.keys():
                        sum_j = 0.0
                        if jkey != kkey:
                            for ij in range(0, clustered[jkey]["n_items"]):
                                for nj in range(0, clustered[jkey]["n_items"]):
                                    if ij != nj:
                                        if scoring_fn == "l1":
                                            diff_in = clustered[jkey]["weights"][ij] - clustered[jkey]["weights"][nj]
                                            l1_in = np.abs(diff_in).sum() / n_in_filters_for_layer
                                            sum_j += l1_in * clustered[jkey]["scores_last_iter"][nj]

                                        elif scoring_fn == "l2":
                                            diff_in = clustered[jkey]["weights"][ij] - clustered[jkey]["weights"][nj]
                                            l2_in = np.linalg.norm(diff_in, 2) / n_in_filters_for_layer
                                            sum_j += l2_in * clustered[jkey]["scores_last_iter"][nj]

                                        # elif scoring_fn == "cc":
                                        #    corr_coeff = np.corrcoef(clustered[jkey]["weights"][ij], clustered[jkey]["weights"][nj])
                                        #    cc = np.abs(corr_coeff).mean()
                                        #    #print(corr_coeff)
                                        #    sum_j += cc * clustered[jkey]["scores_last_iter"][nj]

                        sum_of_sum_term += sum_j
                    score_2 = sum_of_sum_term * ((1 - alpha)/(n_clusters - 1))

                    score = score_1 - score_2
                    if verbose is not None and (ik == verbose and kkey == verbose):
                        print(f">> cluster {kkey}, filter {ik}, iter {t} score: {score},\tprev. iter score: {clustered[kkey]['scores_last_iter'][ik]}")

                    clustered[kkey]["final_scores"][ik].append(score)

            # update scores of last iter after scores for all items in all clusters are computed
            for kkey in clustered.keys():
                for i in range(0, clustered[kkey]["n_items"]):
                    clustered[kkey]["scores_last_iter"][i] = clustered[kkey]["final_scores"][i][-1]

        # get max scores for filters out of all t
        max_filter_scores = {}
        for kkey in clustered.keys():
            max_filter_scores[kkey] = []
            for all_filter_scores in clustered[kkey]["final_scores"]:
                max_filter_scores[kkey].append(max(all_filter_scores))

        # normalize filter scores
        norm_max_filter_scores = {}
        scaled_max_filter_scores = {}
        for kkey in max_filter_scores.keys():
            scaled_max_filter_scores[kkey] = []
            min_kkey = min(max_filter_scores[kkey])
            max_kkey = max(max_filter_scores[kkey])
            if len(max_filter_scores[kkey]) > 1:
                for n_filters in max_filter_scores[kkey]:
                    scaled_max_filter_scores[kkey].append(((n_filters - min_kkey)/(max_kkey - min_kkey)))
            else:
                scaled_max_filter_scores[kkey].append(max_filter_scores[kkey][0]/max_kkey)

        # softmax scale filter scores and append filter indices
        for kkey in scaled_max_filter_scores.keys():
            norm_max_filter_scores[kkey] = []
            for n_filter in scaled_max_filter_scores[kkey]:
                norm_max_filter_scores[kkey].append(np.exp(n_filter)/sum(np.exp(scaled_max_filter_scores[kkey])))

        #print("\nScaled Max Filter Scores ============================================")
        #print(json.dumps(scaled_max_filter_scores, indent=4))

        #print("\nNormalized Max Filter Scores ============================================")
        #print(json.dumps(norm_max_filter_scores, indent=4))

        # build and return mask
        results = {
            "scores": [],
            "idx": []
        }
        for k in norm_max_filter_scores.keys():
            for i in range(0, len(norm_max_filter_scores[k])):
                results["scores"].append(norm_max_filter_scores[k][i])
                results["idx"].append(clustered[k]["idx"][i])

        len_selected = int(round(prune * len(results["scores"]), 0))
        srts = np.argsort(results["scores"])

        """
        x = []
        for k in scaled_max_filter_scores.keys():
            for i in range(0, len(scaled_max_filter_scores[k])):
                x.append(scaled_max_filter_scores[k][i])

        plt.hist(x, alpha=0.5, bins=np.arange(0, 1.1, 0.1), histtype="stepfilled", edgecolor="black")
        plt.hist(results["scores"], alpha=0.5, bins=np.arange(0, 1.1, 0.1), histtype="stepfilled", edgecolor="black")
        plt.legend(["min-max-scaled", "softmax-scaled"])
        plt.title(f"shrutika pruning scores {scoring_fn} for {n_iters} iters. with alpha of {alpha}")
        plt.show()"""

        mask = torch.zeros_like(m.weight)
        for selected_index in srts[len_selected:]:
            mask[results["idx"][selected_index], :, :, :] += 1

        return mask


def shrutika_prune(module, amount, compute_mask_args):
    mask = shrutika_prune_compute_mask(m=module, prune=amount, **compute_mask_args)
    prune.CustomFromMask(mask).apply(module, "weight", mask)


if __name__ == "__main__":
    path = "../_results/saved_models_cluster/reproduceability-test/Reproducability-Test_2021-03-01_07-57-48.json"
    model = load_model_from_json_path(path)

    # print(model)
    conv_layers = []
    for m in model.down_blocks.modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_layers.append(m)
    for m in model.latent.modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_layers.append(m)
    for m in model.up_blocks.modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_layers.append(m)

    #for layer in conv_layers:
    #    #print(layer.weight)
    #    prune.random_structured(layer, name="weight", amount=0.9, dim=0)
    #    prune.remove(layer, "weight")
    #    #print(layer.weight)
    #    break

    shrutika_prune_args = {
        "n_clusters": 10,
        "n_iters": 10,
        "alpha": 0.8,
        "verbose": 0,
        "scoring_fn": "l1"
    }

    import time
    start_time = time.time()

    shrutika_prune(conv_layers[1], 0.2, shrutika_prune_args)

    print("took", time.time() - start_time, "seconds")

