import os
import torch
import torchvision
import joblib as jbl
import pandas as pd
from torchvision.models import *
from tqdm import tqdm

from zarth_utils.config import Config


def load_model(model_name, weight_name):
    model = eval(model_name)
    weights = eval(weight_name)
    model = model(weights=weights).eval()
    preprocess = weights.transforms()
    return model, preprocess


def main():
    config = Config(
        default_config_dict={
            "model_name": "vit_h_14",
            "weight_name": "ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1",
        },
        use_argparse=True,
    )

    dir2save = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "%s--%s" % (config["model_name"], config["weight_name"]),
    )
    os.makedirs(dir2save, exist_ok=True)
    if os.path.exists(os.path.join(dir2save, "meta_info.pkl")):
        print("Already exists, skip")
        return

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model, preprocess = load_model(config["model_name"], config["weight_name"])
    model = model.to(device)

    dataset = torchvision.datasets.ImageNet(
        root=os.path.dirname(os.path.abspath(__file__)),
        split="val",
        transform=preprocess,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=2
    )

    all_prob, all_pred, all_target = [], [], []
    for i, (batch, target) in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            batch = batch.to(device)
            prob = model(batch).softmax(dim=1)
            pred = prob.argmax(dim=1)
            all_prob.append(prob.detach().cpu())
            all_pred.append(pred.detach().cpu())
            all_target.append(target.detach().cpu())
    all_prob = torch.cat(all_prob, dim=0).numpy()
    all_pred = torch.cat(all_pred, dim=0).numpy()
    all_target = torch.cat(all_target, dim=0).numpy()

    jbl.dump(all_prob, os.path.join(dir2save, "prob.pkl"))
    jbl.dump(all_pred, os.path.join(dir2save, "pred.pkl"))
    jbl.dump(all_target, os.path.join(dir2save, "target.pkl"))
    pd.DataFrame({"pred": all_pred, "target": all_target}).to_csv(
        os.path.join(dir2save, "pred_target.tsv"), sep="\t", index=False
    )

    meta_info = {}
    correct = all_pred == all_target
    meta_info["acc"] = correct.mean()
    for i in range(1000):
        subset = all_target == i
        correct[subset].mean()
        meta_info["acc_%d" % i] = correct[subset].mean()
    jbl.dump(meta_info, os.path.join(dir2save, "meta_info.pkl"))


if __name__ == "__main__":
    main()
