import torch
import argparse
import matplotlib.pyplot as plt

from main import Base
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from src.dataset import img_transform
from src.dataset import SynthesisGlueDataset


def main(args):
    model = Base(n_channels=1, n_classes=2, bilinear=True)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to("cuda")
    model.eval()

    for split in ["valid", "test"]:
        dataset = SynthesisGlueDataset(
            root_dir="./dataset",
            split=split,
            file_match_pairs=[
                ("*/*", (1, 2), "bmp"),
                ("*/*", (0, 1), "jpg")
            ]
        )

        acc = []
        f1 = []
        jacc = []
        with torch.inference_mode():
            for data in tqdm(dataset):
                img = data["img"]
                mask_gt = data["mask"]
                result = model.evaluate(
                    img.float().unsqueeze(0).to("cuda"),
                    mask_gt.long().unsqueeze(0).to("cuda")
                )
                mask = (result["features"].softmax(dim=1)[:, 1] > 0.5).cpu()[0]
                mask = mask.to(torch.uint8)
                mask_gt = mask_gt.flatten().tolist()
                mask = mask.flatten().tolist()
                acc.append(
                    accuracy_score(
                        mask_gt,
                        mask
                    )
                )
                f1.append(
                    f1_score(
                        mask_gt,
                        mask
                    )
                )
                jacc.append(
                    jaccard_score(
                        mask_gt,
                        mask
                    )
                )
        print(split, "acc:", sum(acc) / len(acc))
        print(split, "F1:", sum(f1) / len(f1))
        print(split, "IoU:", sum(jacc) / len(jacc))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
