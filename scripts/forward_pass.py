"""Script used to do the forward pass on a previously trained network."""
import argparse
import os
import sys

import torch
import trimesh

from arguments import add_dataset_parameters
from utils import load_config, points_on_sphere

from neural_parts.datasets import build_dataloader
from neural_parts.models import build_network
from neural_parts.metrics import iou_metric


def main(argv):
    parser = argparse.ArgumentParser(
        description="Do the forward pass and estimate a set of primitives"
    )
    parser.add_argument(
        "config_file",
        default="../config/default.yaml",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        help="Save the output files in that directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )

    add_dataset_parameters(parser)
    args = parser.parse_args(argv)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)
    # Build the network architecture to be used for training
    network, _, _ = build_network(config, args.weight_file, device=device)
    network.eval()

    # Instantiate a dataloader to generate the samples for evaluation
    dataloader = build_dataloader(
        config,
        args.model_tags,
        args.category_tags,
        ["train", "val", "test"],
        batch_size=config["validation"]["batch_size"],
        n_processes=0,
        random_subset=args.random_subset
    )

    # Create the prediction input
    with torch.no_grad():
        for sample in dataloader:
            X = sample[0].to(device)
            targets = [yi.to(device) for yi in sample[1:]]
            F = network.compute_features(X)
            phi_volume, _ = network.implicit_surface(F, targets[0])
            y_pred, faces = network.points_on_primitives(
                F, 5000, random=False, mesh=True, union_surface=False
            )
            predictions = {}
            predictions["phi_volume"] = phi_volume
            predictions["y_prim"] = y_pred
            print("IOU:", iou_metric(predictions, targets))
            torch.save(
                [targets, phi_volume, None, None, y_pred, faces],
                os.path.join(args.output_directory, "prediction.pt")
            )
            n_primitives = config["network"]["n_primitives"]
            print(
                "Saving an obj file per primitive in {}".format(
                    args.output_directory
                )
            )
            for i in range(n_primitives):
                vertices = y_pred.detach()
                m = trimesh.Trimesh(vertices[0, :, i].detach(), faces)
                m.export(
                    os.path.join(
                        args.output_directory, "part_{:03d}.obj".format(i)
                    ),
                    file_type="obj"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
