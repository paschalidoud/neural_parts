#!/usr/bin/env python
"""Script used for performing a forward pass on a previously trained model and
visualizing the predicted primitives.
"""
import argparse
import os
import sys

import numpy as np
import torch
import trimesh

from simple_3dviz import Mesh, Spherecloud
from simple_3dviz.behaviours import SceneInit
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.keyboard import SnapshotOnKey
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.utils import render
from simple_3dviz.window import show

from arguments import add_dataset_parameters
from utils import load_config, points_on_sphere
from visualization_utils import scene_init, load_ground_truth, \
    get_colors, jet, colormap

from neural_parts.datasets import build_dataset
from neural_parts.models import build_network
from neural_parts.utils import sphere_mesh
from neural_parts.metrics import iou_metric


def main(argv):
    parser = argparse.ArgumentParser(
        description="Do the forward pass and visualize the recovered parts"
    )
    parser.add_argument(
        "dataset_directory",
        help="Path to the directory containing the dataset"
    )
    parser.add_argument(
        "output_directory",
        help="Save the output files in that directory"
    )
    parser.add_argument(
        "--train_test_splits_file",
        default="../config/all.csv",
        help="Path to the train-test splits file"
    )
    parser.add_argument(
        "--config_file",
        default="../config/default.yaml",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--prediction_file",
        default=None,
        help="The path to the predicted primitives"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=200,
        help="Number of frames to be rendered"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="0,0,0,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,1",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-2.0,-2.0,-2.0",
        help="Camera position in the scene"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object"
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Visualize the target mesh"
    )
    parser.add_argument(
        "--no_color",
        action="store_true",
        help="Use the same color for all the primitives"
    )
    parser.add_argument(
        "--show_vertices",
        action="store_true",
        help="Show the vertices as a sphere cloud"
    )
    parser.add_argument(
        "--n_vertices",
        type=int,
        default=10000,
        help="How many vertices to use per part"
    )
    parser.add_argument(
        "--only_part",
        type=int,
        default=None,
        help="Show only a specific part if given"
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
    # Extract the number of primitives
    n_primitives = config["network"]["n_primitives"]

    # Dictionary to keep the predictions used for the evaluation
    predictions = {}

    if args.prediction_file is None:
        # Instantiate a dataset to generate the samples for evaluation
        dataset = build_dataset(
            config,
            args.dataset_directory,
            args.dataset_type,
            args.train_test_splits_file,
            args.model_tags,
            args.category_tags,
            ["train", "val", "test"],
            random_subset=args.random_subset
        )
        assert len(dataset) == 1

        # Build the network architecture to be used for training
        network, _, _ = build_network(config, args.weight_file, device=device)
        network.eval()

        # Create the prediction input
        with torch.no_grad():
            for sample in dataset:
                sample = [s[None] for s in sample]  # make a batch dimension
                X = sample[0].to(device)
                targets = [yi.to(device) for yi in sample[1:]]
                F = network.compute_features(X)
                phi_volume, _ = network.implicit_surface(F, targets[0])
                y_pred, faces = network.points_on_primitives(
                    F, args.n_vertices, random=False, mesh=True,
                    union_surface=False
                )
            predictions["phi_volume"] = phi_volume
            predictions["y_prim"] = y_pred
    else:
        preds = torch.load(args.prediction_file, map_location="cpu")
        y_pred = preds[4]
        faces = preds[5]
        targets = preds[0]
        predictions["phi_volume"] = preds[1]
        predictions["y_prim"] = y_pred

    print("IOU:", iou_metric(predictions, targets))

    # Get the renderables from the deformed vertices and faces
    vertices = y_pred.detach()
    parts = range(n_primitives)
    if args.only_part is not None:
        parts = [args.only_part]
    renderables = [
        Mesh.from_faces(
            vertices[0, :, i],
            faces,
            colors=get_colors(0 if args.no_color else i)
        )
        for i in parts
    ]
    if args.show_vertices:
        renderables = [
            Mesh.from_faces(
                vertices[0, :, i],
                faces,
                colors=colormap(np.linspace(0, 1, vertices.shape[1]))
            )
            for i in parts
        ]

    behaviours = [
        SceneInit(
            scene_init(
                load_ground_truth(dataset) if args.mesh else None,
                args.up_vector,
                args.camera_position,
                args.camera_target,
                args.background
            )
        ),
        LightToCamera(),
    ]
    if args.with_rotating_camera:
        behaviours += [
            CameraTrajectory(
                Circle(
                    args.camera_target,
                    args.camera_position,
                    args.up_vector
                ),
                speed=1/180
            )
        ]
    if args.without_screen:
        path_to_gif = "/{}/{}.gif".format(
            args.output_directory, args.model_tags[0]
        )
        behaviours += [
            SaveFrames(args.save_frames, 1),
            SaveGif(path_to_gif, 1)
        ]
        render(renderables, size=args.window_size, behaviours=behaviours,
               n_frames=args.n_frames)
    else:
        show(renderables, size=args.window_size,
             behaviours=behaviours + [SnapshotOnKey()])

    print("Saving renderables to file")
    for i in range(n_primitives):
        m = trimesh.Trimesh(vertices[0, :, i].detach(), faces)
        m.export(
            os.path.join(args.output_directory, "part_{:03d}.obj".format(i)),
            file_type="obj"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
