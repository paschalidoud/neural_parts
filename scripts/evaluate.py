"""Script used to evaluate the performance a previously trained network."""
import argparse
import os
import sys

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from arguments import add_dataset_parameters
from utils import load_config, sphere_mesh

from neural_parts.datasets import build_dataset
from neural_parts.models import build_network
from neural_parts.utils import sphere_mesh
from neural_parts.metrics import iou_metric
from neural_parts.metrics.evaluate_mesh import MeshEvaluator


class DirLock(object):
    def __init__(self, dirpath):
        self._dirpath = dirpath
        self._acquired = False

    @property
    def is_acquired(self):
        return self._acquired

    def acquire(self):
        if self._acquired:
            return
        try:
            os.mkdir(self._dirpath)
            self._acquired = True
        except FileExistsError:
            pass

    def release(self):
        if not self._acquired:
            return
        try:
            os.rmdir(self._dirpath)
            self._acquired = False
        except FileNotFoundError:
            self._acquired = False
        except OSError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


@torch.no_grad()
def sample_points(network, F, N=10000):
    assert len(F) == 1

    # First create meshes for each primitive
    y_pred, faces = network.points_on_primitives(
        F, 1000, mesh=True, random=False, union_surface=False
    )
    meshes = [
        trimesh.Trimesh(y_pred[0, :, i], faces)
        for i in range(y_pred.shape[2])
    ]

    # Compute their areas to sample points proportionately
    areas = np.array([m.area for m in meshes])
    areas /= areas.sum()

    # Initialize an empty point cloud to add valid surface points to
    P = np.empty((0, 3))

    # Actually sample on the surface of the primitives
    S = 1000
    while len(P) < N:
        n_points = np.random.multinomial(S, areas)
        for i, (m, n) in enumerate(zip(meshes, n_points)):
            points, _ = trimesh.sample.sample_surface(m, n)
            if len(points) == 0:
                continue
            phi_points, _ = network.implicit_surface(
                F,
                torch.from_numpy(points[None]).float()
            )
            phi_points[0, :, i] += 1e5
            internal = (phi_points[0, :] < 0).any(dim=-1).numpy()
            P = np.vstack([P, points[~internal]])
    P = P[np.random.choice(len(P), N, replace=False)]

    return torch.from_numpy(P).float()[None]


def main(argv):
    parser = argparse.ArgumentParser(
        description="Evaluate a reconstructed mesh"
    )

    parser.add_argument(
        "config_file",
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
    dataset = build_dataset(
        config,
        args.model_tags,
        args.category_tags,
        config["validation"]["splits"],
        random_subset=args.random_subset
    )

    # Create the prediction input
    n_primitives = config["network"]["n_primitives"]
    vertices, faces = sphere_mesh(10000, n_primitives)

    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset)):
            sample = [s[None] for s in sample]  # make a batch dimension
            X = sample[0].to(device)
            targets = [yi.to(device) for yi in sample[1:]]

            # Create the output  path
            tag = dataset.internal_dataset_object[i].tag
            if ":" in tag:
                assert len(tag.split(":")) == 2
                category_tag = tag.split(":")[0]
                model_tag = tag.split(":")[1]
                path_to_file = os.path.join(
                    args.output_directory,
                    "{}_{}.npz".format(category_tag, model_tag)
                )
            else:
                path_to_file = os.path.join(
                    args.output_directory,
                    "{}.npz".format(tag)
                )
            stats_per_tag = dict()

            # Make sure we are the only ones creating this file
            with DirLock(path_to_file + ".lock") as lock:
                if not lock.is_acquired:
                    continue
                if os.path.exists(path_to_file):
                    continue

                F = network.compute_features(X)
                phi_volume, _ = network.implicit_surface(F, targets[0])
                points = sample_points(network, F, 10000)
                pointcloud_gt = targets[-1][:, :, :3]
                normals_gt = targets[-1][:, :, 3:]

                pcl_metrics = MeshEvaluator().eval_pointcloud(
                    points[0], pointcloud_gt[0]
                )
                predictions = {}
                predictions["phi_volume"] = phi_volume
                predictions["y_prim"] = points
                iou = iou_metric(predictions, targets)
                stats_per_tag[tag] = {
                    "iou": iou,
                    "accuracy": pcl_metrics["accuracy"],
                    "normals_completeness": pcl_metrics["normals_completeness"],
                    "normals_accuracy": pcl_metrics["normals_accuracy"],
                    "normals": pcl_metrics["normals"],
                    "completeness2": pcl_metrics["completeness2"],
                    "accuracy2": pcl_metrics["accuracy2"],
                    "ch_l2": pcl_metrics["chamfer_L2"],
                    "ch_l1": pcl_metrics["chamfer_L1"]
                }
                np.savez(path_to_file, stats=stats_per_tag)


if __name__ == "__main__":
    main(sys.argv[1:])
