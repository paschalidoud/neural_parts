import numpy as np
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache
from collections import OrderedDict
import os
import pickle
from PIL import Image

from ..external.libmesh import check_mesh_contains
from .mesh import read_mesh_file
from .occupancy_voxelizer import OccupancyGrid


class BaseModel(object):
    """BaseModel class is wrapper for all models, independent of dataset. Every
       model has a unique model_tag, mesh_file and Mesh object. Optionally, it
       can also have a tsdf file.
    """
    def __init__(self, tag, config):
        self._tag = tag
        self._config = config

        # Initialize the contents of this instance to empty so that they can be
        # lazy loaded
        self._gt_mesh = None
        self._occupancy_pairs = None
        self._signed_distances = None
        self._surface_pairs = None
        self._points_on_surface = None
        self._internal_points = None
        self._images = []
        self._image_paths = None
        self._voxel_grid = None

    @property
    def tag(self):
        return self._tag

    @property
    def path_to_mesh_file(self):
        raise NotImplementedError()

    @property
    def path_to_occupancy_pairs(self):
        raise NotImplementedError()

    @property
    def path_to_surface_pairs(self):
        raise NotImplementedError()

    @property
    def path_to_voxel_grid(self):
        raise NotImplementedError()

    @property
    def path_to_sampled_faces(self):
        raise NotImplementedError()

    @property
    def images_dir(self):
        raise NotImplementedError()

    @property
    def groundtruth_mesh(self):
        if self._gt_mesh is None:
            self._gt_mesh = read_mesh_file(
                self.path_to_mesh_file,
                self._config["data"].get("normalize", True)
            )
        return self._gt_mesh

    @groundtruth_mesh.setter
    def groundtruth_mesh(self, mesh):
        if self._gt_mesh is not None:
            raise RuntimeError("Trying to overwrite a mesh")
        self._gt_mesh = mesh

    @property
    def image_paths(self):
        if self._image_paths is None:
            self._image_paths = [
                os.path.join(self.images_dir, p)
                for p in sorted(os.listdir(self.images_dir))
                if p.endswith(".jpg") or p.endswith(".png")
            ]
        return self._image_paths

    def get_image(self, idx):
        return np.array(Image.open(self.image_paths[idx]).convert("RGB"))

    @property
    def random_image(self):
        random_view = self._config["data"].get("random_view", False)
        if random_view:
            idx = np.random.choice(len(self.image_paths))
        else:
            idx = 0
        return self.get_image(idx)

    def get_voxel_grid(self):
        V = self._config["data"]["voxel_grid_shape"]
        if self._voxel_grid is None:
            try:
                self._voxel_grid = np.load(
                    self.path_to_voxel_grid, mmap_mode="r"
                )
            except:
                occupancy_vox = OccupancyGrid(V)
                self._voxel_grid = occupancy_vox.voxelize(self.groundtruth_mesh)
                np.save(self.path_to_voxel_grid, self._voxel_grid)
        return self._voxel_grid

    def sample_faces(self):
        N = self._config["data"]["n_points_on_mesh"]
        try:
            sampled_faces = np.load(self.path_to_sampled_faces, mmap_mode="r")
        except:
            sampled_faces = \
                self.groundtruth_mesh.sample_faces(100000).astype(np.float32)
            np.random.shuffle(sampled_faces)
            np.save(self.path_to_sampled_faces, sampled_faces)
        t = int(np.random.rand()*(100000-N))
        return np.array(sampled_faces[t:t+N]).astype(np.float32)

    def _get_points_in_bbox(self):
        if self._occupancy_pairs is None:
            if not os.path.exists(self.path_to_occupancy_pairs):
                mesh = self.groundtruth_mesh
                points = np.random.rand(100000, 3).astype(np.float32) - 0.5
                labels = check_mesh_contains(
                    mesh.mesh, points
                ).astype(np.float32)[:, None]
                np.savez(
                    self.path_to_occupancy_pairs,
                    points=points,
                    occupancies=labels[:, 0]
                )
                self._occupancy_pairs = (points, labels)
            else:
                self._occupancy_pairs = np.load(
                    self.path_to_occupancy_pairs,
                    allow_pickle=True
                )
                self._occupancy_pairs = (
                    self._occupancy_pairs["points"].astype(np.float32),
                    self._occupancy_pairs["occupancies"].astype(np.float32)[:, None]
                )
            if self._config["data"].get("augment_negative", False):
                N = len(self._occupancy_pairs[0])
                side = (3**(1/3)/2)
                extra_points = np.zeros((0, 3))
                while len(extra_points) < N:
                    points = np.random.rand(N, 3) * 2 * side - side
                    valid_points = (np.abs(points) > 0.5).all(axis=1)
                    extra_points = np.vstack(
                        [extra_points, points[valid_points]]
                    )
                extra_points = extra_points[:N].astype(np.float32)
                extra_labels = np.zeros((N, 1), dtype=np.float32)
                self._occupancy_pairs = (
                    np.vstack([self._occupancy_pairs[0], extra_points]),
                    np.vstack([self._occupancy_pairs[1], extra_labels])
                )

        return self._occupancy_pairs

    def _get_signed_distances_in_bbox(self):
        if self._signed_distances is None:
            self._signed_distances = np.load(
                self.path_to_signed_distances,
                allow_pickle=True
            )

        return (
            self._signed_distances["points"].astype(np.float32),
            self._signed_distances["distances"].astype(np.float32).reshape(-1, 1)
        )

    def _get_points_on_surface(self):
        if self._surface_pairs is None:
            self._surface_pairs = np.load(
                self.path_to_surface_pairs,
                allow_pickle=True
            )

        return (
            self._surface_pairs["points"].astype(np.float32),
            self._surface_pairs["occupancies"].astype(np.float32).reshape(-1, 1)
        )

    def _sample_points_on_surface(self, N):
        if N <= 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32)
            )

        points, labels = self._get_points_on_surface()
        weights = np.ones((N, 1), dtype=np.float32)
        idxs = np.random.choice(len(points), N)

        return (
            points[idxs],
            labels[idxs],
            weights
        )

    def _sample_points_uniform(self, N):
        if N <= 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32)
            )

        points, labels = self._get_points_in_bbox()
        weights = np.ones((N, 1), dtype=np.float32)
        if N >= (len(points) // 2):
            t = np.random.randint(0, len(points)-N+1)
            idxs = slice(t, t+N)
        else:
            idxs = np.random.choice(len(points), N)

        return (
            points[idxs],
            labels[idxs],
            weights
        )

    def _sample_points_weighted(self, N, positive_weight):
        points, labels = self._get_points_in_bbox()
        p = positive_weight * labels + (1-labels)
        p /= p.sum()
        idxs = np.random.choice(len(points), N, p=p.ravel())

        return (
            points[idxs],
            labels[idxs],
            np.float32(1.0/len(points)) / p[idxs]
        )

    def _sample_points_equal(self, N):
        points, labels = self._get_points_in_bbox()
        n_positive = labels.sum()
        n_negative = len(labels) - n_positive
        p = n_negative * labels + n_positive * (1-labels)
        p /= p.sum()
        idxs = np.random.choice(len(points), N, p=p.ravel())

        return (
            points[idxs],
            labels[idxs],
            np.float32(1.0/len(points)) / p[idxs]
        )

    def _sample_biased(self, p, l, w):
        return p, l, np.ones_like(w)

    def sample_internal_points(self):
        """Sample points uniformly inside the object."""
        data_config = self._config["data"]
        N = data_config.get("n_points_in_mesh", 1000)
        points, labels = self._get_points_in_bbox()
        indices = np.nonzero(labels[:, 0])[0]
        np.random.shuffle(indices)
        return (
            points[indices[:N]],
        )

    def sample_points_with_signed_distances(self):
        """Sample points uniformly in space and return their signed distance.
        """
        data_config = self._config["data"]
        N = data_config.get("n_points_in_mesh", 1000)
        points, signed_distances = self._get_signed_distances_in_bbox()
        weights = np.ones((N, 1), dtype=np.float32)
        idxs = np.random.choice(len(points), N)

        return (
            points[idxs],
            signed_distances[idxs],
            weights
        )

    def sample_points(self):
        """Sample points in space and return whether they are in the object or
        not."""
        def compose(f, g):
            def inner(*args):
                return f(*g(*args))
            return inner

        def stack_points(*pts):
            return (
                np.vstack([p[0] for p in pts]),
                np.vstack([p[1] for p in pts]),
                np.vstack([p[2] for p in pts])
            )

        data_config = self._config["data"]
        N = data_config.get("n_points_in_mesh", 1000)
        Ns = data_config.get("n_points_on_mesh_surface", 0)
        equal = data_config.get("equal", False)
        positive_weight = data_config.get("positive_weight", 0)
        biased = data_config.get("biased", False)

        # Sampler key is (equal, positive_weight>0, biased)
        samplers = {
            (False, False, False): self._sample_points_uniform,
            (False, False, True): self._sample_points_uniform,
            (True, False, False): self._sample_points_equal,
            (True, True, False): self._sample_points_equal,
            (False, True, False): lambda N: self._sample_points_weighted(
                N,
                positive_weight
            ),
            (True, False, True): compose(
                self._sample_biased,
                self._sample_points_equal
            ),
            (True, True, True): compose(
                self._sample_biased,
                self._sample_points_equal
            ),
            (False, True, True): compose(
                self._sample_biased,
                lambda N: self._sample_points_weighted(N, positive_weight)
            )
        }

        surface_sampler = self._sample_points_on_surface

        return stack_points(
            samplers[equal, positive_weight > 0, biased](N),
            surface_sampler(Ns)
        )


class ModelCollection(object):
    def __len__(self):
        raise NotImplementedError()

    def _get_model(self, i):
        raise NotImplementedError()

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()
        return self._get_model(i)


class ModelSubset(ModelCollection):
    def __init__(self, collection, subset):
        self._collection = collection
        self._subset = subset

    def __len__(self):
        return len(self._subset)

    def _get_sample(self, i):
        return self._collection[self._subset[i]]

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()
        return self._get_sample(i)


class TagSubset(ModelSubset):
    def __init__(self, collection, tags):
        tags = set(tags)
        subset = [i for (i, m) in enumerate(collection) if m.tag in tags]
        super(TagSubset, self).__init__(collection, subset)


class RandomSubset(ModelSubset):
    def __init__(self, collection, percentage):
        N = len(collection)
        subset = np.random.choice(N, int(N*percentage)).tolist()
        super(RandomSubset, self).__init__(collection, subset)


class CategorySubset(ModelSubset):
    def __init__(self, collection, category_tags):
        category_tags = set(category_tags)
        subset = [
            i
            for (i, m) in enumerate(collection)
            if m.category in category_tags
        ]
        super(CategorySubset, self).__init__(collection, subset)


class DynamicFaust(ModelCollection):
    class Model(BaseModel):
        def __init__(self, base_dir, tag, config):
            super(DynamicFaust.Model, self).__init__(tag, config)
            self._base_dir = base_dir
            self._category, self._sequence = tag.split(":")

            config = config["data"]
            self._mesh_folder = config["mesh_folder"]
            self._points_folder = config["points_folder"]
            self._surface_points_folder = config["surface_points_folder"]
            self._renderings_folder = config["renderings_folder"]

        @property
        def category(self):
            return self._category

        @property
        def path_to_mesh_file(self):
            return os.path.join(self._base_dir, self._category,
                                self._mesh_folder, self._sequence+".obj")

        @property
        def path_to_occupancy_pairs(self):
            if self._points_folder.endswith("_2"):
                seq = "{}.npz".format(self._sequence)
            else:
                seq = "000{}.npz".format(self._sequence)
            return os.path.join(self._base_dir, self._category,
                                self._points_folder, seq)

        @property
        def path_to_surface_pairs(self):
            return os.path.join(self._base_dir, self._category,
                                self._surface_points_folder,
                                "{}.npz".format(self._sequence))

        @property
        def path_to_sampled_faces(self):
            return os.path.join(self._base_dir, self._category,
                                self._mesh_folder,
                                "{}_sampled_faces.npy".format(self._sequence))

        @property
        def image_paths(self):
            return [os.path.join(self._base_dir, self._category,
                                 self._renderings_folder,
                                 "{}.png".format(self._sequence))]

    def __init__(self, base_dir, config):
        self._base_dir = base_dir
        self._config = config
        self._paths = sorted([
            d
            for d in os.listdir(self._base_dir)
            if os.path.isdir(os.path.join(self._base_dir, d))
        ])
        mesh_folder = config["data"].get("mesh_folder", "mesh_seq_2")

        # Note that we filter out the first 20 meshes from the sequence to
        # "discard" the neutral pose that is used for calibration purposes.
        self._tags = sorted([
            "{}:{}".format(d, l[:-4]) for d in self._paths
            for l in sorted(os.listdir(os.path.join(self._base_dir, d, mesh_folder)))[20:]
            if l.endswith(".obj")
        ])

        print("Found {} Dynamic Faust models".format(len(self)))

    def __len__(self):
        return len(self._tags)

    def _get_model(self, i):
        return self.Model(self._base_dir, self._tags[i], self._config)


class FreiHand(ModelCollection):
    class Model(BaseModel):
        def __init__(self, base_dir, tag, config):
            super().__init__(tag, config)
            self._base_dir = base_dir

        @property
        def category(self):
            return ""

        @property
        def path_to_mesh_file(self):
            return os.path.join(self._base_dir, self.tag + ".obj")

        @property
        def path_to_occupancy_pairs(self):
            return os.path.join(self._base_dir, self.tag + ".npz")

        @property
        def path_to_surface_pairs(self):
            return os.path.join(self._base_dir, self.tag + "_surface.npz")

        @property
        def path_to_sampled_faces(self):
            return os.path.join(self._base_dir, self.tag + ".npy")

        @property
        def path_to_voxel_grid(self):
            return os.path.join(self._base_dir, self.tag + "_voxel_grid.npy")

        @property
        def image_paths(self):
            return [os.path.join(self._base_dir, self.tag + ".png")]

    def __init__(self, base_dir, config):
        self._base_dir = base_dir
        self._config = config
        self._tags = sorted([
            f[:-4]
            for f in os.listdir(self._base_dir)
            if f.endswith(".png")
        ])

    def __len__(self):
        return len(self._tags)

    def _get_model(self, i):
        return self.Model(self._base_dir, self._tags[i], self._config)


class TurbosquidAnimal(ModelCollection):
    class Model(BaseModel):
        def __init__(self, base_dir, tag, config):
            super().__init__(tag, config)
            self._base_dir = base_dir
            self._tag = tag

        @property
        def path_to_mesh_file(self):
            return os.path.join(
                self._base_dir, self._tag, "model_watertight.off"
            )

        @property
        def path_to_occupancy_pairs(self):
            return os.path.join(self._base_dir, self._tag, "points.npz")

        @property
        def path_to_surface_pairs(self):
            return os.path.join(self._base_dir, self.tag, "points_surface.npz")

        @property
        def path_to_sampled_faces(self):
            return os.path.join(self._base_dir, self.tag, "sampled_faces.npy")

        @property
        def path_to_voxel_grid(self):
            return os.path.join(self._base_dir, self.tag, "voxel_grid.npy")

        @property
        def image_paths(self):
            return [os.path.join(self._base_dir, self.tag, "image_00000.png")]

    def __init__(self, base_dir, config):
        self._base_dir = base_dir
        self._config = config

        self._tags = sorted([
            fi for fi in os.listdir(self._base_dir)
            if os.path.isdir(os.path.join(self._base_dir, fi))
        ])

        print("Found {} TurbosquidAnimal models".format(len(self)))

    def __len__(self):
        return len(self._tags)

    def _get_model(self, i):
        return self.Model(self._base_dir, self._tags[i], self._config)


class MultiModelsShapeNetV1(ModelCollection):
    class Model(BaseModel):
        def __init__(self, base_dir, tag, config):
            super(MultiModelsShapeNetV1.Model, self).__init__(tag, config)
            self._base_dir = base_dir
            self._category, self._model = tag.split(":")
            self._mesh_file = config["data"].get(
                "mesh_file", "model_watertight.off"
            )

        @property
        def category(self):
            return self._category

        @property
        def path_to_mesh_file(self):
            return os.path.join(self._base_dir, self._category, self._model,
                                self._mesh_file)

        @property
        def path_to_occupancy_pairs(self):
            return os.path.join(self._base_dir, self._category, self._model,
                                "points.npz")

        @property
        def path_to_surface_pairs(self):
            return os.path.join(self._base_dir, self._category, self._model,
                                "points_surface.npz")

        @property
        def path_to_signed_distances(self):
            return os.path.join(self._base_dir, self._category, self._model,
                                "signed_distances.npz")

        @property
        def path_to_voxel_grid(self):
            return os.path.join(self._base_dir, self._category, self._model,
                                "voxel_grid.npy")

        @property
        def path_to_sampled_faces(self):
            return os.path.join(self._base_dir, self._category, self._model,
                                "sampled_faces.npy")

        @property
        def images_dir(self):
            return os.path.join(self._base_dir, self._category, self._model,
                                "img_choy2016")

    def __init__(self, base_dir, config):
        self._base_dir = base_dir
        self._config = config
        self._models = sorted([
            d
            for d in os.listdir(self._base_dir)
            if os.path.isdir(os.path.join(self._base_dir, d))
        ])

        self._tags = sorted([
            "{}:{}".format(d, l) for d in self._models
            for l in os.listdir(os.path.join(self._base_dir, d))
            if os.path.isdir(os.path.join(self._base_dir, d, l))
        ])

        print("Found {} MultiModelsShapeNetV1 models".format(len(self)))

    def __len__(self):
        return len(self._tags)

    def _get_model(self, i):
        return self.Model(self._base_dir, self._tags[i], self._config)


class MeshCache(ModelCollection):
    """Cache the meshes from a collection and give them to the model before
    returning it."""
    def __init__(self, collection):
        self._collection = collection
        self._meshes = [None]*len(collection)

    def __len__(self):
        return len(self._collection)

    def _get_model(self, i):
        model = self._collection._get_model(i)
        if self._meshes[i] is not None:
            model.groundtruth_mesh = self._meshes[i]
        else:
            self._meshes[i] = model.groundtruth_mesh

        return model


class LRUCache(ModelCollection):
    def __init__(self, collection, n=2000):
        self._collection = collection
        self._cache = OrderedDict([])
        self._maxsize = n

    def __len__(self):
        return len(self._collection)

    def _get_model(self, i):
        m = None
        if i in self._cache:
            m = self._cache.pop(i)
        else:
            m = self._collection._get_model(i)
            if len(self._cache) > self._maxsize:
                self._cache.popitem()
        self._cache[i] = m
        return m


def model_factory(dataset_type):
    return {
        "dynamic_faust": DynamicFaust,
        "shapenet_v1": MultiModelsShapeNetV1,
        "freihand": FreiHand,
        "turbosquid_animal": TurbosquidAnimal
    }[dataset_type]


class ModelCollectionBuilder(object):
    def __init__(self, config):
        self._dataset_class = None
        self._cache_meshes = False
        self._lru_cache = 0
        self._tags = []
        self._category_tags = []
        self._percentage = 1.0
        self._config = config
        self._train_test_splits = []

    def with_dataset(self, dataset_type):
        self._dataset_class = model_factory(dataset_type)
        return self

    def with_cache_meshes(self):
        self._cache_meshes = True
        return self

    def without_cache_meshes(self):
        self._cache_meshes = False
        return self

    def lru_cache(self, n=2000):
        self._lru_cache = n
        return self

    def filter_tags(self, tags):
        self._tags = tags
        return self

    def filter_category_tags(self, tags):
        self._category_tags = tags
        return self

    def filter_train_test(self, split_builder, keep_splits):
        self._train_test_splits = split_builder.get_splits(keep_splits)
        return self

    def random_subset(self, percentage):
        self._percentage = percentage
        return self

    def build(self, base_dir):
        dataset = self._dataset_class(base_dir, self._config)
        if self._cache_meshes:
            dataset = MeshCache(dataset)
        if self._lru_cache > 0:
            dataset = LRUCache(dataset, self._lru_cache)
        if len(self._train_test_splits) > 0:
            prev_len = len(dataset)
            dataset = TagSubset(dataset, self._train_test_splits)
            print("Keep {}/{} based on train/test splits".format(
                len(dataset), prev_len)
            )
        if len(self._tags) > 0:
            prev_len = len(dataset)
            dataset = TagSubset(dataset, self._tags)
            print("Keep {}/{} based on tags".format(len(dataset), prev_len))
        if len(self._category_tags) > 0:
            prev_len = len(dataset)
            dataset = CategorySubset(dataset, self._category_tags)
            print("Keep {}/{} based on category tags".format(
                len(dataset), prev_len)
            )
        if self._percentage < 1.0:
            dataset = RandomSubset(dataset, self._percentage)

        return dataset
