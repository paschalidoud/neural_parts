import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.transforms import Normalize as TorchNormalize


class BaseDataset(Dataset):
    """Dataset is a wrapper for all datasets we have
    """
    def __init__(self, dataset_object, transform=None):
        """
        Arguments:
        ---------
            dataset_object: a dataset object that can be an object of type
                            model_collections.ModelCollection
            transform: Callable that applies a transform to a sample
        """
        self._dataset_object = dataset_object
        self._transform = transform

    def __len__(self):
        return len(self._dataset_object)

    def __getitem__(self, idx):
        datapoint = self._get_item_inner(idx)
        if self._transform:
            datapoint = self._transform(datapoint)

        return datapoint

    def get_random_datapoint(self):
        return self.__getitem__(np.random.choice(len(self)))

    def _get_item_inner(self, idx):
        raise NotImplementedError()

    @property
    def shapes(self):
        raise NotImplementedError()

    @property
    def internal_dataset_object(self):
        return self._dataset_object


class VoxelInput(BaseDataset):
    """Get a voxel-based representation of a mesh."""
    def _get_item_inner(self, idx):
        m = self._dataset_object[idx]
        return m.get_voxel_grid()

    @property
    def shapes(self):
        raise NotImplementedError()


class ImageInput(BaseDataset):
    """Get a random image of a mesh."""
    def __init__(self, dataset_object, transform=None):
        super(ImageInput, self).__init__(dataset_object, transform)
        self._image_size = None

    @property
    def image_size(self):
        if self._image_size is None:
            self._image_size = self._dataset_object[0].get_image(0).shape
            self._image_size = (self._image_size[2],) + self._image_size[:2]
        return self._image_size

    def _get_item_inner(self, idx):
        img = self._dataset_object[idx].random_image
        return np.transpose(img, (2, 0, 1))

    @property
    def shapes(self):
        return [self.image_size]


class PointsOnMesh(BaseDataset):
    """Get random points on the surface of a mesh."""
    def _get_item_inner(self, idx):
        return self._dataset_object[idx].sample_faces()

    @property
    def shapes(self):
        return [(self._n_points_from_mesh, 6)]


class PointsAndLabels(BaseDataset):
    """Get random points in the bbox and label them inside or outside."""
    def _get_item_inner(self, idx):
        return self._dataset_object[idx].sample_points()

    @property
    def shapes(self):
        return [
            (self._n_points_from_mesh, 3),
            (self._n_points_from_mesh, 1),
            (self._n_points_from_mesh, 1)
        ]


class PointsAndSignedDistances(BaseDataset):
    """Get random points in the bbox and their signed distances."""
    def _get_item_inner(self, idx):
        return self._dataset_object[idx].sample_points_with_signed_distances()

    @property
    def shapes(self):
        return [
            (self._n_points_from_mesh, 3),
            (self._n_points_from_mesh, 1),
            (self._n_points_from_mesh, 1)
        ]


class InternalPoints(BaseDataset):
    """Get points that are internal to the objects."""
    def _get_item_inner(self, idx):
        return self._dataset_object[idx].sample_internal_points()

    @property
    def shapes(self):
        raise NotImplementedError()


class DatasetCollection(BaseDataset):
    """Implement a pytorch Dataset with a list of BaseDataset objects."""
    def __init__(self, *datasets):
        super(DatasetCollection, self).__init__(
            datasets[0]._dataset_object,
            None
        )
        self._datasets = datasets

    def _get_item_inner(self, idx):
        def flatten(x):
            if not isinstance(x, (tuple, list)):
                return [x]
            return [
                xij
                for xi in x
                for xij in flatten(xi)
            ]

        return flatten([d[idx] for d in self._datasets])

    @property
    def shapes(self):
        return sum(
            (d.shapes for d in self._datasets),
            []  # initializer
        )


class DatasetWithTags(BaseDataset):
    """Implement a Dataset with tags."""
    def __init__(self, dataset):
        super(DatasetWithTags, self).__init__(
            dataset._dataset_object, None
        )
        self._dataset = dataset

    def _get_item_inner(self, idx):
        return self._dataset[idx] + [self._dataset._dataset_object[idx].tag]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, x):
        if isinstance(x, (tuple, list)):
            return [torch.from_numpy(xi) for xi in x]
        return torch.from_numpy(x)


class Normalize(object):
    """Normalize image based based on ImageNet."""
    def __call__(self, x):
        X = x.float()

        normalize = TorchNormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        X = X.float() / 255.0
        return normalize(X)


def dataset_factory(name, dataset_object):
    to_tensor = ToTensor()
    norm = Normalize()
    on_surface = PointsOnMesh(dataset_object, transform=to_tensor)
    in_bbox = PointsAndLabels(dataset_object, transform=to_tensor)
    signed_distanes = PointsAndSignedDistances(
        dataset_object, transform=to_tensor
    )
    image_input = ImageInput(
        dataset_object, transform=Compose([to_tensor, norm])
    )
    internal_pts = InternalPoints(dataset_object, transform=to_tensor)
    voxelized = VoxelInput(dataset_object, transform=to_tensor)

    return {
        "ImageDataset": DatasetCollection(image_input, on_surface),
        "ImageDatasetWithOccupancyLabels": DatasetCollection(
            image_input,
            in_bbox
        ),
        "ImageDatasetForChamferAndIOU": DatasetCollection(
            image_input,
            in_bbox,
            on_surface
        ),
        "ImageDatasetWithSignedDistances": DatasetCollection(
            image_input,
            signed_distanes
        ),
        "ImageDatasetForChamferAndSignedDistances": DatasetCollection(
            image_input,
            signed_distanes,
            on_surface
        ),
        "ImageDatasetWithInternalPoints": DatasetCollection(
            image_input,
            internal_pts
        ),
        "VoxelDatasetWithOccupancyLabels": DatasetCollection(
            voxelized,
            in_bbox
        ),
        "VoxelDatasetForChamferAndIOU": DatasetCollection(
            voxelized,
            in_bbox,
            on_surface
        ),
        "ChamferAndIOU": DatasetCollection(
            on_surface,
            in_bbox,
            on_surface
        )
    }[name]
