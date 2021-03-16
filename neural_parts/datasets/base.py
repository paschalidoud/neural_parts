from .dataset import dataset_factory
from .model_collections import ModelCollectionBuilder
from .parse_splits import ShapeNetSplitsBuilder, DynamicFaustSplitsBuilder, \
    CSVSplitsBuilder

from torch.utils.data import DataLoader


def splits_factory(dataset_type):
    return {
        "dynamic_faust": DynamicFaustSplitsBuilder,
        "shapenet_v1": ShapeNetSplitsBuilder,
        "freihand": CSVSplitsBuilder,
        "turbosquid_animal": CSVSplitsBuilder
    }[dataset_type]


def build_dataset(
    config,
    dataset_directory,
    dataset_type,
    train_test_splits_file,
    model_tags,
    category_tags,
    keep_splits,
    random_subset=1.0,
    cache_size=0
):
    # Create a dataset instance to generate the samples for training
    dataset = dataset_factory(
        config["data"]["dataset_factory"],
        (ModelCollectionBuilder(config)
            .with_dataset(dataset_type)
            .filter_train_test(
                splits_factory(dataset_type)(train_test_splits_file),
                keep_splits
             )
            .filter_category_tags(category_tags)
            .filter_tags(model_tags)
            .random_subset(random_subset)
            .build(dataset_directory)),
    )
    return dataset


def build_dataloader(
    config,
    dataset_directory,
    dataset_type,
    train_test_splits_file,
    model_tags,
    category_tags,
    split,
    batch_size,
    n_processes,
    random_subset=1.0,
    cache_size=0,
    shuffle=True
):
    # Create a dataset instance to generate the samples for training
    dataset = build_dataset(
        config,
        dataset_directory,
        dataset_type,
        train_test_splits_file,
        model_tags,
        category_tags,
        split,
        random_subset=random_subset,
        cache_size=cache_size,
    )
    print("Dataset has {} elements".format(len(dataset)))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_processes,
        shuffle=shuffle
    )

    return dataloader
