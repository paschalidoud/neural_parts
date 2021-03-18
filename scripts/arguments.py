def add_dataset_parameters(parser):
    parser.add_argument(
        "--model_tags",
        type=lambda x: x.split(","),
        default=[],
        help="Tags of the models to be used"
    )
    parser.add_argument(
        "--category_tags",
        type=lambda x: x.split(","),
        default=[],
        help="Category tags of the models to be used"
    )
    parser.add_argument(
        "--random_subset",
        type=float,
        default=1.0,
        help="Percentage of dataset to be used"
    )
    parser.add_argument(
        "--val_random_subset",
        type=float,
        default=1.0,
        help="Percentage of dataset to be used for validation"
    )
