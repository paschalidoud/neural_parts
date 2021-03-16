import csv
import numpy as np


class SplitsBuilder(object):
    def __init__(self, train_test_splits_file):
        self._train_test_splits_file = train_test_splits_file
        self._splits = {}

    def train_split(self):
        return self.splits["train"]

    def test_split(self):
        return self.splits["test"]

    def val_split(self):
        return self.splits["val"]

    def _parse_train_test_splits_file(self):
        with open(self._train_test_splits_file, "r") as f:
            data = [row for row in csv.reader(f)]
        return np.array(data)

    def get_splits(self, keep_splits=["train, val"]):
        if not isinstance(keep_splits, list):
            keep_splits = [keep_splits]
        # Return only the split
        s = []
        for ks in keep_splits:
            s.extend(self._parse_split_file()[ks])
        return s


class CSVSplitsBuilder(SplitsBuilder):
    def _parse_split_file(self):
        if not self._splits:
            data = self._parse_train_test_splits_file()
            for s in ["train", "test", "val"]:
                self._splits[s] = [r[0] for r in data if r[1] == s]
        return self._splits


class DynamicFaustSplitsBuilder(SplitsBuilder):
    def _parse_split_file(self):
        if not self._splits:
            data = self._parse_train_test_splits_file()
            header = data[0]
            for s in ["train", "test", "val"]:
                # Only keep the data for the current split
                d = data[data[:, -1] == s]
                tags = [
                    "{}:{}".format(oi, mi)
                    for oi, mi in zip(d[:, 0], d[:, 1])
                ]
                self._splits[s] = tags

        return self._splits


class ShapeNetSplitsBuilder(SplitsBuilder):
    def _parse_split_file(self):
        if not self._splits:
            data = self._parse_train_test_splits_file()
            header = data[0]
            for s in ["train", "test", "val"]:
                # Only keep the data for the current split
                d = data[data[:, -1] == s]
                tags = [
                    "{}:{}".format(oi, mi)
                    for oi, mi in zip(d[:, 1], d[:, 3])
                ]
                self._splits[s] = tags

        return self._splits
