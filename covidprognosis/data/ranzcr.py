"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import os
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset

# https://www.kaggle.com/ihelon/catheter-position-exploratory-data-analysis
class RANZRCDataset(BaseDataset):
    """
    Data loader for RANZRC dataset.

    Args:
        directory: Base directory for data set.
        split: String specifying split.
            options include:
                'all': Include all splits.
                'train': Include training split.
        label_list: String specifying labels to include. Default is 'all',
            which loads all labels.
        transform: A composible transform list to be applied to the data.
    """

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        split: str = "train",
        label_list: Union[str, List[str]] = "all",
        subselect: Optional[str] = None,
        transform: Optional[Callable] = None,
        resplit: bool = False,
        resplit_seed: int = 2019,
        resplit_ratios: List[float] = [0.7, 0.2, 0.1],
    ):
        super().__init__(
            "ranzcr-chest-xrays", directory, split, label_list, subselect, transform
        )

        if label_list == "all":
            self.label_list = self.default_labels()
        else:
            self.label_list = label_list

        self.metadata_keys = [
            "StudyInstanceUID",  # unique ID for each image
            "ETT - Abnormal",  # endotracheal tube placement abnormal
            "ETT - Borderline",  # endotracheal tube placement borderline abnormal
            "ETT - Normal",  # endotracheal tube placement normal
            "NGT - Abnormal",  # nasogastric tube placement abnormal
            "NGT - Borderline",  # nasogastric tube placement borderline abnormal
            "NGT - Incompletely Imaged",  # nasogastric tube placement inconclusive due to imaging
            "NGT - Normal",  # nasogastric tube placement borderline normal
            "CVC - Abnormal",  # central venous catheter placement abnormal
            "CVC - Borderline",  # central venous catheter placement borderline abnormal
            "CVC - Normal",  # central venous catheter placement normal
            "Swan Ganz Catheter Present",  # TODO: ??
            "PatientID",  # unique ID for each patient in the dataset
        ]

        if resplit:
            rg = np.random.default_rng(resplit_seed)

            self.csv_path = self.directory / "train.csv"
            csv = pd.read_csv(self.csv_path)
            patient_list = csv["PatientID"].unique()

            rand_inds = rg.permutation(len(patient_list))

            train_count = int(np.round(resplit_ratios[0] * len(patient_list)))
            val_count = int(np.round(resplit_ratios[1] * len(patient_list)))

            grouped = csv.groupby("Patient ID")

            if self.split == "train":
                patient_list = patient_list[rand_inds[:train_count]]
                self.csv = pd.concat([grouped.get_group(pat) for pat in patient_list])
            elif self.split == "val":
                patient_list = patient_list[
                    rand_inds[train_count : train_count + val_count]
                ]
                self.csv = pd.concat([grouped.get_group(pat) for pat in patient_list])
            elif self.split == "test":
                patient_list = patient_list[rand_inds[train_count + val_count :]]
                self.csv = pd.concat([grouped.get_group(pat) for pat in patient_list])
            else:
                logging.warning(
                    "split {} not recognized for dataset {}, "
                    "not returning samples".format(split, self.__class__.__name__)
                )
        else:
            if self.split == "train":
                self.csv_path = self.directory / "train.csv"
                self.csv = pd.read_csv(self.csv_path)
            elif self.split == "all":
                self.csv_path = self.directory / "train.csv"
                self.csv = pd.read_csv(self.csv_path)
            else:
                logging.warning(
                    "split {} not recognized for dataset {}, "
                    "not returning samples".format(split, self.__class__.__name__)
                )

        self.csv = self.preproc_csv(self.csv, self.subselect)

    @staticmethod
    def default_labels() -> List[str]:
        return [
            "ETT - Abnormal",  # endotracheal tube placement abnormal
            "ETT - Borderline",  # endotracheal tube placement borderline abnormal
            "ETT - Normal",  # endotracheal tube placement normal
            "NGT - Abnormal",  # nasogastric tube placement abnormal
            "NGT - Borderline",  # nasogastric tube placement borderline abnormal
            "NGT - Incompletely Imaged",  # nasogastric tube placement inconclusive due to imaging
            "NGT - Normal",  # nasogastric tube placement borderline normal
            "CVC - Abnormal",  # central venous catheter placement abnormal
            "CVC - Borderline",  # central venous catheter placement borderline abnormal
            "CVC - Normal",  # central venous catheter placement normal
            "Swan Ganz Catheter Present",  # TODO: ??
        ]

    def preproc_csv(self, csv: pd.DataFrame, subselect: Optional[str]) -> pd.DataFrame:
        # TODO: what is this do on other csv
        # if csv is not None:

        #    def format_view(s):
        #        return "frontal" if s in ("AP", "PA") else None

        #    csv["view"] = csv["View Position"].apply(format_view)

        #    if subselect is not None:
        #        csv = csv = csv.query(subselect)

        return csv

    def __len__(self) -> int:
        length = 0
        if self.csv is not None:
            length = len(self.csv)

        return length

    def __getitem__(self, idx: int) -> Dict:
        assert self.csv is not None
        exam = self.csv.iloc[idx]

        filename = self.directory / "train" / f"{exam['StudyInstanceUID']}.jpg"
        image = self.open_image(filename)

        metadata = self.retrieve_metadata(idx, filename, exam)

        # example: exam['Finding Labels'] = 'Pneumonia|Cardiomegaly'
        # goal here is to see if label is a substring of
        # 'Pneumonia|Cardiomegaly' for each label in self.label_list
        labels = [int(exam[label]) for label in self.label_list]
        labels = np.array(labels).astype(np.float)

        sample = {"image": image, "labels": labels, "metadata": metadata}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
