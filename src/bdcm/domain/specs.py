from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class SCMExperimentSpec:
    """Per-SCM metadata: graph size, intervention targets, and NN head layout."""

    name_of_folder: str
    d: int
    ind_cause: int
    ind_result: int
    array_titles: np.ndarray
    array_index_for_epsilon: np.ndarray
    array_num_input_for_nn: np.ndarray
    parent: List[List[int]]
    dag_parents_note: Optional[str] = None


SCM1_SPEC = SCMExperimentSpec(
    name_of_folder="SCM1",
    d=5,
    ind_cause=1,
    ind_result=4,
    array_titles=np.array(["X_5 (DCM)", "X_5 (BDCM)"], dtype=object),
    array_index_for_epsilon=np.array([4, 4]),
    array_num_input_for_nn=np.array([3, 4]),
    parent=[[1], [1, 3]],
    dag_parents_note="[[], [1], [1], [3], [2, 4]]",
)

SCM2_SPEC = SCMExperimentSpec(
    name_of_folder="SCM2",
    d=6,
    ind_cause=3,
    ind_result=5,
    array_titles=np.array(["X_5", "X_6 (DCM)", "X_6 (BDCM)"], dtype=object),
    array_index_for_epsilon=np.array([4, 5, 5]),
    array_num_input_for_nn=np.array([3, 4, 4]),
    parent=[[2], [3, 4], [2, 3]],
    dag_parents_note="[[], [1], [2], [3], [3], [2, 4, 5]]",
)

SCM3_SPEC = SCMExperimentSpec(
    name_of_folder="SCM3",
    d=7,
    ind_cause=5,
    ind_result=6,
    array_titles=np.array(["X_5", "X_7 (DCM)", "X_7 (BDCM)"], dtype=object),
    array_index_for_epsilon=np.array([4, 6, 6]),
    array_num_input_for_nn=np.array([3, 4, 6]),
    parent=[[2], [4, 5], [1, 2, 3, 5]],
    dag_parents_note="[[], [1], [1], [1], [3], [2, 4], [1, 5, 6]]",
)

SCM4_SPEC = SCMExperimentSpec(
    name_of_folder="SCM4",
    d=10,
    ind_cause=8,
    ind_result=9,
    array_titles=np.array(
        ["X_7", "X_8", "X_{10} (DCM)", "X_{10} (BDCM)"], dtype=object
    ),
    array_index_for_epsilon=np.array([6, 7, 9, 9]),
    array_num_input_for_nn=np.array([3, 3, 4, 5]),
    parent=[[2], [3], [6, 8], [2, 3, 8]],
    dag_parents_note="[[], [], [1], [2], [3], [4], [3], [4], [5, 8], [6, 7, 9]]",
)

SCM5_SPEC = SCMExperimentSpec(
    name_of_folder="SCM5",
    d=11,
    ind_cause=8,
    ind_result=10,
    array_titles=np.array(
        ["X_{10}", "X_{11} (DCM)", "X_{11} (BDCM)"], dtype=object
    ),
    array_index_for_epsilon=np.array([9, 10, 10]),
    array_num_input_for_nn=np.array([3, 4, 6]),
    parent=[[8], [8, 9], [1, 5, 8, 9]],
    dag_parents_note="[[], [1], [2], [2], [1], [5], [6], [6], [1, 5], [9], [3, 4, 7, 8, 10]]",
)
