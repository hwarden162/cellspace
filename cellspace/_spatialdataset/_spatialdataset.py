from typing import List

import dask.array as da
from dask_ml.metrics.pairwise import euclidean_distances


class SpatialDataSet:

    def __init__(
        self,
        data: da.Array,
        coords: da.Array,
        col_names: List[str],
        chunk_size: int = 8192,
    ) -> None:
        if not isinstance(data, da.Array):
            raise TypeError("data should be a dask array.")
        if not data.ndim == 2:
            raise ValueError("data should be a 2 dimensional dask array.")
        if not isinstance(coords, da.Array):
            raise TypeError("coords should be a dask array.")
        if not coords.ndim == 2:
            raise ValueError("coords should be a 2 dimensional dask array.")
        if not data.shape[0] == coords.shape[0]:
            raise ValueError("data and coords should have the same number of rows.")
        if not isinstance(col_names, list):
            raise TypeError("col_names should be a list.")
        if not all(isinstance(entry, str) for entry in col_names):
            raise TypeError("col_names should be a list of strings.")
        if not len(col_names) == data.shape[1]:
            raise ValueError("col_names does not match the dimension of the data.")
        if not isinstance(chunk_size, int):
            raise TypeError("chunk_size should be an integer.")
        if not chunk_size > 0:
            raise ValueError("chunk_size should be a positive integer.")
        self._data = data.rechunk((chunk_size, -1))
        self._coords = coords.rechunk((chunk_size, -1))
        self._col_names = col_names
        self._chunk_size = chunk_size
        dist_mat = euclidean_distances(coords)
        dist_mat[da.eye(data.shape[0]).astype(bool)] = 0
        self._dist_mat = dist_mat

    @property
    def data(self) -> da.Array:
        return self._data

    @property
    def coords(self) -> da.Array:
        return self._coords

    @property
    def col_names(self) -> List[str]:
        return self._col_names

    @property
    def chunk_size(self) -> int:
        return self._chunk_size
