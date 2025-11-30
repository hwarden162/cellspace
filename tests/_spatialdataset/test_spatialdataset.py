from cellspace._spatialdataset._spatialdataset import  SpatialDataSet
import dask.array as da
import pytest

def test_spatialdataset_validaion():
    data = da.random.random((100, 10))
    coords = da.random.random((100, 2))
    col_names = [f"Var{i+1}" for i in range(10)]
    array_3d = da.random.random((100, 10, 10))
    alt_coords = da.random.random((200, 2))
    with pytest.raises(TypeError, match="data should be a dask array."):
        sds = SpatialDataSet("test", coords, col_names)
    with pytest.raises(ValueError, match="data should be a 2 dimensional dask array."):
        sds = SpatialDataSet(array_3d, coords, col_names)
    with pytest.raises(TypeError, match="coords should be a dask array."):
        sds = SpatialDataSet(data, "test", col_names)
    with pytest.raises(ValueError, match="coords should be a 2 dimensional dask array."):
        sds = SpatialDataSet(data, array_3d, col_names)
    with pytest.raises(ValueError, match="data and coords should have the same number of rows."):
        sds = SpatialDataSet(data, alt_coords, col_names)
    with pytest.raises(TypeError, match="col_names should be a list."):
        sds = SpatialDataSet(data, coords, "hello")
    with pytest.raises(TypeError, match="col_names should be a list of strings."):
        sds = SpatialDataSet(data, coords, [1,2,3])
    with pytest.raises(ValueError, match="col_names does not match the dimension of the data."):
        sds = SpatialDataSet(data, coords, ["Va1", "Var2"])
    with pytest.raises(TypeError, match="chunk_size should be an integer."):
        sds = SpatialDataSet(data, coords, col_names, chunk_size=1.5)
    with pytest.raises(ValueError, match="chunk_size should be a positive integer."):
        sds = SpatialDataSet(data, coords, col_names, chunk_size=-1)
