from importlib import resources as impresources

import numpy as np
import pytest

from sbdynt import MLdata
from sbdynt import machine_learning as ml


def _is_lfs_pointer(fn):
    with open(fn, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs")


def _write_g08(fn, features_train):
    np.savez(
        fn,
        class_ids=np.array([0, 1]),
        class_names=np.array(["a", "b"]),
        classes_train=np.array([0, 1]),
        features_train=features_train,
        classes_test=np.array([1]),
        features_test=np.zeros((1, 2)),
    )


def test_g08_loader_roundtrip(tmp_path):
    fn = tmp_path / "g08.npz"
    _write_g08(fn, np.ones((2, 2)))
    classes_dict, ctr, ftr, cte, fte = ml.load_G08_training_testing_data(fn)
    assert classes_dict == {0: "a", 1: "b"}
    assert ftr.shape == (2, 2) and fte.shape == (1, 2)


def test_loaders_refuse_pickled_objects(tmp_path):
    """Training-data loaders must never unpickle (object arrays require pickle)."""
    fn = tmp_path / "g08.npz"
    _write_g08(fn, np.array([object(), None], dtype=object))
    with pytest.raises(ValueError):
        ml.load_G08_training_testing_data(fn)

    fn = tmp_path / "phi.npz"
    obj = np.array([None], dtype=object)
    np.savez(fn, label_train=obj, hog_train=obj, label_test=obj, hog_test=obj)
    with pytest.raises(ValueError):
        ml.load_phi_training_testing_data(fn)


def test_cache_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SBDYNT_CACHE_DIR", str(tmp_path / "cache"))
    d = ml.trained_classifier_cache_dir()
    assert d == str(tmp_path / "cache")
    assert (tmp_path / "cache").is_dir()


@pytest.mark.parametrize(
    "name, loader, n",
    [
        (ml.default_TNO_training_data, ml.load_G08_training_testing_data, 5),
        (ml.default_phi_training_data, ml.load_phi_training_testing_data, 4),
    ],
)
def test_shipped_training_data_loads_without_pickle(name, loader, n):
    fn = impresources.files(MLdata) / name
    assert fn.is_file()
    if _is_lfs_pointer(fn):
        pytest.skip(f"{name} is a Git LFS pointer (checkout without LFS)")
    data = loader(fn)
    assert len(data) == n
