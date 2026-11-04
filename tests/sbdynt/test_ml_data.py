import hashlib

import numpy as np
import pytest

from sbdynt import machine_learning as ml


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
    d = ml.sbdynt_cache_dir()
    assert d == str(tmp_path / "cache")
    assert (tmp_path / "cache").is_dir()


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    """A local file:// mirror serving one small training-data file."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    payload = b"not really an npz"
    (mirror / "data.npz").write_bytes(payload)
    monkeypatch.setenv("SBDYNT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SBDYNT_DATA_URL", mirror.as_uri())
    monkeypatch.setitem(ml.training_data_sha256, "data.npz", hashlib.sha256(payload).hexdigest())
    return mirror, payload


def test_fetch_training_data_downloads_and_caches(mirror, tmp_path):
    mirror_dir, payload = mirror
    fn = ml.fetch_training_data("data.npz")
    assert fn == str(tmp_path / "cache" / "data.npz")
    assert open(fn, "rb").read() == payload
    # second call uses the cache, even if the mirror goes away
    (mirror_dir / "data.npz").unlink()
    assert ml.fetch_training_data("data.npz") == fn


def test_fetch_training_data_rejects_bad_hash(mirror, tmp_path):
    mirror_dir, _ = mirror
    (mirror_dir / "data.npz").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        ml.fetch_training_data("data.npz")
    assert list((tmp_path / "cache").iterdir()) == []  # nothing left behind


def test_default_training_data_is_pinned():
    for name in (ml.default_TNO_training_data, ml.default_phi_training_data):
        assert len(ml.training_data_sha256[name]) == 64
    assert ml.training_data_base_url.startswith("https://")
