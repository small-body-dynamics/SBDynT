import pooch

G08_DATA = pooch.create(
    path=pooch.os_cache("sbdynt"),
    base_url="https://www.dropbox.com/scl/fi/9me9e539wkiqngq51cqip/",
    known_hash=None,
    registry={
        "G08_TNO_training_test_set.pkl": None
    }
)

PHI_DATA = pooch.create(
    path=pooch.os_cache("sbdynt"),
    base_url="https://www.dropbox.com/scl/fi/e77qbzn8h8z219lpoa4u2/",
    known_hash=None,
    registry={
        "phi_TNO_training_test_set.pkl": None
    }
)
