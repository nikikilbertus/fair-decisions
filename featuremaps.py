"""This modules contains feature map implementations.
We only use the identity as a feature map so far, i.e., no feature map.
"""


class FeatureMapIdentity:
    """A feature map that simply returns the original features."""

    def __init__(self):
        """
        Initialize the feature map.
        """
        self.n_components = None

    def __call__(self, *args, **kwargs):
        """Apply the feature map."""
        return self.transform(*args, **kwargs)

    def fit(self, x, y=None):
        """Fit the feature map."""
        self.n_components = x.shape[1]

    def transform(self, x):
        """Transform the input."""
        return x


class FeatureMapRBF:
    def __init__(self):
        # TODO: could use random fourier features instead
        pass
