"""Example tests for hydra."""

from pdum import hydra


def test_version():
    """Test that the package has a version."""
    assert hasattr(hydra, "__version__")
    assert isinstance(hydra.__version__, str)
    assert len(hydra.__version__) > 0


def test_import():
    """Test that the package can be imported."""
    assert hydra is not None

