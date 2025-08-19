import pytest
from src.app.utils import (
    define_seed,
    fetch_image_as_base64,
    get_image_bytes_from_url,
    is_data_url,
    remove_b64_header,
)


def test_define_seed():
    assert not define_seed(-1) == -1
    assert define_seed(42) == 42


def test_is_data_url():
    assert is_data_url("http://example.com/image.png")
    assert is_data_url("https://example.com/image.png")
    assert not is_data_url("ftp://example.com/image.png")
    assert not is_data_url("example.com/image.png")


def test_fetch_image_as_base64():
    # This test requires an actual image URL to work.
    url = "https://yt3.googleusercontent.com/2eI1TjX447QZFDe6R32K0V2mjbVMKT5mIfQR-wK5bAsxttS_7qzUDS1ojoSKeSP0NuWd6sl7qQ=s900-c-k-c0x00ffffff-no-rj"
    fetch_image_as_base64({"imageUrl": url})


def test_get_image_bytes_from_url():
    # This test requires an actual image URL to work.
    url = "https://yt3.googleusercontent.com/2eI1TjX447QZFDe6R32K0V2mjbVMKT5mIfQR-wK5bAsxttS_7qzUDS1ojoSKeSP0NuWd6sl7qQ=s900-c-k-c0x00ffffff-no-rj"
    get_image_bytes_from_url(url)


def test_remove_b64_header():
    data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA..."
    b64_no_padding = "iVBORw0KGgoAAAANSUhEUgAAAAUA..."
    result = remove_b64_header(data_url)
    assert result.startswith(b64_no_padding)
    assert len(result) % 4 == 0

    non_data_url = "https://example.com/image.png"
    assert remove_b64_header(non_data_url) == non_data_url
