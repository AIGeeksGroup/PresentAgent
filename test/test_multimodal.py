from test.conftest import test_config

from pptagent.multimodal import ImageLabler
from pptagent.presentation import Presentation
from pptagent.utils import pjoin


def test_load_captions():
    prs = Presentation.from_file(
        pjoin(test_config.template, "source.pptx"), test_config.config
    )
    image_labler = ImageLabler(prs, test_config.config)
    image_labler.apply_stats(test_config.get_image_stats())
