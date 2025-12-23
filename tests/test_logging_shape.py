import logging

from liq.runner.runner import logger


def test_logging_handlers_present() -> None:
    # Ensure logger is configured to at least accept handlers (shape check)
    assert isinstance(logger, logging.Logger)
    # Logging shouldn't raise; we don't assert output content here
    logger.info("log_shape_test", extra={"foo": "bar"})
