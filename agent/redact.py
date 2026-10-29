"""Hide API keys in any text that is shown, logged or saved."""

import logging
import re

# Anthropic API keys look like sk-ant-api03-...; admin keys like sk-ant-admin01-...
API_KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_-]+")
MASK = "sk-ant-***"


def mask_secrets(text: str) -> str:
    """Replace every API key in the text with a mask."""
    return API_KEY_PATTERN.sub(MASK, text)


class SecretFilter(logging.Filter):
    """Masks API keys in a log record's message and traceback."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = mask_secrets(record.getMessage())
        record.args = None
        if record.exc_info:
            # The formatter uses exc_text as it is once it is set.
            record.exc_text = mask_secrets(logging.Formatter().formatException(record.exc_info))
        return True
