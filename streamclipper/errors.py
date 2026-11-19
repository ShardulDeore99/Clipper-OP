class StreamClipperError(Exception):
    """Base class: anything we raise on purpose, with a message a person can act on."""


class ConfigError(StreamClipperError):
    pass


class StageError(StreamClipperError):
    def __init__(self, stage: str, message: str, hint: str = ""):
        self.stage, self.message, self.hint = stage, message, hint
        super().__init__(f"{message} {hint}".strip())
