# Lazy imports to avoid requiring dependencies unless actually needed
def __getattr__(name):
    if name == "BaseLLMClient":
        from .base import BaseLLMClient
        return BaseLLMClient
    elif name == "RemoteLLMClient":
        from .remote import RemoteLLMClient
        return RemoteLLMClient
    elif name == "LocalLLMClient":
        from .local import LocalLLMClient
        return LocalLLMClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseLLMClient",
    "LocalLLMClient",
    "RemoteLLMClient",
]

