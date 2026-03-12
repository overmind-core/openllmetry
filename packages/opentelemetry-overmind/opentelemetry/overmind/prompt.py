import hashlib
from dataclasses import dataclass


@dataclass
class Prompt:
    id: str
    prompt: str
    kwargs: dict

    def __str__(self) -> str:
        return self.prompt.format(**self.kwargs)

    def __repr__(self) -> str:
        return self.prompt.format(**self.kwargs)

    @property
    def hash(self):
        return hashlib.md5(self.prompt.encode()).hexdigest()
