import os
import hashlib
from dataclasses import dataclass

PROMPT_REGISTRY_ENABLED = os.getenv("PROMPT_REGISTRY_ENABLED") != "false"


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


def build_prompt(id: str, prompt: str, **kwargs: dict) -> Prompt:
    prompt = Prompt(id=id, prompt=prompt, kwargs=kwargs)
    if PROMPT_REGISTRY_ENABLED:
        return prompt

    return str(prompt)
