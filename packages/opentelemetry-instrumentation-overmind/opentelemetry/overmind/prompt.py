import hashlib


class PromptString(str):
    id: str
    template: str
    kwargs: dict
    hash: str
    traceable: bool

    def __new__(
        cls,
        *args,
        id: str | None = None,
        template: str | None = None,
        kwargs: dict | None = None,
        **_extra,
    ):
        """
        Supports two construction patterns:
        1) Our explicit usage:
           PromptString(id=\"...\", template=\"...\", kwargs={...})
        2) Library/internal usage (e.g. Agno) that may call:
           PromptString(\"prompt text\", ...)
        """
        if id is not None and template is not None:
            # Explicit, structured construction with template formatting.
            kwargs = kwargs or {}
            prompt = template.format(**kwargs)
            traceable = True
        elif args:
            # Fallback: treat first positional arg as the already-formatted prompt.
            base_prompt = str(args[0])
            prompt = base_prompt
            id = id or "prompt"
            template = template or base_prompt
            kwargs = kwargs or {}
            traceable = False
        else:
            # Extremely defensive fallback – empty prompt.
            prompt = ""
            id = id or "prompt"
            template = template or ""
            kwargs = kwargs or {}
            traceable = False

        obj = super().__new__(cls, prompt)
        obj.id = id
        obj.template = template
        obj.kwargs = kwargs
        obj.hash = hashlib.md5(template.encode()).hexdigest() if template else ""
        obj.traceable = traceable
        return obj
