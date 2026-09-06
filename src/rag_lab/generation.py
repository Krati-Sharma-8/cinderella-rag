import json
import os
from urllib.request import Request, urlopen

DISABLED_MESSAGE = "Answer generation is disabled; retrieval results are available above."

class Generator:
    def generate(self, question: str, context: str) -> str:
        raise NotImplementedError

class DisabledGenerator(Generator):
    def generate(self, question: str, context: str) -> str:
        return DISABLED_MESSAGE

class OpenAIGenerator(Generator):
    """Tiny OpenAI-compatible HTTP provider; retrieval has no dependency on it."""
    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")

    def generate(self, question: str, context: str) -> str:
        prompt = ("Answer using only the supplied Cinderella story context. If it is insufficient, "
                  "say the answer cannot be determined from the retrieved context. Do not invent "
                  f"details. Mention supporting chunk IDs.\n\nCONTEXT\n{context}\n\nQUESTION\n{question}")
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": prompt}]}).encode()
        request = Request(f"{self.base_url}/chat/completions", body,
                          {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urlopen(request, timeout=60) as response:
            return json.load(response)["choices"][0]["message"]["content"]

def generator_from_environment(enabled: bool = True) -> Generator:
    key = os.getenv("OPENAI_API_KEY")
    if not enabled or not key:
        return DisabledGenerator()
    return OpenAIGenerator(key, os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                           os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
