"""
Committed letters -> a readable sentence, via an LLM.

Two providers, same prompt:
  ollama  a model running on this machine (default: gemma3:4b). No key; nothing leaves the PC.
  groq    Groq's hosted API (default: openai/gpt-oss-20b). Needs an API key; used online,
          where there is no Ollama. Only the letters and the instruction below are sent.
"""

DEFAULT_MODEL = "gemma3:4b"

PROMPT = ("Rephrase the following input so that it is grammatically "
          "and semantically correct. The input is a bunch of words "
          "without punctuation, so correct that. If no sentence can "
          "be formed, just give an appropriate response. "
          "The input is {input}")


def make_llm(provider="ollama", model=DEFAULT_MODEL, base_url=None, api_key=None):
    # imported here so recognition never depends on the LLM libraries being installed
    if provider == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=model, **({"base_url": base_url} if base_url else {}))
    if provider == "groq":
        if not api_key:
            raise RuntimeError(
                "Groq needs an API key (GESTUREFLOW_GROQ_API_KEY)")
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=api_key, max_tokens=512, temperature=0.2, reasoning_effort="low" if model.startswith("openai/gpt-oss") else None)
    raise ValueError(f"unknown sentence provider {provider!r}")


def generate_sentence(letters, model=DEFAULT_MODEL, base_url=None, provider="ollama",
                      api_key=None):
    """letters: list of committed letters. Raises if the provider is unreachable."""
    if not letters:
        return "No gestures captured."

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    llm = make_llm(provider, model, base_url, api_key)
    chain = PromptTemplate(template=PROMPT, input_variables=[
                           "input"]) | llm | StrOutputParser()
    return chain.invoke({"input": letters}).strip()
