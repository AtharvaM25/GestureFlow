"""
Committed letters -> a readable sentence, via an LLM.

Two providers, same prompt:
  ollama  a model running on this machine (default: gemma3:4b). No key; nothing leaves the PC.
  groq    Groq's hosted API (default: openai/gpt-oss-20b). Needs an API key; used online,
          where there is no Ollama. Only the letters and the instruction below are sent.
"""

DEFAULT_MODEL = "gemma3:4b"

PROMPT = (
    "You are an intelligent English sentence reconstruction assistant.\n\n"

    "Your task is to reconstruct meaningful, grammatically correct "
    "English sentences from a continuous stream of alphabetic characters.\n\n"

    "The input may contain:\n"
    "- Words concatenated together without spaces.\n"
    "- Missing alphabets within individual words.\n"
    "- Missing spaces and punctuation.\n"
    "- Incomplete words or minor spelling errors.\n\n"

    "Follow these instructions:\n\n"

    "1. WORD SEGMENTATION:\n"
    "Identify individual English words within the continuous sequence "
    "of characters. Determine the most probable word boundaries using "
    "context and language understanding.\n\n"

    "2. MISSING CHARACTER RECOVERY:\n"
    "Some alphabets may be missing from the input. Identify incomplete "
    "or misspelled words and infer their most likely intended forms "
    "using surrounding characters and sentence context. Insert missing "
    "letters when necessary to reconstruct recognizable words.\n\n"

    "3. SENTENCE RECONSTRUCTION:\n"
    "Combine the identified words into a meaningful and natural-sounding "
    "English sentence.\n\n"

    "4. GRAMMAR CORRECTION:\n"
    "Correct spelling, grammar, capitalization, and sentence structure.\n\n"

    "5. PUNCTUATION:\n"
    "Add appropriate punctuation, including commas, periods, and "
    "question marks.\n\n"

    "IMPORTANT RULES:\n"
    "- Do not assume that the input contains every alphabet of the "
    "original sentence.\n"
    "- Use contextual understanding to recover missing letters.\n"
    "- Preserve the original intended meaning as much as possible.\n"
    "- Do not invent unrelated words or unnecessary information.\n"
    "- If multiple interpretations are possible, choose the most "
    "probable and natural interpretation.\n"
    "- Prioritize reconstructing the intended sentence over preserving "
    "the exact input characters.\n"
    "- Return ONLY the reconstructed sentence.\n"
    "- Do not provide explanations or alternative interpretations.\n"
    "- If the input cannot reasonably be interpreted, return "
    "'Unable to reconstruct a meaningful sentence.'\n\n"

    "EXAMPLES:\n\n"

    "Input: hellohowareyou\n"
    "Output: Hello, how are you?\n\n"

    "Input: iamgoingtocollege\n"
    "Output: I am going to college.\n\n"

    "Input: iwnttogotocolege\n"
    "Output: I want to go to college.\n\n"

    "Input: hwareyduing\n"
    "Output: How are you doing?\n\n"

    "Input: cnuyhlpm\n"
    "Output: Can you help me?\n\n"

    "Input: iwanttobeadataengineer\n"
    "Output: I want to be a data engineer.\n\n"

    "Now reconstruct the following input.\n"
    "Input: {input}\n"
    "Output:"
)


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
        return ChatGroq(model=model, api_key=api_key, max_tokens=512, temperature=0.2,
                        reasoning_effort="low" if model.startswith("openai/gpt-oss") else None)
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
    return chain.invoke({"input": "".join(letters).lower()}).strip()
