import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  
def get_llm(model: str = "nvidia/nemotron-3.5-super-vl:free", temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,  # OpenRouter model id, e.g. "anthropic/claude-sonnet-4.5"
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=temperature,
    )