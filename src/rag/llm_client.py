"""
Wrapper unificado para chamadas a LLMs (OpenAI, Gemini, etc.).

Usa os parâmetros definidos em config.py e lê a API key de variáveis de ambiente.
"""

import os
from typing import Literal
from dotenv import load_dotenv
from pathlib import Path
from config import (
    LLM_PROVIDER,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    OPENAI_API_KEY_ENVVAR,
    GEMINI_API_KEY_ENVVAR,
)

def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    api_key = os.getenv(OPENAI_API_KEY_ENVVAR)
    if not api_key:
        raise RuntimeError(
            f"Variável de ambiente {OPENAI_API_KEY_ENVVAR} não encontrada."
        )

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": "Você é um assistente especialista em recomendação de livros."},
            {"role": "user", "content": prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return resp.choices[0].message.content or ""

def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai
    dotenv_path = Path(__file__).parent.parent.parent / '.env'
    print(dotenv_path)
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv(GEMINI_API_KEY_ENVVAR)
    if not api_key:
        raise RuntimeError(
            f"Variável de ambiente {GEMINI_API_KEY_ENVVAR} não encontrada."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(LLM_MODEL_NAME)
    resp = model.generate_content(prompt)
    return resp.text or ""

def generate_llm_response(prompt: str) -> str:
    """
    Gera uma resposta de LLM usando o provider definido em config.py.
    """
    provider: Literal["openai", "gemini"] = LLM_PROVIDER

    if provider == "openai":
        return _call_openai(prompt)
    elif provider == "gemini":
        return _call_gemini(prompt)
    else:
        raise ValueError(
            f"LLM_PROVIDER='{provider}' não suportado. Use 'openai' ou 'gemini'."
        )