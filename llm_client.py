import json
import os
import config


def _extract_json(text: str):
    # attempt to find first JSON object/list in text
    start = text.find('{')
    if start == -1:
        start = text.find('[')
    if start == -1:
        raise ValueError('No JSON found in LLM response')
    # naive extract: try progressively larger substrings
    for end in range(len(text), start, -1):
        try:
            candidate = text[start:end]
            return json.loads(candidate)
        except Exception:
            continue
    raise ValueError('Failed to parse JSON from LLM response')


def call_llm(prompt: str, system: str | None = None, model: str | None = None) -> str:
    provider = config.LLM_PROVIDER.lower() if hasattr(config, 'LLM_PROVIDER') else 'openai'
    key = config.LLM_API_KEY
    if not key:
        raise RuntimeError('LLM enabled but LLM_API_KEY not set in environment')

    if provider == 'openai':
        try:
            import openai
        except Exception as e:
            raise RuntimeError('openai package not installed') from e
        openai.api_key = key
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        resp = openai.ChatCompletion.create(model=model or config.LLM_MODEL, messages=messages, temperature=0.2)
        text = resp.choices[0].message.content
        return text

    elif provider in ('dashscope', 'qwen'):
        try:
            from dashscope import Generation
        except Exception as e:
            raise RuntimeError('dashscope package not installed') from e
        # dashscope Generation.call can return structured output; request JSON/text and extract robustly
        try:
            resp = Generation.call(model=model or config.LLM_MODEL, prompt=prompt, api_key=key, output_format='json')
        except TypeError:
            # some versions may not accept output_format='json'
            resp = Generation.call(model=model or config.LLM_MODEL, prompt=prompt, api_key=key)

        # Try common fields for text
        out = ''
        try:
            out = resp.output.get('text', '') if hasattr(resp, 'output') and isinstance(resp.output, dict) else ''
        except Exception:
            out = ''
        if not out:
            try:
                out = str(resp.output)
            except Exception:
                out = str(resp)
        return out

    else:
        raise RuntimeError(f'Unsupported LLM provider: {provider}')


def call_llm_json(prompt: str, system: str | None = None, model: str | None = None):
    text = call_llm(prompt, system=system, model=model)
    try:
        return json.loads(text)
    except Exception:
        return _extract_json(text)
