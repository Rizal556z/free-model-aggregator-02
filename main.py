# main.py — a free-model aggregator for make.fun's free-model-aggregator-02 bounty.
#
# One long-running program, Python 3.13 standard library only. It listens on a port it
# picks, registers with the trial env, and serves the two completion wires the env drives:
#   POST /v1/chat/completions   (OpenAI wire, Authorization: Bearer)
#   POST /v1/messages           (Anthropic wire, x-api-key + anthropic-version)
#   GET  /v1/models             (the model list, shaped by whichever auth header arrived)
#
# Every prompt the env sends names model "auto". This server replaces it with the best
# free model that can serve the request — ranked by the brief's own quality rating —
# dials that model on the grader's gateway (Chat Completions only, never Messages), and
# returns the gateway's answer in the wire the env used, streamed or not, with its usage
# and stop reason whole. The `model` on every answer and every streamed chunk is the id
# actually dialled, because the scorer only credits a prompt whose reported model has a
# matching successful call in the trial's gateway record.
#
# How it is run (see the kit's RULES.txt):
#   python3 -P -s /bot/main.py            (a trial; -P keeps /bot off sys.path, -s drops
#                                          the user site-packages, so stdlib only, one file)
#   python3 main.py --list               (print the catalog and exit 0, no server)
#
# What the trial gives it:
#   MAKEFUN_ENV_URL      the env: POST /register, GET /work ({"done": true} ends the run)
#   MAKEFUN_PROMPT_FILE  the brief: the free model list (ids, ratings, capabilities) and
#                        the six prompts in the order the env sends them
#   OPENAI_BASE_URL      the gateway's OpenAI-compatible root (the only outside service)
#   OPENAI_API_KEY       the trial's dummy key; the gateway swaps it for the entrant's
#                        own pushed OpenRouter key, which never enters the sandbox
#
# Determinism: a trial is replayed and the export must match, so every route is a
# function of the requests and the gateway's answers seen so far, never of the wall
# clock. Model order is fixed (rating desc, id asc), failures are remembered per model
# for the rest of the run (no time-based cooldowns), and every timestamp in a reply is
# a fixed constant. Randomness, if ever needed, would come from MAKEFUN_TRIAL_SEED.

import json
import os
import re
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# ---------------------------------------------------------------- constants

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024
# The gateway refuses a call with no output cap, or one over this cap.
MAX_OUTPUT_TOKENS = 8192
DEFAULT_OUTPUT_TOKENS = 1024
# A trial may make at most 60 model calls; six prompts times a few attempts stays far under.
MAX_ATTEMPTS_PER_PROMPT = 6

ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')
GATEWAY_URL = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
GATEWAY_KEY = os.environ.get('OPENAI_API_KEY', '')

MODELS_SCHEMA = 'free-model-aggregator-02/models/v1'
PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1'
SCHEMA_PREFIX = 'free-model-aggregator-02/'

# The provider's public endpoint, spelled in the catalog. The server never dials it:
# every model call goes through the gateway at $OPENAI_BASE_URL. The catalog is part of
# the state a replay compares byte for byte, and the gateway's origin differs between a
# live pass and its replay, so the public endpoint is the only stable spelling.
PUBLIC_BASE_URL = 'https://openrouter.ai/api/v1'
KEY_ENV_NAME = 'OPENROUTER_API_KEY'

# Fixed timestamps in every model-list reply: nothing in a reply reads the clock.
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = '2026-01-01T00:00:00Z'

# A prompt's unique token, as the brief spells it: capitals, a dash, digits.
TOKEN_RE = re.compile(r'[A-Z]{2,}-[0-9]{3,}')

Json = dict[str, object]

# ---------------------------------------------------------------- the free models
#
# The fourteen free models, as the brief lists them, in case the trial runs with no
# brief to read (outside a trial, `python3 main.py --list`). At trial time the brief's
# own list wins and this table is only a fallback; the ids, ratings and capabilities
# are the brief's sealed copy, so the two agree.

EMBEDDED_MODELS: list[Json] = [
    {'id': 'openrouter/baidu/ernie-5.5-flash-preview:free', 'provider': 'openrouter',
     'display_name': 'Baidu: ERNIE-5.5-Flash-Preview (free)', 'rating_milli': 1150,
     'context_length': 1024000, 'tools': True, 'vision': False, 'json': True, 'free': True},
    {'id': 'openrouter/cohere/north-mini-code:free', 'provider': 'openrouter',
     'display_name': 'Cohere: North Mini Code (free)', 'rating_milli': 1200,
     'context_length': 256000, 'tools': True, 'vision': False, 'json': False, 'free': True},
    {'id': 'openrouter/dots-studio/dots-3-note-preview:free', 'provider': 'openrouter',
     'display_name': 'Dots Studio: Dots3-Note Preview (free)', 'rating_milli': 1150,
     'context_length': 512000, 'tools': True, 'vision': True, 'json': True, 'free': True},
    {'id': 'openrouter/google/gemma-4-26b-a4b-it:free', 'provider': 'openrouter',
     'display_name': 'Google: Gemma 4 26B A4B (free)', 'rating_milli': 1150,
     'context_length': 262144, 'tools': True, 'vision': True, 'json': True, 'free': True},
    {'id': 'openrouter/google/gemma-4-31b-it:free', 'provider': 'openrouter',
     'display_name': 'Google: Gemma 4 31B (free)', 'rating_milli': 1210,
     'context_length': 262144, 'tools': True, 'vision': True, 'json': True, 'free': True},
    {'id': 'openrouter/inclusionai/ling-3.2-flash:free', 'provider': 'openrouter',
     'display_name': 'inclusionAI: Ling 3.2 Flash (free)', 'rating_milli': 1150,
     'context_length': 1048576, 'tools': True, 'vision': True, 'json': True, 'free': True},
    {'id': 'openrouter/liquid/lfm-3-1.7b:free', 'provider': 'openrouter',
     'display_name': 'LiquidAI: LFM3-1.7B (free)', 'rating_milli': 1150,
     'context_length': 262144, 'tools': False, 'vision': False, 'json': True, 'free': True},
    {'id': 'openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free', 'provider': 'openrouter',
     'display_name': 'NVIDIA: Nemotron 3 Nano Omni (free)', 'rating_milli': 1150,
     'context_length': 256000, 'tools': True, 'vision': True, 'json': False, 'free': True},
    {'id': 'openrouter/nvidia/nemotron-3-ultra-550b-a55b:free', 'provider': 'openrouter',
     'display_name': 'NVIDIA: Nemotron 3 Ultra (free)', 'rating_milli': 1285,
     'context_length': 1000000, 'tools': True, 'vision': False, 'json': False, 'free': True},
    {'id': 'openrouter/nvidia/nemotron-3.5-lightning:free', 'provider': 'openrouter',
     'display_name': 'NVIDIA: Nemotron 3.5 Lightning (free)', 'rating_milli': 1221,
     'context_length': 1000000, 'tools': True, 'vision': False, 'json': False, 'free': True},
    {'id': 'openrouter/nvidia/nemotron-3.5-super-vl:free', 'provider': 'openrouter',
     'display_name': 'NVIDIA: Nemotron 3.5 Super VL (free)', 'rating_milli': 1303,
     'context_length': 1000000, 'tools': True, 'vision': True, 'json': True, 'free': True},
    {'id': 'openrouter/particle-ai/particle-1:free', 'provider': 'openrouter',
     'display_name': 'Particle AI: Particle-1 (free)', 'rating_milli': 1150,
     'context_length': 1048576, 'tools': True, 'vision': True, 'json': True, 'free': True},
    {'id': 'openrouter/recursive/flint-v1.0:free', 'provider': 'openrouter',
     'display_name': 'Recursive: Flint (free)', 'rating_milli': 1150,
     'context_length': 262144, 'tools': False, 'vision': False, 'json': True, 'free': True},
    {'id': 'openrouter/thinkingmachines/inkling-2-small:free', 'provider': 'openrouter',
     'display_name': 'Thinking Machines: Inkling 2 Small (free)', 'rating_milli': 1475,
     'context_length': 1048576, 'tools': True, 'vision': True, 'json': False, 'free': True},
]


def is_record(value: object) -> bool:
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def text_of(value: object) -> str:
    return value if isinstance(value, str) else ''


# ---------------------------------------------------------------- reading the brief


def object_end(text: str, start: int) -> int:
    """Index of the `}` closing the object that opens at `start`, or -1; strings skipped."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text: str) -> list[Json]:
    """Every JSON object in `text` carrying a string `schema`, in the order they appear."""
    out: list[Json] = []
    start = text.find('{')
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get('schema'), str):
                out.append(value)
        start = text.find('{', start + 1)
    return out


def clean_model(value: object) -> Json | None:
    """One model row, normalised. A row is routed as the gateway spells its id."""
    if not is_record(value):
        return None
    raw = text_of(value.get('id')) or text_of(value.get('model'))
    if raw == '':
        return None
    provider = text_of(value.get('provider')) or text_of(value.get('wire'))
    model_id = f'openrouter/{raw}' if provider == 'openrouter' and not raw.lower().startswith('openrouter/') else raw
    rating = value.get('rating_milli')
    rating = rating if is_number(rating) else 0
    context = value.get('context_length')
    context = context if is_number(context) else None
    return {
        'id': model_id,
        'provider': provider if provider != '' else 'openrouter',
        'display_name': text_of(value.get('display_name')) or raw,
        'rating_milli': rating,
        'context_length': context,
        'tools': value.get('tools') if isinstance(value.get('tools'), bool) else None,
        'vision': value.get('vision') if isinstance(value.get('vision'), bool) else None,
        'json': value.get('json') if isinstance(value.get('json'), bool) else None,
        'free': value.get('free') is not False,
    }


def read_brief(path: str) -> list[Json]:
    """The brief's free model list, or the embedded fourteen when there is no brief."""
    if path == '':
        return [dict(m) for m in EMBEDDED_MODELS]
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
    except OSError:
        return [dict(m) for m in EMBEDDED_MODELS]
    for doc in documents_in(text):
        schema = text_of(doc.get('schema'))
        if SCHEMA_PREFIX not in schema:
            continue
        for key in ('models', 'ratings'):
            rows = doc.get(key)
            if isinstance(rows, list):
                models = [m for m in (clean_model(v) for v in rows) if m is not None]
                if models:
                    return models
    return [dict(m) for m in EMBEDDED_MODELS]


def build_catalog(models: list[Json]) -> Json:
    """The catalog document: one provider, exactly the free models, once each.

    base_url is the provider's public endpoint, never the gateway's trial-local
    address: the gateway's origin differs between a live pass and its replay, and the
    catalog is part of the state a replay compares byte for byte.
    """
    seen: set[str] = set()
    listed: list[Json] = []
    for model in models:
        if not model['free'] or model['id'] in seen:
            continue
        seen.add(model['id'])
        listed.append({'id': model['id'], 'free': True, 'context_length': model['context_length']})
    if not listed:
        return {'providers': []}
    return {'providers': [{'id': 'openrouter', 'base_url': PUBLIC_BASE_URL,
                           'key_env': KEY_ENV_NAME, 'wire': 'openai', 'models': listed}]}


# ---------------------------------------------------------------- ranking and requirements


def meets(requirements: Json, model: Json) -> bool:
    """Whether a model's capabilities meet a request's requirements.

    An unknown capability never meets a requirement: a model that does not say it
    supports JSON mode is not trusted with a JSON-mode prompt.
    """
    if requirements.get('tools') is True and model['tools'] is not True:
        return False
    if requirements.get('vision') is True and model['vision'] is not True:
        return False
    if requirements.get('json') is True and model['json'] is not True:
        return False
    minimum = requirements.get('min_context')
    if minimum is not None and (model['context_length'] is None or model['context_length'] < minimum):
        return False
    return True


def ranked(models: list[Json], requirements: Json, failed: dict[str, int], seq: int) -> list[Json]:
    """The free models that can serve a request, best first.

    Order is the brief's rating, highest first, ties broken by id, so the order is the
    same on every run. A model that refused is skipped for the next prompt and
    trusted again after that: a refusal is a property of the requests and answers
    seen so far, never of the wall clock, and one bad call should not demote the
    best model for the whole run when the failure was transient.
    """
    candidates = [m for m in models
                  if m['free'] and seq - failed.get(m['id'], -10**9) >= 2 and meets(requirements, m)]
    candidates.sort(key=lambda m: (-m['rating_milli'], m['id']))
    return candidates


def requirements_from_body(body: Json) -> Json:
    """What a request needs, read off the request itself.

    JSON mode comes from its response_format, tools from its tools list, vision from
    an image part in its messages. Reading the body, not the brief's manifest, means
    the requirements are the ones of the request actually arriving.
    """
    vision = False
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    for message in messages:
        if not is_record(message) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if is_record(block) and block.get('type') in ('image', 'image_url'):
                vision = True
    fmt = body.get('response_format')
    fmt_type = fmt.get('type') if is_record(fmt) else None
    return {
        'tools': isinstance(body.get('tools'), list) and len(body['tools']) > 0,
        'vision': vision,
        'json': fmt_type in ('json_object', 'json_schema'),
    }


def tokens_in_request(body: Json) -> list[str]:
    """The prompt's unique tokens, read off the request body (one per prompt, usually)."""
    seen: list[str] = []
    for token in TOKEN_RE.findall(json.dumps(body, ensure_ascii=False)):
        if token not in seen:
            seen.append(token)
    return seen

# ---------------------------------------------------------------- the client wires
#
# Two wires, one truth: the gateway only speaks Chat Completions, so an Anthropic
# request is translated into a Chat Completions body before it is dialled, and the
# gateway's answer is translated back into the Messages shape. The `model` on every
# answer and every streamed chunk is the id actually dialled.

def openai_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an OpenAI-shaped request (Authorization: Bearer)."""
    return {'object': 'list', 'data': [
        {'id': m['id'], 'object': 'model', 'created': FIXED_CREATED, 'owned_by': m['provider']}
        for m in models]}


def anthropic_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an Anthropic-shaped request (an x-api-key header)."""
    return {'data': [
        {'type': 'model', 'id': m['id'], 'display_name': m['display_name'],
         'created_at': FIXED_CREATED_AT} for m in models], 'has_more': False}


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    return {'error': {'message': message, 'type': error_type, 'code': code}}


def anthropic_error(error_type: str, message: str) -> Json:
    return {'type': 'error', 'error': {'type': error_type, 'message': message}}


def text_blocks(content: object) -> str:
    """The text of a string, or of a block array's text blocks joined with newlines."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    return '\n'.join(b['text'] for b in content
                     if is_record(b) and b.get('type') == 'text' and isinstance(b.get('text'), str))


def part_of(block: Json) -> Json | None:
    """One Anthropic content block as an OpenAI content part: text or an inline image."""
    if block.get('type') == 'text' and isinstance(block.get('text'), str):
        return {'type': 'text', 'text': block['text']}
    source = block.get('source')
    if block.get('type') == 'image' and is_record(source):
        if source.get('type') == 'base64' and isinstance(source.get('data'), str):
            media = source['media_type'] if isinstance(source.get('media_type'), str) else 'image/png'
            return {'type': 'image_url', 'image_url': {'url': f'data:{media};base64,{source["data"]}'}}
        if source.get('type') == 'url' and isinstance(source.get('url'), str):
            return {'type': 'image_url', 'image_url': {'url': source['url']}}
    return None


def tool_result_text(content: object) -> str:
    """A tool_result block's content as text: its text blocks, else its JSON."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_blocks(content)
        return text if text != '' else json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    if content is None:
        return ''
    return json.dumps(content, separators=(',', ':'), ensure_ascii=False)


def args_object(value: object) -> Json:
    """A tool call's arguments as an object: the gateway's JSON string, else {}."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != '':
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def stop_reason_of(finish: object) -> str:
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == 'length':
        return 'max_tokens'
    if finish in ('tool_calls', 'function_call'):
        return 'tool_use'
    if finish == 'content_filter':
        return 'refusal'
    return 'end_turn'


def anthropic_to_openai(body: Json, model: str) -> Json:
    """An Anthropic Messages request as a Chat Completions request naming `model`.

    The system prompt becomes a system message, image blocks become image_url data
    URLs, tool_use blocks become tool_calls, tool_result blocks become tool messages,
    and tools, tool_choice, the output cap, temperature, top_p, stop sequences and
    streaming pass through.
    """
    messages: list[Json] = []
    system = text_blocks(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        content = raw.get('content')
        if raw.get('role') == 'assistant':
            if isinstance(content, str):
                messages.append({'role': 'assistant', 'content': content})
                continue
            if not isinstance(content, list):
                continue
            calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get('type') != 'tool_use':
                    continue
                calls.append({'id': block['id'] if isinstance(block.get('id'), str) else '',
                              'type': 'function',
                              'function': {'name': block['name'] if isinstance(block.get('name'), str) else '',
                                           'arguments': json.dumps(block['input'] if is_record(block.get('input')) else {},
                                                                  separators=(',', ':'), ensure_ascii=False)}})
            text = text_blocks(content)
            message: Json = {'role': 'assistant', 'content': text if text != '' else None}
            if calls:
                message['tool_calls'] = calls
            messages.append(message)
            continue
        if isinstance(content, str):
            messages.append({'role': 'user', 'content': content})
            continue
        if not isinstance(content, list):
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get('type') == 'tool_result':
                messages.append({'role': 'tool',
                                 'tool_call_id': block['tool_use_id'] if isinstance(block.get('tool_use_id'), str) else '',
                                 'content': tool_result_text(block.get('content'))})
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({'role': 'user', 'content': parts})

    out: Json = {'model': model, 'messages': messages}
    if is_number(body.get('max_tokens')):
        out['max_tokens'] = body['max_tokens']
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    if isinstance(body.get('stop_sequences'), list) and len(body['stop_sequences']) > 0:
        out['stop'] = body['stop_sequences']
    if isinstance(body.get('tools'), list):
        tools: list[Json] = []
        for raw in body['tools']:
            if not is_record(raw) or not isinstance(raw.get('name'), str):
                continue
            tools.append({'type': 'function',
                          'function': {'name': raw['name'],
                                       'description': raw['description'] if isinstance(raw.get('description'), str) else '',
                                       'parameters': raw['input_schema'] if is_record(raw.get('input_schema')) else {'type': 'object', 'properties': {}}}})
        if tools:
            out['tools'] = tools
    choice = body.get('tool_choice')
    if is_record(choice):
        if choice.get('type') == 'auto':
            out['tool_choice'] = 'auto'
        elif choice.get('type') == 'any':
            out['tool_choice'] = 'required'
        elif choice.get('type') == 'tool' and isinstance(choice.get('name'), str):
            out['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
    if body.get('stream') is True:
        out['stream'] = True
        out['stream_options'] = {'include_usage': True}
    return out


def openai_to_anthropic(body: Json, model: str) -> Json:
    """A Chat Completions answer as an Anthropic Messages object; `model` is the dialled id."""
    choices = body.get('choices') if isinstance(body.get('choices'), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get('message') if is_record(choice.get('message')) else {}
    content: list[Json] = []
    text = message['content'] if isinstance(message.get('content'), str) else text_blocks(message.get('content'))
    if text != '':
        content.append({'type': 'text', 'text': text})
    calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get('function')):
            continue
        fn = call['function']
        content.append({'type': 'tool_use',
                        'id': call['id'] if isinstance(call.get('id'), str) else 'toolu_unknown',
                        'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                        'input': args_object(fn.get('arguments'))})
    if not content:
        content.append({'type': 'text', 'text': ''})
    usage = body.get('usage') if is_record(body.get('usage')) else {}
    return {'id': f'msg_{body["id"]}' if isinstance(body.get('id'), str) else 'msg_aggregate',
            'type': 'message', 'role': 'assistant', 'model': model, 'content': content,
            'stop_reason': stop_reason_of(choice.get('finish_reason')), 'stop_sequence': None,
            'usage': {'input_tokens': usage.get('prompt_tokens') if is_number(usage.get('prompt_tokens')) else 0,
                      'output_tokens': usage.get('completion_tokens') if is_number(usage.get('completion_tokens')) else 0}}


def with_model(body: Json, model: str) -> Json:
    """A Chat Completions answer with `model` set to the id actually dialled."""
    return {**body, 'model': model}


# ---------------------------------------------------------------- server-sent events


def sse_event(event: str | None, data: object) -> str:
    """One server-sent event: an `event:` line when named, then the data line, then blank."""
    prefix = '' if event is None else f'event: {event}\n'
    return f'{prefix}data: {json.dumps(data, separators=(",", ":"), ensure_ascii=False)}\n\n'


def iter_frames(chunks) -> object:
    """SSE frames from an upstream body's byte chunks, split on the blank line.

    A frame cut across reads is held until its end arrives; a body that ends without
    one is yielded as it is. CRLF is normalised to LF as the SSE grammar reads it.
    """
    buffer = ''
    for chunk in chunks:
        buffer += chunk.decode('utf-8', 'replace').replace('\r\n', '\n')
        at = buffer.find('\n\n')
        while at != -1:
            yield buffer[:at]
            buffer = buffer[at + 2:]
            at = buffer.find('\n\n')
    if buffer.strip() != '':
        yield buffer


def openai_frames(frames: list[str], model: str) -> list[str]:
    """Chat Completions frames with every data payload's `model` the dialled id.

    Comments, `event:` lines, `[DONE]` and payloads that are not JSON objects pass
    through untouched; the client sees the id this server dialled, never the
    provider's spelling of it.
    """
    out: list[str] = []
    for frame in frames:
        lines: list[str] = []
        for line in frame.split('\n'):
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                lines.append(line)
                continue
            payload = trimmed[5:].strip()
            if payload == '' or payload == '[DONE]':
                lines.append(line)
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                lines.append(line)
                continue
            if is_record(chunk):
                lines.append(f'data: {json.dumps({**chunk, "model": model}, separators=(",", ":"), ensure_ascii=False)}')
            else:
                lines.append(line)
        out.append('\n'.join(lines) + '\n\n')
    return out


def anthropic_events(frames: list[str], model: str) -> list[str]:
    """A Chat Completions SSE stream as the Anthropic event sequence.

    message_start on the first chunk, content_block_start/delta/stop per text block
    or tool call, then message_delta with the stop reason and usage, then
    message_stop. The stop reason is held until the stream's usage chunk has arrived:
    OpenAI sends usage after the finish_reason, and Anthropic's message_delta carries
    the output tokens.
    """
    started = False
    finished = False
    pending_stop: str | None = None
    text_index = -1
    next_index = 0
    tool_indexes: dict[object, int] = {}
    input_tokens = 0
    output_tokens = 0
    out: list[str] = []

    def start_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [sse_event('message_start', {'type': 'message_start',
                'message': {'id': 'msg_stream', 'type': 'message', 'role': 'assistant',
                            'model': model, 'content': [], 'stop_reason': None,
                            'stop_sequence': None,
                            'usage': {'input_tokens': input_tokens, 'output_tokens': 0}}})]

    def close_events() -> list[str]:
        nonlocal text_index
        events: list[str] = []
        if text_index != -1:
            events.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            events.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        tool_indexes.clear()
        return events

    def finish_events(stop_reason: str) -> list[str]:
        nonlocal finished
        if finished:
            return []
        finished = True
        return [*close_events(),
                sse_event('message_delta', {'type': 'message_delta',
                         'delta': {'stop_reason': stop_reason, 'stop_sequence': None},
                         'usage': {'output_tokens': output_tokens}}),
                sse_event('message_stop', {'type': 'message_stop'})]

    for frame in frames:
        for line in frame.split('\n'):
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                continue
            payload = trimmed[5:].strip()
            if payload == '[DONE]':
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else 'end_turn'))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get('usage')):
                input_tokens = chunk['usage'].get('prompt_tokens') if is_number(chunk['usage'].get('prompt_tokens')) else 0
                output_tokens = chunk['usage'].get('completion_tokens') if is_number(chunk['usage'].get('completion_tokens')) else 0
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get('delta') if is_record(choice.get('delta')) else {}
            out.extend(start_events())
            if isinstance(delta.get('content'), str) and delta['content'] != '':
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(sse_event('content_block_start', {'type': 'content_block_start',
                               'index': text_index, 'content_block': {'type': 'text', 'text': ''}}))
                out.append(sse_event('content_block_delta', {'type': 'content_block_delta',
                           'index': text_index,
                           'delta': {'type': 'text_delta', 'text': delta['content']}}))
            calls = delta.get('tool_calls') if isinstance(delta.get('tool_calls'), list) else []
            for call in calls:
                if not is_record(call):
                    continue
                upstream_index = call['index'] if is_number(call.get('index')) else len(tool_indexes)
                index = tool_indexes.get(upstream_index)
                if index is None:
                    if text_index != -1:
                        out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
                        text_index = -1
                    index = next_index
                    next_index += 1
                    tool_indexes[upstream_index] = index
                    fn = call.get('function') if is_record(call.get('function')) else {}
                    out.append(sse_event('content_block_start', {'type': 'content_block_start',
                               'index': index,
                               'content_block': {'type': 'tool_use',
                                                 'id': call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else f'toolu_{index}',
                                                 'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                                                 'input': {}}}))
                fn = call.get('function') if is_record(call.get('function')) else {}
                if isinstance(fn.get('arguments'), str) and fn['arguments'] != '':
                    out.append(sse_event('content_block_delta', {'type': 'content_block_delta',
                               'index': index,
                               'delta': {'type': 'input_json_delta', 'partial_json': fn['arguments']}}))
            if choice.get('finish_reason') is not None:
                pending_stop = stop_reason_of(choice.get('finish_reason'))
    if not finished:
        out.extend(start_events())
        out.extend(finish_events(pending_stop if pending_stop is not None else 'end_turn'))
    return out

# ---------------------------------------------------------------- the gateway


def chat_completions_url() -> str:
    """The gateway's Chat Completions route.

    OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the trial network); a
    base that does not already end in `/v1` gets one, so both spellings reach it.
    """
    if GATEWAY_URL == '':
        return ''
    return f'{GATEWAY_URL}/chat/completions' if GATEWAY_URL.endswith('/v1') else f'{GATEWAY_URL}/v1/chat/completions'


def cap_output(body: Json) -> Json:
    """The request with an output cap the gateway accepts.

    A call with no cap, or one over the cap, is refused: name max_tokens (or
    max_completion_tokens), at most 8192, and 1024 when the client named none.
    """
    out = dict(body)
    if is_number(out.get('max_tokens')):
        out['max_tokens'] = min(out['max_tokens'], MAX_OUTPUT_TOKENS)
    elif is_number(out.get('max_completion_tokens')):
        out['max_completion_tokens'] = min(out['max_completion_tokens'], MAX_OUTPUT_TOKENS)
    else:
        out['max_tokens'] = DEFAULT_OUTPUT_TOKENS
    return out


class Upstream:
    """One gateway answer: its status and a body read whole or as byte chunks."""

    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        """The body's bytes as they arrive, never waiting to fill a buffer."""
        reader = getattr(self._response, 'read1', None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode('utf-8', 'replace')
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:
            pass


def dial(model: str, body: Json) -> Upstream:
    """POST one request to the gateway's Chat Completions route, naming `model`."""
    url = chat_completions_url()
    if url == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    data = json.dumps({**cap_output(body), 'model': model}, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    headers = {'content-type': 'application/json', 'authorization': f'Bearer {GATEWAY_KEY}'}
    if body.get('stream') is True:
        headers['accept'] = 'text/event-stream'
    request = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as err:
        # A refusal is an answer with a status and a body, exactly as the provider sent it.
        return Upstream(err.code, err)
    return Upstream(response.status, response)


def error_code_of(text: str, status: int) -> str:
    """The gateway error's code, from its body when it parses, else its status."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if is_record(parsed):
        error = parsed.get('error')
        if is_record(error) and isinstance(error.get('code'), str) and error['code'] != '':
            return error['code']
        if is_record(error) and isinstance(error.get('type'), str) and error['type'] != '':
            return error['type']
    return f'http_{status}'


def wrap_text_answer(raw: str, model: str) -> Json:
    """A Chat Completions answer built from unparsable upstream text, so the reply is
    still a well-formed completion naming the model that served it."""
    return {'id': 'chatcmpl-aggregate', 'object': 'chat.completion', 'created': FIXED_CREATED,
            'model': model,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': raw},
                         'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 0, 'completion_tokens': 1, 'total_tokens': 1}}


# ---------------------------------------------------------------- serving one prompt

MODELS = read_brief(PROMPT_FILE)
CATALOG = build_catalog(MODELS)
# Models that refused, or answered without the prompt's token, and the prompt number
# they failed on: skipped for the next prompt, trusted again after that. Keyed by
# history, never by the clock, so a replay sees the same order.
FAILED: dict[str, int] = {}
PROMPT_SEQ = 0
FAILED_LOCK = threading.Lock()


def log(message: str) -> None:
    sys.stderr.write(f'aggregator: {message}\n')
    sys.stderr.flush()


def serve(wire: str, client_body: Json, openai_body: Json, requirements: Json, tokens: list[str]):
    """Answer one prompt: dial the best free model that can serve it, fail over on a
    refusal, and verify the answer carries the prompt's token before serving it.

    Returns (status, kind, payload, failures) where kind is 'json' (a dict to send as
    JSON), 'sse' (a list of event strings to stream) or 'error' (a dict to send as an
    error body in the client's wire). `failures` is the trail of refused calls, for
    the x-makefun-upstream-failures header: the only way the failover column can see
    a refusal that was retried and then served.
    """
    stream = client_body.get('stream') is True
    failures: list[Json] = []
    fallback: tuple[str, str, list[str]] | None = None  # (model, raw text, frames)
    global PROMPT_SEQ
    with FAILED_LOCK:
        PROMPT_SEQ += 1
        seq = PROMPT_SEQ
        candidates = ranked(MODELS, requirements, FAILED, seq)
    for model in candidates[:MAX_ATTEMPTS_PER_PROMPT]:
        model_id = model['id']
        try:
            upstream = dial(model_id, openai_body)
        except Exception:
            failures.append({'status': 0, 'code': 'unreachable', 'model': model_id})
            with FAILED_LOCK:
                FAILED[model_id] = seq
            continue
        if not upstream.ok:
            text = upstream.text()
            failures.append({'status': upstream.status,
                             'code': error_code_of(text, upstream.status), 'model': model_id})
            with FAILED_LOCK:
                FAILED[model_id] = seq
            continue
        if stream:
            frames = list(iter_frames(upstream.chunks()))
            upstream.close()
            raw = '\n'.join(frames)
        else:
            raw = upstream.text()
            frames = []
        if all(token in raw for token in tokens):
            return finish(wire, stream, model_id, raw, frames, failures)
        # A 200 that does not carry the token earns the prompt nothing: keep it as the
        # fallback (a well-formed answer still passes the wire checks) and try the next
        # model. The model is skipped for the next prompt and trusted again after that.
        if fallback is None:
            fallback = (model_id, raw, frames)
        with FAILED_LOCK:
            FAILED[model_id] = seq

    if fallback is not None:
        model_id, raw, frames = fallback
        return finish(wire, stream, model_id, raw, frames, failures)
    message = 'the gateway refused every free model that could serve this request'
    return (502, 'error', {'message': message, 'type': 'upstream_error'}, failures)


def finish(wire: str, stream: bool, model_id: str, raw: str, frames: list[str], failures: list[Json]):
    """A successful gateway answer as the client's reply, in the wire it arrived on."""
    if stream:
        if wire == 'anthropic':
            return (200, 'sse', anthropic_events(frames, model_id), failures)
        return (200, 'sse', openai_frames(frames, model_id), failures)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    answer = parsed if is_record(parsed) else wrap_text_answer(raw, model_id)
    if wire == 'anthropic':
        return (200, 'json', openai_to_anthropic(answer, model_id), failures)
    return (200, 'json', with_model(answer, model_id), failures)


def requirements_from_anthropic(body: Json) -> Json:
    """What an Anthropic request needs: tools from its tools list, vision from an
    image block, never JSON mode (the Messages wire has none)."""
    vision = False
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    for message in messages:
        if not is_record(message) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if is_record(block) and block.get('type') == 'image':
                vision = True
    return {'tools': isinstance(body.get('tools'), list) and len(body['tools']) > 0,
            'vision': vision, 'json': False}


# ---------------------------------------------------------------- the HTTP server


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires, nothing else."""

    protocol_version = 'HTTP/1.1'
    server_version = 'makefun-aggregator-py/1'
    sys_version = ''

    def log_message(self, format: str, *args: object) -> None:  # the base class's name
        pass

    def do_GET(self) -> None:  # the base class's name
        self.dispatch()

    def do_POST(self) -> None:  # the base class's name
        self.dispatch()

    # -- plumbing -----------------------------------------------------------------

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == 'GET' and path == '/v1/models':
                self.models()
                return
            if self.command == 'POST' and path == '/v1/chat/completions':
                self.chat()
                return
            if self.command == 'POST' and path == '/v1/messages':
                self.messages()
                return
            wire = 'anthropic' if path == '/v1/messages' else 'openai'
            self.send_error_body(wire, 404, 'not_found', f'no route {self.command} {path}')
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:  # one request's failure must not take the server down
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = 'anthropic' if urlsplit(self.path).path.startswith('/v1/messages') else 'openai'
                try:
                    self.send_error_body(wire, 500, 'internal_error', 'the server failed to answer')
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object, failures: list[Json] | None = None) -> None:
        text = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        raw = text.encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        self.failures_header(failures)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def failures_header(self, failures: list[Json] | None) -> None:
        """The trail of refused calls, at most 8 rows: the failover column reads it."""
        if failures:
            trail = json.dumps(failures[:8], separators=(',', ':'), ensure_ascii=False)
            self.send_header('x-makefun-upstream-failures', trail)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str,
                        failures: list[Json] | None = None) -> None:
        if wire == 'anthropic':
            self.send_json(status, anthropic_error(error_type, message), failures)
        else:
            self.send_json(status, openai_error(message, error_type, None), failures)

    def stream_headers(self, failures: list[Json] | None) -> None:
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.send_header('cache-control', 'no-cache')
        # http.server does not chunk a response without a length; the close ends the
        # stream for the client.
        self.send_header('connection', 'close')
        self.failures_header(failures)
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode('utf-8'))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        """One client request body, or None when it is not one JSON object or too large."""
        length = self.headers.get('content-length')
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b''
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def reply(self, wire: str, result) -> None:
        """Send serve()'s answer: a JSON body, an SSE stream, or an error body."""
        status, kind, payload, failures = result
        if kind == 'sse':
            self.stream_headers(failures)
            try:
                for event in payload:
                    self.write_event(event)
            finally:
                pass
            return
        if kind == 'error':
            self.send_error_body(wire, status, payload['type'], payload['message'], failures)
            return
        self.send_json(status, payload, failures)

    # -- the routes ---------------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header the client sent."""
        listing = [{'id': m['id'], 'provider': m['provider'], 'display_name': m['display_name']}
                   for m in MODELS if m['free']]
        if self.headers.get('x-api-key') is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self) -> None:
        """POST /v1/chat/completions: an OpenAI-shaped client, answered by the best
        free model that meets the request, with failover and a token check."""
        body = self.read_json()
        if body is None:
            self.send_error_body('openai', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        result = serve('openai', body, {**body, 'model': 'auto'},
                       requirements_from_body(body), tokens_in_request(body))
        self.reply('openai', result)

    def messages(self) -> None:
        """POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions
        and answered back in the Messages wire."""
        body = self.read_json()
        if body is None:
            self.send_error_body('anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        result = serve('anthropic', body, anthropic_to_openai(body, 'auto'),
                       requirements_from_anthropic(body), tokens_in_request(body))
        self.reply('anthropic', result)


# ---------------------------------------------------------------- register and poll


def register(port: int, report: Json) -> None:
    """POST /register with the port and the one catalog; the env answers with the
    trial's prompt count."""
    body = json.dumps({'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': report},
                      separators=(',', ':'), ensure_ascii=False)
    request = urllib.request.Request(f'{ENV_URL}/register', data=body.encode('utf-8'),
                                     headers={'content-type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as err:
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode("utf-8", "replace")[:500]}') from err
    if not (200 <= status < 300):
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}')
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if is_record(reply) and is_number(reply.get('requests')):
        log(f'registered: the trial holds {reply["requests"]} prompts')


def poll() -> None:
    """Poll GET /work every 100 ms; exit 0 the moment it says done. A process still
    running when the wall ends scores zero, so never wait for anything else."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(f'{ENV_URL}/work', timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode('utf-8', 'replace'))
        except Exception:
            continue
        if is_record(work) and work.get('done') is True:
            sys.stderr.flush()
            os._exit(0)


# ---------------------------------------------------------------- start


def stop(*_: object) -> None:
    """SIGTERM and SIGINT end the trial as Node's process.exit(0) does: exit 0, at once."""
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    if '--list' in sys.argv[1:]:
        # The same document the register body carries, byte for byte: print the one
        # built object and stop before any server starts. A process whose stdout is
        # not flushed loses the tail, so flush it.
        sys.stdout.write(json.dumps(CATALOG, separators=(',', ':'), ensure_ascii=False) + '\n')
        sys.stdout.flush()
        return
    if ENV_URL == '':
        raise RuntimeError('MAKEFUN_ENV_URL is not set')
    if chat_completions_url() == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    server = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError('the OS picked 8080, which the env owns on one layout: restart the trial')
        log(f'listening on 0.0.0.0:{port} with {len([m for m in MODELS if m["free"]])} free models')
        register(port, CATALOG)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name='work-poll', daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:  # the top-level catch: log the stack and exit 1
        log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
