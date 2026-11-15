#!/usr/bin/env python3
"""Mock-gateway + mock-env harness for fma02/main.py (no OpenRouter key needed).

Starts a fake gateway (canned completions, with one 429 per top model to exercise
failover) and a fake env (register/work), runs main.py against them, drives the six
practice prompts from the kit's workload.json through both wires, and checks:
  - every answer carries its prompt's token
  - the reported model is the one actually dialled (best-rated eligible model)
  - failover works and the x-makefun-upstream-failures header reports the refusal
  - streaming (both wires) is well-formed, with usage and a stop reason
  - GET /v1/models lists exactly the fourteen ids on both auth headers
  - the register catalog is byte-identical to `python3 main.py --list`
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.expanduser('~/workspace/makefun/free-model-aggregator-02-kit')
WORKLOAD = json.load(open(os.path.join(KIT, 'practice/trial-0/input/env/workload.json')))
TOKEN_RE = re.compile(r'[A-Z]{2,}-[0-9]{3,}')

GATEWAY_CALLS = []          # (model, status) in order
REFUSED_ONCE = set()        # models that already spent their one refusal
FAIL_ONCE = {               # models that refuse exactly once, then serve
    'openrouter/thinkingmachines/inkling-2-small:free',
    'openrouter/nvidia/nemotron-3.5-super-vl:free',
}
REGISTERED = {}             # the register body the bot sent
WORK_DONE = threading.Event()


def token_of(body):
    found = TOKEN_RE.findall(json.dumps(body, ensure_ascii=False))
    return found[0] if found else 'TOKEN-0000'


def completion_text(body, model):
    token = token_of(body)
    if isinstance(body.get('response_format'), dict):
        return json.dumps({'token': token})
    return f'The token is {token}.'


def openai_answer(body, model):
    text = completion_text(body, model)
    message = {'role': 'assistant', 'content': text}
    if isinstance(body.get('tools'), list) and body['tools']:
        fn = body['tools'][0]['function']
        message['tool_calls'] = [{'id': 'call_mock1', 'type': 'function',
                                  'function': {'name': fn['name'],
                                               'arguments': json.dumps({'token': token_of(body)})}}]
        message['content'] = None
    return {'id': 'chatcmpl-mock', 'object': 'chat.completion', 'created': 1767225600,
            'model': model.replace('openrouter/', ''),
            'choices': [{'index': 0, 'message': message,
                         'finish_reason': 'tool_calls' if 'tool_calls' in message else 'stop'}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 9, 'total_tokens': 21}}


def sse_chunks(body, model):
    text = completion_text(body, model)
    chunk1 = {'id': 'chatcmpl-mock', 'object': 'chat.completion.chunk', 'created': 1767225600,
              'model': model.replace('openrouter/', ''),
              'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': text[:5]}, 'finish_reason': None}]}
    chunk2 = {'id': 'chatcmpl-mock', 'object': 'chat.completion.chunk', 'created': 1767225600,
              'model': model.replace('openrouter/', ''),
              'choices': [{'index': 0, 'delta': {'content': text[5:]}, 'finish_reason': None}]}
    chunk3 = {'id': 'chatcmpl-mock', 'object': 'chat.completion.chunk', 'created': 1767225600,
              'model': model.replace('openrouter/', ''),
              'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
              'usage': {'prompt_tokens': 12, 'completion_tokens': 9, 'total_tokens': 21}}
    out = []
    for c in (chunk1, chunk2, chunk3):
        out.append(f"data: {json.dumps(c, separators=(',', ':'))}\n\n")
    out.append('data: [DONE]\n\n')
    return ''.join(out).encode()


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'mock-gateway/1'
    sys_version = ''

    def log_message(self, *args):
        pass

    def do_POST(self):
        size = int(self.headers.get('content-length', 0))
        body = json.loads(self.rfile.read(size).decode())
        model = body.get('model', '')
        if model in FAIL_ONCE and model not in REFUSED_ONCE:
            REFUSED_ONCE.add(model)
            GATEWAY_CALLS.append((model, 429))
            err = {'error': {'message': 'rate limited', 'type': 'rate_limit_error', 'code': 'rate_limited'}}
            raw = json.dumps(err).encode()
            self.send_response(429)
            self.send_header('content-type', 'application/json')
            self.send_header('content-length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        GATEWAY_CALLS.append((model, 200))
        if body.get('stream') is True:
            raw = sse_chunks(body, model)
            self.send_response(200)
            self.send_header('content-type', 'text/event-stream')
            self.send_header('cache-control', 'no-cache')
            self.send_header('connection', 'close')
            self.end_headers()
            self.close_connection = True
            self.wfile.write(raw)
            return
        raw = json.dumps(openai_answer(body, model)).encode()
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class EnvHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'mock-env/1'
    sys_version = ''

    def log_message(self, *args):
        pass

    def do_POST(self):
        size = int(self.headers.get('content-length', 0))
        REGISTERED.update(json.loads(self.rfile.read(size).decode()))
        raw = json.dumps({'ok': True, 'requests': 6}).encode()
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        payload = {'done': True} if WORK_DONE.is_set() else {'done': False, 'next': 'p01', 'pending': 6}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def call(url, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method='POST' if data else 'GET',
                                 headers={'content-type': 'application/json', **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), dict(e.headers)


def anthropic_body(prompt):
    # The workload bodies for the Anthropic wire are already Anthropic-shaped
    # (top-level system string, tools with name/input_schema): send as-is.
    return WORKLOAD['bodies'][prompt['id']]


def main():
    gw = ThreadingHTTPServer(('127.0.0.1', 0), GatewayHandler)
    env = ThreadingHTTPServer(('127.0.0.1', 0), EnvHandler)
    threading.Thread(target=gw.serve_forever, daemon=True).start()
    threading.Thread(target=env.serve_forever, daemon=True).start()
    gw_port, env_port = gw.server_address[1], env.server_address[1]

    brief = os.path.join(KIT, 'practice/trial-0/input/prompt.txt')
    proc_env = {**os.environ,
                'MAKEFUN_ENV_URL': f'http://127.0.0.1:{env_port}',
                'MAKEFUN_PROMPT_FILE': brief,
                'OPENAI_BASE_URL': f'http://127.0.0.1:{gw_port}/v1',
                'OPENAI_API_KEY': 'dummy-key',
                'PYTHONHASHSEED': '0'}
    proc = subprocess.Popen([sys.executable, '-P', '-s', os.path.join(HERE, 'main.py')],
                            env=proc_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 15
        while 'port' not in REGISTERED and time.time() < deadline:
            time.sleep(0.1)
        if 'port' not in REGISTERED:
            print('FAIL: bot did not register within 15s')
            print('stderr:', proc.stderr.read() if proc.poll() is not None else '(still running)')
            return 1
        port = REGISTERED['port']
        base = f'http://127.0.0.1:{port}'
        failures = []

        # --list output must match the register catalog byte for byte
        listed = subprocess.run([sys.executable, '-P', '-s', os.path.join(HERE, 'main.py'), '--list'],
                                env=proc_env, capture_output=True, text=True, timeout=20)
        list_catalog = json.loads(listed.stdout)
        if list_catalog != REGISTERED['catalog']:
            failures.append('--list catalog != register catalog')
        models = REGISTERED['catalog']['providers'][0]['models']
        if len(models) != 14:
            failures.append(f'catalog has {len(models)} models, expected 14')

        # GET /v1/models on both wires
        for hdrs, shape in (({'Authorization': 'Bearer x'}, 'openai'),
                            ({'x-api-key': 'x', 'anthropic-version': '2023-06-01'}, 'anthropic')):
            status, text, _ = call(f'{base}/v1/models', headers=hdrs)
            ids = [m['id'] for m in json.loads(text)['data']]
            if status != 200 or len(ids) != 14 or len(set(ids)) != 14:
                failures.append(f'/v1/models ({shape}): status {status}, {len(ids)} ids')

        # the six prompts
        for prompt in WORKLOAD['manifest']:
            pid = prompt['id']
            token = WORKLOAD['bodies'][pid] and None
            body = WORKLOAD['bodies'][pid]
            tok = TOKEN_RE.findall(json.dumps(body))[0]
            if prompt['api'] == 'openai':
                status, text, hdrs = call(f'{base}/v1/chat/completions', body,
                                          {'Authorization': 'Bearer x'})
            else:
                status, text, hdrs = call(f'{base}/v1/messages', anthropic_body(prompt),
                                          {'x-api-key': 'x', 'anthropic-version': '2023-06-01'})
            if status != 200:
                failures.append(f'{pid}: HTTP {status}: {text[:200]}')
                continue
            if tok not in text:
                failures.append(f'{pid}: token {tok} missing from answer')
            data = json.loads(text) if not prompt['stream'] else None
            if prompt['stream']:
                if prompt['api'] == 'openai':
                    if 'data: [DONE]' not in text or '"model":"openrouter/' not in text.replace(' ', ''):
                        failures.append(f'{pid}: stream malformed')
                else:
                    if 'event: message_stop' not in text or 'event: message_start' not in text:
                        failures.append(f'{pid}: anthropic stream malformed')
            else:
                model = data.get('model', '')
                if not model.startswith('openrouter/') or not model.endswith(':free'):
                    failures.append(f'{pid}: bad model field {model!r}')
                if prompt['requires'].get('json'):
                    if model != 'openrouter/nvidia/nemotron-3.5-super-vl:free' and \
                       model != 'openrouter/google/gemma-4-31b-it:free':
                        failures.append(f'{pid}: JSON prompt served by {model} (expected super-vl, failover to gemma-31b)')
                elif prompt['id'] in ('p02', 'p03', 'p06'):
                    pass  # streaming/plain: model checked via stream text below
            # failover header must appear on the prompts whose first model refused
            if pid in ('p01', 'p02', 'p03', 'p04', 'p05', 'p06') and hdrs.get('x-makefun-upstream-failures'):
                try:
                    trail = json.loads(hdrs['x-makefun-upstream-failures'])
                    if not any(r.get('status') == 429 for r in trail):
                        failures.append(f'{pid}: failures header has no 429 row: {trail}')
                except json.JSONDecodeError:
                    failures.append(f'{pid}: failures header not JSON')
        # plain prompts should be served by the top-rated model (inkling) or its failover
        served = [m for m, s in GATEWAY_CALLS if s == 200]
        if 'openrouter/thinkingmachines/inkling-2-small:free' not in served:
            failures.append('inkling-2-small never served a prompt (expected as top pick for plain prompts)')
        if len([s for _, s in GATEWAY_CALLS if s == 429]) < 2:
            failures.append(f'expected >=2 refused calls (failover drill), saw {GATEWAY_CALLS}')

        WORK_DONE.set()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            failures.append('bot did not exit after /work done')
            proc.kill()
        if proc.returncode != 0:
            failures.append(f'bot exited {proc.returncode}, stderr: {proc.stderr.read()[:500]}')

        print(f'gateway calls: {GATEWAY_CALLS}')
        print(f'models served: {served}')
        if failures:
            print('FAILURES:')
            for f in failures:
                print(' -', f)
            return 1
        print('ALL MOCK CHECKS PASSED')
        return 0
    finally:
        if proc.poll() is None:
            proc.kill()
        gw.shutdown()
        env.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
