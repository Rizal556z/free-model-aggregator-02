# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Rizal556z/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Rizal556z/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A free-model aggregator: one long-running Python 3.13 server (standard library only)
that serves the OpenAI and Anthropic completion wires, dialling the best-rated free
OpenRouter model behind every answer — with failover to the next-best model on a
refusal, per-model recovery, and a check that every completion carries the prompt's
token before it is served.

## Run it

```sh
python3 main.py --list   # print the catalog the server registers
```

In a trial the server runs as `python3 -P -s /bot/main.py` with `MAKEFUN_ENV_URL`,
`MAKEFUN_PROMPT_FILE`, `OPENAI_BASE_URL` and `OPENAI_API_KEY` set by the harness. It
registers with the env, serves `GET /v1/models`, `POST /v1/chat/completions` and
`POST /v1/messages`, and exits 0 when the env reports the work done.

## Test it

```sh
python3 test_mock.py   # mock gateway + mock env, no OpenRouter key needed
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=8497f1e3&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
