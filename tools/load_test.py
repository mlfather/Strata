#!/usr/bin/env python3
"""Load test for a running Strata server: C clients at once, each streaming an answer, and what each one waited.

  STRATA_KEY=... python3 tools/load_test.py http://127.0.0.1:8080 --clients 4 --rounds 3 --max-tokens 256

Every client sends its own prompt (a different essay topic each, so the prompt cache does not join them), streams
`/v1/chat/completions` with thinking off and greedy sampling, and records when the first content token arrived,
how many tokens came and when the answer ended.  The report gives, per round and over all rounds: the median and
the slowest first token, the median tokens per second one client sees once it writes, and the total tokens per
second of the round (every client's tokens over the round's wall time).  That is the table docs/BATCHING.md
measures by hand; `--json FILE` keeps the raw numbers for an A/B of two settings.

Standard library only, like the other tools.  A client that gets an error (a 4xx, a dropped stream) is reported
as failed and leaves the round's totals.
"""
import argparse, json, os, statistics, sys, threading, time, urllib.error, urllib.request

TOPICS = [
    "the history of the printing press", "how a sailing ship tacks against the wind", "why bread rises",
    "the water cycle on a mountain", "how a bicycle stays upright", "the life of a honeybee colony",
    "what makes a violin sound like a violin", "how a lock and key work", "the rules of chess for a beginner",
    "how glass is made from sand", "why the sky is blue at noon and red at dusk", "how a steam engine turns heat into motion",
    "what a compiler does with a program", "how tides follow the moon", "the way a camera lens forms an image",
    "how coffee is grown and roasted",
]


def prompt_for(i: int, words: int) -> str:
    topic = TOPICS[i % len(TOPICS)]
    return (f"Write an essay of about {words} words about {topic}. Use plain language, several paragraphs, "
            f"and no headings. This is request number {i}.")


class Client:
    """One streamed request: timings and token counts, or the error that ended it."""

    def __init__(self, base: str, key: str, i: int, words: int, max_tokens: int, timeout: float):
        self.base, self.key, self.i, self.words, self.max_tokens, self.timeout = base, key, i, words, max_tokens, timeout
        self.start = self.first = self.end = None
        self.tokens = 0          # completion tokens as the server counts them (usage), else the deltas
        self.deltas = 0
        self.error = None
        self.finish = None

    def run(self):
        body = {"model": "strata", "stream": True, "temperature": 0, "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt_for(self.i, self.words)}],
                "chat_template_kwargs": {"enable_thinking": False},
                "stream_options": {"include_usage": True}}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers)
        self.start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if "error" in chunk:
                        self.error = str(chunk["error"])
                        break
                    usage = chunk.get("usage")
                    if usage and usage.get("completion_tokens") is not None:
                        self.tokens = int(usage["completion_tokens"])
                    for ch in chunk.get("choices") or []:
                        delta = ch.get("delta") or {}
                        if delta.get("content") or delta.get("reasoning_content"):
                            if self.first is None:
                                self.first = time.perf_counter()
                            self.deltas += 1
                        if ch.get("finish_reason"):
                            self.finish = ch["finish_reason"]
        except urllib.error.HTTPError as e:
            with e:
                self.error = f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'replace')}"
        except Exception as e:  # a dropped stream, a timeout: the client counts as failed
            self.error = f"{type(e).__name__}: {e}"
        self.end = time.perf_counter()
        if not self.tokens:
            self.tokens = self.deltas


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return s[k]


def summarize(clients, t0):
    """One round's numbers: first token (median, slowest), per-client tok/s (median), total tok/s, failures."""
    ok = [c for c in clients if c.error is None and c.first is not None and c.tokens > 0]
    ttft = [c.first - c.start for c in ok]
    rates = [c.tokens / (c.end - c.first) for c in ok if c.end > c.first]
    wall = max((c.end for c in ok), default=t0) - t0
    total = sum(c.tokens for c in ok)
    return {
        "clients": len(clients), "ok": len(ok), "failed": [c.error for c in clients if c.error],
        "ttft_median_s": round(statistics.median(ttft), 3) if ttft else None,
        "ttft_p95_s": round(pct(ttft, 95), 3) if ttft else None,
        "ttft_last_s": round(max(ttft), 3) if ttft else None,
        "client_tok_s_median": round(statistics.median(rates), 1) if rates else None,
        "client_tok_s_min": round(min(rates), 1) if rates else None,
        "total_tok_s": round(total / wall, 1) if wall > 0 and total else None,
        "tokens": total, "wall_s": round(wall, 2),
        "finish": sorted({c.finish for c in ok if c.finish}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base", help="the server, e.g. http://127.0.0.1:8080")
    ap.add_argument("--clients", type=int, default=4, help="requests sent at once per round (default 4)")
    ap.add_argument("--rounds", type=int, default=3, help="rounds; the first one warms the caches (default 3)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--words", type=int, default=800, help="the essay length the prompt asks for (default 800)")
    ap.add_argument("--timeout", type=float, default=600.0, help="per request, seconds (a long prompt read counts)")
    ap.add_argument("--key", default=os.environ.get("STRATA_KEY", ""), help="API key (or STRATA_KEY)")
    ap.add_argument("--json", default="", help="write every client's timings and the summaries to this file")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    rounds = []
    for r in range(a.rounds):
        clients = [Client(base, a.key, r * a.clients + i, a.words, a.max_tokens, a.timeout) for i in range(a.clients)]
        t0 = time.perf_counter()
        threads = [threading.Thread(target=c.run, daemon=True) for c in clients]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        s = summarize(clients, t0)
        rounds.append({"summary": s, "clients": [
            {"i": c.i, "ttft_s": round(c.first - c.start, 3) if c.first else None, "tokens": c.tokens,
             "tok_s": round(c.tokens / (c.end - c.first), 1) if c.first and c.end > c.first else None,
             "wall_s": round(c.end - c.start, 3) if c.end else None, "finish": c.finish, "error": c.error}
            for c in clients]})
        print(f"round {r + 1}/{a.rounds}: {s['ok']}/{s['clients']} ok, first token median {s['ttft_median_s']} s "
              f"/ last {s['ttft_last_s']} s, per client {s['client_tok_s_median']} tok/s (min "
              f"{s['client_tok_s_min']}), total {s['total_tok_s']} tok/s, {s['tokens']} tokens in {s['wall_s']} s"
              + (f", failed: {s['failed']}" if s["failed"] else ""), flush=True)
    measured = rounds[1:] if len(rounds) > 1 else rounds   # the first round warms up
    def med(key):
        v = [x["summary"][key] for x in measured if x["summary"][key] is not None]
        return round(statistics.median(v), 3) if v else None
    overall = {"clients": a.clients, "rounds_measured": len(measured), "ttft_median_s": med("ttft_median_s"),
               "ttft_last_s": med("ttft_last_s"), "client_tok_s_median": med("client_tok_s_median"),
               "total_tok_s": med("total_tok_s"),
               "failed": sum(len(x["summary"]["failed"]) for x in rounds)}
    print(f"\n{a.clients} clients, median of {len(measured)} measured round(s): first token {overall['ttft_median_s']} s "
          f"(last of the round {overall['ttft_last_s']} s), per client {overall['client_tok_s_median']} tok/s, "
          f"total {overall['total_tok_s']} tok/s" + (f", {overall['failed']} failed" if overall["failed"] else ""))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"args": vars(a), "overall": overall, "rounds": rounds}, f, indent=1)
        print(f"written: {a.json}")
    return 0 if overall["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
