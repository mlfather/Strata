# Strata engine review: what limits it today, and a plan for a faster engine that serves many requests at once

Written against commit 82f46a8 (October 2026). Every file reference is to that tree. Numbers quoted as measured come
from `docs/DETAILS.md`, `docs/BATCHING.md` and the paper, with the machine they were measured on; everything else in
this document is an expectation, said so, with the measurement that would confirm or refute it.

## 1. Summary

Strata is a specialised engine for one model (Qwen3.8-Flash-Next and its variants) on one class of machine (one
consumer GPU, lots of RAM, one CPU). Its design is sound and unusually well measured: the expert tiers, the exact
speculative verify window, the per-layer CUDA graphs and the conversation cache are the right ideas for this model,
and the project's habit of attaching a measurement to every claim is its greatest asset. On the reference PC (RTX
5070 12 GB, Ryzen 5 7600, 64 GB DDR5) it writes 93 tokens/s at 4K context with Q2_0 and reads 32K prompts at 2,650
tokens/s (engine 0.1.36).

The engine is built for one stream. Concurrency was added afterwards as "batch slots": N full copies of the
per-sequence state, a server-side state machine in Python that decides when a request runs alone, in a slot, or gives
way, and one text pipe between the two. It works and it is exact, but it caps rows at 8, costs 0.56 GiB of VRAM per
slot at 32K whether the slot is used or not, drops MTP drafts unless opted in, drops the penalties, captures a new
CUDA graph per row layout, reads prompts one at a time, and puts every scheduling decision one Python round trip
away from the GPU. On a 12 GB card the slots buy waiting time, not throughput (4 clients: 63 tokens/s together
against 71 one after the other), because every extra row routes to its own CPU experts.

Three defects in the batch mode of the server need fixing before anything else is built on it: a reproducible
deadlock in the yield path (two long prompts and one short one on `parallel: 2` hang every request; 6 of 6 trials
here), an engine `ERR` during an admission that holds the control lock for 300 s, and the absence of the silence
watchdog in batch mode. Section 3.3 has the details. On the engine side, the adaptive expert cache does not run
while slots decode, so concurrent conversations run on whatever the last solo request left in VRAM.

The report's recommendations, in order of value for the effort:

0. **Fix the three server defects above and add their tests** (a day's work, no engine change).
1. **Move scheduling into the engine.** An iteration-level scheduler in `generate.cpp`'s serve loop that owns
   admission, prefill slicing and preemption, with a framed protocol carrying request ids. The server becomes a
   translator. (Section 6, phase 1.)
2. **Make per-sequence state a pool, not a slot.** Paged K/V from one global page pool and a pool of GDN state
   blocks, so VRAM per sequence follows its real length (about 113 MB plus 13 KB per token at INT8) instead of a
   fixed 0.56 GiB. (Section 6, phase 2.)
3. **One graph per row count.** Read the row-to-sequence binding from a device table filled per window instead of
   baking slot pointers into the captured graph, so rows can carry drafts and sequences in any mix. (Section 6,
   phase 2.)
4. **A batch-aware expert tier.** Choose the PCIe share per window from rows per expert instead of a fixed
   fraction, and prefetch the next layer's likely experts. (Section 5 and 6, phase 3.)
5. **CI and a load generator before any of it.** There is no CI at all today; the server tests run in seconds
   without a GPU and should gate every PR, and a throughput/latency harness over HTTP is needed to measure the
   phases above. (Section 7.)
6. **Pay down the structure.** `generate.cpp` (11,102 lines), `server.py` (5,015), `setup.py` (300 KB), 358
   environment switches and a diverged 112k-line copy of the engine under `sycl/` are the main obstacles to
   contributions. (Section 7.)

## 2. What Strata is today

### 2.1 The model, in the engine's terms

- 48 layers: 36 Gated DeltaNet (GDN) layers with a fixed recurrent state, 12 Qwen Sparse Attention (QSA) layers
  with a paged K/V cache and an indexer that selects 2,048 positions (`include/strata/core/layout.hpp`,
  `include/strata/kernels/qsa.hpp`).
- Hyper-connections: the residual is 4 streams of 2,560 floats (`hc = 4`), read and written by every block through
  the gated-residual kernels (`fused_gr.cu`).
- 512 experts per layer, 10 routed per token plus a shared expert; 24,576 experts in all.
- A 28.8 GB n-gram embedding table (PLE) read from the SSD by the previous token ids.
- A one-layer MTP draft head, kept in VRAM with its own reduced vocabulary.

### 2.2 Where the state lives

| State | Size | Where |
| --- | ---: | --- |
| GDN recurrence, per layer | 128 x 48 x 128 floats = 3.1 MB | VRAM, 36 layers = 113 MB per sequence |
| QSA K/V, INT8, per layer | 2 heads x 256 x (1 B + scale) x 2 | about 13 KB per token over the 12 layers |
| QSA indexer pooled keys | (context / 4) x 128 floats | 4 MB at 32K |
| Conv history, PLE history, scratch | small | VRAM |
| One slot's whole session at 32K, INT8 | 0.56 GiB (measured, BATCHING.md) | VRAM, carved once at `max_cells` |

A session is carved once for `max_cells` (`session_init`, `src/core/session.cpp`), so its K/V is reserved for the
whole context whether or not the conversation gets there. KV streaming (`--kv-resident`) keeps only the attention's
window in VRAM and the rest in pinned RAM.

### 2.3 One decode window

The engine decodes in verify windows of T tokens (the last accepted token plus up to T-1 drafts, T at most 8,
`include/strata/core/verify.hpp`). One captured CUDA graph per T runs all 48 layers. In each layer the GPU runs the
mixer, the router and the shared expert, publishes the routed expert ids through a mapped pinned "doorbell", and
the host thread, spinning on it, splits the experts three ways: the ones in the VRAM cache (grouped kernels on the
GPU), a share of the misses copied over PCIe and computed on the GPU, and the rest computed by the CPU pool in
place in the pinned arena. The GPU waits on a flag for the CPU's rows, combines, and moves to the next layer. The
MTP head then drafts the next window while the commit graph advances the recurrent state.

Measured on the reference PC at 4K with Q2_0 (paper, Table 5): a window takes 34.3 ms, of which the host waits
14.6 ms for the GPU and the GPU waits 13.4 ms for the CPU's experts; 3.23 tokens are produced per window; 72% of
the routed experts are served from VRAM; the CPU streams expert weights at 41 GB/s. The two halves are the same
size, and they wait for each other at every layer, which is the single most important fact about this engine's
performance: no part can be sped up alone by more than about a third, and anything that overlaps the two halves
better is worth as much as a faster kernel.

### 2.4 Reading prompts

Prompts are read in chunks of up to 8,192 tokens (`--prefill auto`). For each layer the experts the chunk needs
and the cache does not hold are streamed over PCIe into a ring borrowed from the expert cache while the attention of
the previous layer runs, then multiplied with llama.cpp-style MMQ kernels or, for Q2_0 on RTX 30 and newer, fused
int8 tensor-core kernels (`src/prefill/`). Prompt speed is flat with context (2,650 tokens/s at 32K, 2,468 at 128K,
RTX 5070) because the sparse attention reads a bounded set of positions.

### 2.5 Concurrency as it exists: batch slots

`--batch N` gives the engine N extra sessions. A batch window holds one token of up to 8 sequences; the dense
weights, the shared expert and the head are read once for all rows, and the CPU pool computes the union of the
rows' missed experts (`Verifier::run_slot_rows`, `src/core/verify.cpp:2339`). The arithmetic of a row is the solo
window's, so greedy output is identical to the solo run (verified by `tools/batch_test.py`).

The server decides everything else (`StrataEngine.generate_batched`, `serve/server.py:1073`): a request alone
runs on the solo path with drafts; when a second one arrives the first is stopped and continued in a slot from the
prompt cache; a prompt is read on the solo path, one at a time, under a lock (`_take_control`,
`serve/server.py:1015`), with the slots decoding for half of each chunk's time between chunks
(`STRATA_BATCH_DECODE_SHARE`); a long read gives way at a chunk boundary to a prompt under half its length
(`BYIELD`); a request left alone goes back to the solo path (at most twice). All of it travels as text lines over
the engine's stdin and stdout.

What the design costs, in code rather than in the docs:

- A batch graph is captured per row layout: the slot pointers are kernel arguments at capture time
  (`Verifier::capture_batch`, `src/core/verify.cpp:2095`; `capture_commit_batch`, `:2127`), so every distinct
  (slots, order) combination is a new graph and a new commit graph, kept in `std::map`s with an optional LRU limit.
  With drafts per row (`--batch-mtp`) the number of layouts grows fast, which is why that mode is capped.
- Rows are at most 8 (`brow_[8]`, `b_out_[8]`, `last_rows_[8]` in `verify.hpp`).
- No drafts in batch windows by default; one draft per row with `--batch-mtp`, one GPU only.
- Penalties are not applied in batch windows (`set_slot_sampling`, `verify.hpp`).
- A slot is a full session at `max_cells`: 0.56 GiB at 32K whether idle or busy, more at longer contexts unless the
  K/V streams.
- Admission reads the prompt into the main session and then copies the state into the slot
  (`src/program/generate.cpp:9990`, "slot %d takes %lld tokens (copied in %.1f ms)"), one prompt at a time.
- Every scheduling decision (solo or slot, yield, go back solo) is made in Python after a round trip over the pipe,
  under a lock that serialises prompt reads.

Measured (BATCHING.md, RTX 5070, Q2_0, 32K): 4 clients with `parallel 4` get their first token after 1.0 s
(median) instead of 6.0 s, and decode 63 tokens/s together instead of 71 one after the other; a 4-row window takes
54 ms against about 20 ms for a 1-row window because it reads 24 CPU experts per layer against about 8. On a
4 x 16 GB layer split with most experts in VRAM, 8 clients reach 360 tokens/s against 120 for one, which says
where batching pays: wherever the experts fit.

## 3. Review findings

Severity is about what a user or a contributor runs into: **high** means wrong output, a crash, a hang or a cost
that grows with context on a common path; **medium** means a real but bounded cost, a race that has not been seen
to bite, or a trap for the next change; **low** is hygiene. Appendix A lists every item with its file and line.

### 3.1 The engine's decode and batch paths

**High**

- **Batch admission and the return to the solo path copy the whole K/V through pageable host memory, with device
  syncs per QSA layer.** `copy_to_slot` (`src/program/generate.cpp:7411-7445`) saves each QSA layer's K/V into a
  `std::vector` image, synchronises the device, and restores it into the slot; `copy_from_slot` (`:7449-7489`) is
  the mirror image. The cost is proportional to the context, and the single host thread does nothing else
  meanwhile, so every other slot stalls. BATCHING.md's "50-60 ms for a short conversation" is the short case; at
  32K-128K this is seconds, paid at every admission, every return to solo and every `BYIELD`.
- **One captured graph pair per row layout, never freed in the default batch mode.** `capture_batch` and
  `capture_commit_batch` (`src/core/verify.cpp:2095-2125`, `:2127-2216`) key the graphs on the exact row order;
  the rotating `next_slot` (`generate.cpp:7498-7522`) makes many orders out of the same set; each capture is a
  synchronous instantiate and upload on the serving thread, and the graphs' VRAM is released only in the
  destructor. The LRU limit exists only with `--batch-mtp` (`generate.cpp:6139`).
- **Any error in a batch window ends the process for every slot** (`generate.cpp:7537-7552`, `:7743`). There is
  no per-slot error isolation.
- **The adaptive expert cache is frozen while slots decode.** `batch_step` calls only `apply_pending(false)`
  (`generate.cpp:7529`); `adapt()` runs from the solo loop, the pipelined loop and the one-shot path only
  (`:9727`, `:9267`, `:10796`), and `--adapt-async` is refused with `--batch` (`:5446`). Usage counts keep
  growing undecayed during batched decode and act only when a solo request next runs. On a 12 GB card, where
  the cache serves half the experts, several concurrent conversations run on whatever the last solo request left
  in VRAM.
- **Unchecked CUDA calls on the window path**: the graph upload and sync in `capture_batch`
  (`verify.cpp:2119-2120`), the arena memset (`:514`), the penalty-history upload (`generate.cpp:9699`), the
  adaptive tier's event creation and record (`:6692`, `:6829`), two copies in `src/core/mtp.cpp` (`:290`,
  `:313`). A failed call is noticed, if at all, by a later sync with a misleading message.

**Medium**

- The batch commit is synchronous (`verify.cpp:2464`) while the solo commit records an event and lets the next
  window queue behind it (`:2026-2034`). With `--batch-mtp` each slot also does a synchronous device-to-device
  copy of two residual rows and a serial draft ending in its own stream sync (`generate.cpp:7583-7592`,
  `mtp.cpp:1298`).
- In batch windows the QSA K/V append, indexer append, block scores, top-k, resolve and attention are launched
  once per row (`verify.cpp:965-997`, `:1042-1054`); the dense projections, router, shared expert, experts and
  head are batched over the rows. That is 12 layers x 4 kernels x (rows - 1) extra launches per window.
- A `BSTOP` for a slot that is being admitted is lost (the admission resets the slot after the stop was set,
  `generate.cpp:9965` after `:7755` / `:8808`); a `BSTOP` otherwise takes effect at the next window and one more
  token is emitted before `BDONE cancel` (`:7566-7573`).
- `adapt()` runs on a freshly created `std::thread` every `adapt_every` windows (`:9727`, `:10796`) although a
  `JobThread` exists (`:1214-1286`); the blocking tier then waits on `cudaEventSynchronize` at the next window
  (`:6705`), which the code's own comment measures at about 4.6 ms per window on a PCIe 3.0 x8 card
  (`:1202-1206`) with the default lag of 1.
- The penalty-history copy is a pageable `cudaMemcpy` on the legacy stream while the sampler reads it on the
  verifier's non-blocking stream (`generate.cpp:9699`); ordering holds only because the window takes far longer
  than the copy.
- `BSlot.checks` keeps whole checkpoint copies per slot (`:9984-9989`, `:8835-8838`): up to N x ~118 MB of host
  RAM that no budget counts.
- The tier policy exists in five copies (serve `generate.cpp:6726-6852`, one-shot `:10553-10645`, async
  `a_choose` `:6900-6949`, `src/core/peer_experts.cpp:615-658`, `src/core/remote_expert_opt.cu:807-864`) with
  drifting filters; the decode loop exists three times (`generate.cpp:9637-9781`, `:9067-9636`,
  `:10690-10838`); `apply_pending` twice; the batch timing line twice (`:7600-7611`, `:7721-7732`).
- `main` is about 9,600 lines of reference-capturing lambdas over dozens of mutable locals, with function-static
  state inside lambdas acting as hidden globals (`next_slot` `:7498`, `logpos` `:8635`, `reread` `:8425`, `said`
  `:9062`, `equal_chunks` `:5103`).
- `drive_pool_multi` writes unit weights to the routing trace (`:956-957`), so a trace from `--serve` cannot feed a
  weighted profile.

**Low**

- `res_put` uploads the whole residency table with a synchronous copy and a legacy-stream sync on every lend,
  refill and apply (`:4933-4936`), including after every prompt chunk (`:8674`).
- A `BYIELD` that lands outside `read_part` is dropped (`:7758`) while the server has already reserved a slot
  (`serve/server.py:1122-1129`).
- The watchdog aborts from a detached thread while the main thread may be inside a driver call (`:7221-7227`);
  acceptable as a last resort and documented.
- A `GEN` while slots are active freezes the slots (the solo loop never calls `batch_step`); the server avoids it
  (`server.py:1106-1111`) but the engine does not enforce it.

### 3.2 The expert tiers

**High**

- **Data races under `--pipeline-windows`.** `adapt()` on its own thread writes `host_res` (`generate.cpp:6780`,
  `:6804`, `:6819`), decays the usage counts (`:6850`) and calls `blob()` (non-atomic counter,
  `src/core/expert_source.cpp:2251`) while the pool thread reads `host_res` (`:2468`, `:2484`) and writes usage
  (`:2444`); `commit_exchanges` from `pl_release` (`:9231-9237`) mutates `complement_offsets_` read by
  `resident_blob` on pool threads. Undefined behaviour formally; lost usage updates in practice.
- **DMA mode 0 ignores copy errors and still raises flag B** (`verify.cpp:1928-1929`, `:1939-1942`), so a failed
  upload would compute on stale staging bytes. Opt-in only (the default is the in-graph copy kernel), and the
  comment admits it.
- **An exchange whose bounds check fails is dropped but its override is cleared** (`expert_source.cpp:2169-2175`):
  the expert is then read from the file with no log line.
- **`std::abort()` in a serving process** on a rotation commit failure (`:2160-2164`, `:2198-2201`).

**Medium**

- The file tier creates up to 8 `std::thread`s per layer per window for its prefetch (`:1010-1019`,
  `:978-982`), unbounded across pipelined stages.
- `ExpertCache::residency_` goes stale after the first swap (the tier mutates only `host_res`); `slot_of()` is
  read by `pin_cache_complement`, `PeerExperts::open` and `RemoteExperts::open` (`expert_source.cpp:1667-1675`,
  `peer_experts.cpp:143`, `remote_experts.cpp:147`), correct only because they run at startup.
- The swap gain is a raw count difference: it ignores blob size (native blobs vary from 1.4 to 2.3 MB by layer) and
  counts a batch row 8 times for one blob read (`expert_source.cpp:2442-2444`, `generate.cpp:6746-6749`).
- The PCIe share takes the **last m misses in routing order** (`expert_source.cpp:2494-2504`), not the ones most
  worth copying; evicted experts in exchange buffers are never PCIe-eligible although the buffers are
  `cudaHostAlloc`'d (`:2255-2271`).
- `adapt_swaps` above 96 is silently capped by the exchange reservation (`generate.cpp:5497`, `:6884`);
  `ExpertCache::replace` has no bounds check (`expert_cache.hpp:148-152`).
- No NUMA policy: huge pages are handled (`src/core/pinned.cu:205-270`) but loader threads first-touch layers in
  `fetch_add` order (`pinned.cu:705`) while the pool pins workers per core; on a two-socket or chiplet-split
  machine the pages land on arbitrary nodes. Not measured in the repo.

### 3.3 The server

The Python layer itself is not slow: measured here on the mock engine (Python 3.13, 20,279 tokens), the whole
server path costs about 32 microseconds per token for one SSE stream and 33 for four streams together, so at the
engine's rates (8 slots at 50 tokens/s) it uses about 1% of one core. Nothing on the token path is quadratic. The
problems are in the batch-mode state machine and in what is missing for several clients.

**High**

- **A reproducible deadlock in the yield path.** A request holds the control lock while it waits for a free slot
  (`serve/server.py:1150-1161`), and a request that gave way holds its slot while it waits for the control lock
  (`:1162-1172`). With `parallel: 2`, two long prompts and one short one: the first long prompt yields into slot
  0 for the short one, the second long prompt wins the lock (a plain `Lock`, no priority for the shorter waiter),
  is admitted into slot 1, sees the short one waiting and yields too; the short one then takes the lock and finds
  both slots busy. Every request hangs until a client times out. Reproduced against the fake engine of
  `serve/test_parallel.py` in 6 of 6 trials (the script is a 60-line driver: start the service with two slots,
  send "long" x 600, "lung" x 600 and "short" 0.25 s apart, watch `ctl.locked()`, `slot_busy` and `waiting`).
  Three possible fixes: never wait for a slot while holding the lock; refuse a `BYIELD` that would leave no free
  slot; or give the shortest waiter the next turn explicitly instead of racing on the lock.
- **An `ERR` during a batch-mode admission or solo read holds the control lock for 300 s.** `_control` raises on
  an `ERR` line (`:950`) with the phase still "admit" or "solo"; the `finally` then drains for a `BADM` or `DONE`
  the engine never prints (a refused `BGEN` prints `ERR` and continues, `src/program/generate.cpp:7775-7778`;
  a bad `GEN` likewise, `:8173-8179`). The one-at-a-time path marks itself done on `ERR` (`:1397-1398`); the batch
  path does not. A runtime trigger exists today: "the K/V cannot grow" (`generate.cpp:8218`).
- **No silence watchdog in batch mode.** `engine_silence_s` (#481) is implemented in `generate()` (`:1343-1356`,
  `:1378-1383`, `:1403-1426`) and `session_file`, but `_control` and the slot loop only poll with 10 s heartbeats
  forever (`:909`, `:1214`). A hung engine with `parallel: N` keeps the lock and its slots indefinitely, and every
  later request waits because `alive()` stays true. No test covers it.

**Medium**

- Four of eight malformed request shapes drop the connection with no HTTP response (a traceback in the server
  window): an Anthropic message without `role` (`serve/frontend.py:434`), a content block that is a string
  (`:444`), `max_tokens: [5]` (`server.py:4206`), `chat_template_kwargs: [1]` (`frontend.py:335`). The
  `do_POST` ladder catches a fixed list of exceptions (`:3987-4008`); a catch-all mapping to a 400 or 500 JSON
  body fixes all four.
- `_release_slot_when_done` gives up after 600 s and marks the slot free (`:990-1008`) while the engine may still
  hold it; the next `BGEN` to that slot is refused, which lands in the `ERR` case above.
- No body-size cap or read timeout on `/v1/*` (`:3598`, handler timeout `None`); the 64 KiB cap applies to the
  control endpoints only (`:4035-4056`). Mitigated by the default loopback bind.
- With `parallel`, image encoding takes the `fifo` lock (`:2762`) but batch requests do not hold it (`:2913`), so
  the rule that the encoder must not run on the GPU during a request (`:2766-2770`) is not enforced there.
- `/v1/vram` can block `fifo` and then the control lock for up to 300 s each (`:2376-2404`).
- Head-of-line blocking by design: admissions are strictly one at a time; a long prompt blocks every new request
  until it finishes or yields (only to a prompt under half its length, at a chunk boundary, at most twice); the
  solo-to-slot promotion round trip (STOP, DONE, BGEN re-read from the cache, BADM) holds the lock before the
  newcomer is admitted.
- The pure-Python BPE tokenizer runs on the request thread under the GIL for the whole prompt on every request
  (`:4214`, `tools/strata_tokenizer.py:236-248`). It could not be timed here (no pack present); it is the most
  likely stall for concurrent streams on 100K-token agent prompts and should be measured first.

**Low**

- `totals["tool_calls_from_reasoning"]` is updated outside `status_lock` (`:3172`); an old request's `finally`
  running after a restart writes into the new slot lists (`:1283-1292` against `:653-657`); `EOS_IDS` is
  hard-coded for one vocabulary (`:524`); engine-wide `progress` and `status` show the newest request only in
  batch mode (`:2839-2850`, `:2930`); deeply nested JSON raises `RecursionError` instead of a 400 (`:3914`).

**Security.** The measures in place are solid: the DNS-rebinding `Host` check, the `Origin` check for keyless
browser POSTs, the own-page gate on the control endpoints, CORS only for configured origins, a constant-time key
compare, the control-body cap, slot file name validation, network-path refusal for images, a 32 MiB URL cap, a
whitelist for config edits and a sandboxed Jinja. Gaps: with an API key set the `Host` check is skipped entirely
(`:3637`); a non-browser client with no key configured can have any readable local file encoded as an image
(`Vision.load`, `:1665-1686`), which is local-file disclosure through the model when the server is bound to
`0.0.0.0` without a key; `cors_origins: ["*"]` without a key is allowed with only a warning (`:4906-4909`); and
the missing body limit above.

**What is missing for several clients** (beyond the engine): a queue with an order (both locks are plain and
polled every 0.5 s; the only ordering rule is `after_epoch`); per-client identity and limits (one shared key);
admission by budget (only the per-request context check at `:2790-2808`; no cap on concurrent prompt tokens,
total `max_new`, queue length or threads, with a listen backlog of 5); backpressure (unbounded queues at `:640`
and `:653`, blocking `sendall`); per-request metrics (history rows carry duration and engine ms but no first-token
or queue time, `:3109-3129`; those exist only in the opt-in API monitor); a graceful drain (shutdown cuts daemon
threads mid-stream, `:5002-5006`).

### 3.4 CPU pool, prefill and GPU kernels

**High**

- **Unchecked `cudaMemcpyAsync` on the prompt path's expert ring and in DMA mode** (`src/prefill/prefill.cpp:2003`,
  `:2050`, `:2931`, `:2937`; `src/core/verify.cpp:1929`). A refused copy leaves the previous expert's bytes in a
  slot whose "copied" event still fires: silently wrong experts. Only sticky errors surface at the end-of-prompt
  sync (`prefill.cpp:3317`).
- **Fatal exits inside kernel wrappers**: `std::exit(1)` on any CUDA error (`src/prefill/moe_fused.cu:34-39`,
  `moe_mmq.cu:16-21`, `src/kernels/cuda/iq_kernels.cu:27-30`, `qsa_decode_attn.cu:528-532`,
  `qsa_select.cu:1276`) and `std::abort()` in the pool (`src/kernels/cpu/pool.cpp:551`, `:578`, `:601`). The
  server restarts the engine, so the user sees "the engine stopped unexpectedly"; a transient allocation failure
  kills every request in flight. Deliberate (issue #29), but it closes the door to per-request error isolation.
- **Greedy output of the IQ packs depends on the draft grouping by default.** `STRATA_IQ_MT_MIN` defaults to 2
  (`src/kernels/cpu/native_expert.cpp:79-96`): a group of one token uses ggml's dot product, two or more use the
  multi-token kernel, and the two round differently. DETAILS.md documents the opt-in (`=1`, -1 to -3% on IQ3_S),
  but the window's own contract says "bit for bit" (`verify.hpp:3-8`), and the batch path inherits it.

**Medium**

- The pool keeps a post-phase re-park barrier (`pool.cpp:693`, `:833`) that the epoch-tagged claim protocol made
  redundant (`pool.hpp:14-27`); it now only exposes sleeper wake-up latency after the 20 ms idle that a large
  VRAM cache makes common. The `pool phases ... repark` line (`generate.cpp:11030-11033`) measures it.
- Host-serial work while the workers idle: per-token activation quantization (`src/core/expert_source.cpp:
  2582-2590`), an O(n^2) distinct-expert scan and a `std::find` miss list (`:2461-2471`, `:2594-2599`), and the
  intermediate quantization between the gate/up and down phases (`pool.cpp:738-740`). All three are timed
  (`ms_actq`, `ms_plan`, `ms_multi_q`) and none is parallel.
- Allocation on the token path: `run_split_multi`'s fallback builds a `std::vector<ExpertJob>`
  (`pool.cpp:718-726`) against the rule in `expert.hpp:80-81`. Every `ActQ` carries the AVX2-only bit-plane
  image (`expert.hpp:74-77`) that AVX-512 machines never read, so 8 rows of activations (59 KB) overflow a 48 KB
  L1D.
- The down-row dispatchers split more than 4 tokens into chunks of 4 and decode the weights again per chunk
  (`iq_avx512.cpp:236-240`, `iq_avx2_rows.inl:202-206`, `q2_avx2_rows.inl:58-62`) while gate/up handles 8.
- The MMQ prompt path (the default for the native packs) synchronises the stream once per MoE layer to sort the
  routing on the host (`prefill.cpp:2589-2595`): 48 syncs per chunk; the fused path groups on the GPU.
- The GGUF-quantized dense weights are dequantized to an FP16 scratch before every product
  (`include/strata/prefill/gemm.hpp:362-366`, `src/prefill/gemm.cu:773-809`), on every chunk.
- Three unused kernels are compiled into the engine library (`s2_gemv.cu`, `s2_gemv_quads.cu`, `s2_gemv_fast.cu`,
  `CMakeLists.txt:309-312`; only the parity test calls them, and `s2_gemv_fast.cu:3` says "NOT FASTER. DO NOT
  ADOPT"). The same holds for the losing arms of several A/B families: five `gr_down` variants
  (`fused_gr.cu:856-919`, `:1587-1708`), four MMVQ layouts, seven native gate/up families
  (`iq_kernels.cu:2669-2681`, `:3429-3457`). Every loaded kernel costs VRAM under eager module loading
  (`CMakeLists.txt:73-75`).
- 251 distinct `STRATA_*` variables are read through `getenv` in the engine (74 in `generate.cpp`, 53 in
  `prefill.cpp`, 26 in `verify.cpp`), most as function-local `static const bool` in hot paths
  (`prefill.cpp:2021`, `:2088`, `:2954`; `expert_source.cpp:2570-2573`), so behaviour is fixed by the first
  call's environment and cannot be tested in-process.

**Low**

- A namespace-scope `static const bool kZmm = getenv(...)` in an AVX-512 translation unit (`expert.cpp:153`)
  runs before `cpu_require_expert_support()` (`generate.cpp:2371`), the hazard class fixed for #391 in
  `iq_avx2.cpp:82-88`; `cpu_features()` (`expert.cpp:352-373`) checks CPUID only while `cpu_avx512_ok()`
  (`expert_layout.cpp:40-71`) checks OSXSAVE and XCR0 correctly. The prefill issuer thread busy-yields for the
  whole chunk (`prefill.cpp:2040-2043`).

No data race was found in the pool's claim protocol (`pool.cpp:537-560`, four cache-line-separated atomics) or in
the prefill issuer/consumer pair (`:2040`, `:2061`, `:2073`, acquire/release).

### 3.5 Build, tests, CI and structure

**High**

- **There is no CI.** `.github/` holds `FUNDING.yml` only; no formatter, linter, type checker, pre-commit or
  sanitizer configuration exists anywhere. The GPU-less tests run in minutes (below) and would have caught at
  least one recent breakage (commit 1cbcacb, "nvcc on Linux rejects its std::array locals", is a compile error a
  compile-only CUDA job sees without a GPU).
- **The default CUDA architecture is forced to sm_120** (`CMakeLists.txt:152-154`): a contributor who configures
  without `-DCMAKE_CUDA_ARCHITECTURES` gets a binary for RTX 50 only. The Dockerfile builds `75;80;86;89;120`
  (`Dockerfile:58`) while INSTALL.md says the default covers 80/86/89/120.
- **Setup's supply chain trusts what it downloads.** When a pinned Hugging Face revision answers 404 the download
  falls back to the repository's current `main` with only a warning (`setup.py:1177-1180`), a silent downgrade of
  the pin. Only the Unsloth shards carry SHA-256 hashes (`:155-176`, verified at `:4704-4705`); the other families
  are checked by reading their GGUF directories. The prebuilt engine zips (`get_prebuilt`, `:2313-2400`;
  `get_prebuilt_hip`, `:1983-2055`) and the llama.cpp archive (`:1389-1420`) are downloaded over HTTPS and
  extracted with no checksum or signature, then executed; the CUDA keyring `.deb` is installed with `sudo dpkg
  -i` without a checksum (`:2560-2565`). On the positive side there is no `shell=True`, `os.system`, `eval` or
  `exec` anywhere in `setup.py`, every subprocess is an argv list, and the Python dependencies are fully pinned.

**Medium**

- `STRATA_BUILD_TESTS` defaults to off because `tests/CMakeLists.txt` and `bench/micro/` are not in the published
  tree (`CMakeLists.txt:94-101`, confirmed absent); references to them, to Catch2 and to private fixtures are dead
  (`:201`, `:878`, `:919`, `:955`, `:967`, `:974`, `:1366`, `:1426-1430`). Conversely, about 45 parity
  executables are built and registered on every GPU build with no test guard (`:634-917`), and three registered
  tests can never pass without private fixtures (`ple_parity`, `expert_parity`, `pool_test`;
  `:878-882`, `:1399-1410`). AMD_HIP.md's "42 of 46" style counts are this effect.
- `pytest tools/` aborts collection: `tools/test_iq_pack.py` imports `tools/_paths.py`, which calls `sys.exit`
  when llama.cpp's gguf-py is absent (`_paths.py:21-22`), and pytest also collects `*_test.py`, so
  `tools/early_close_test.py` POSTs to a live server at import. With `STRATA_GGUF_PY` set and the right file
  glob the suite passes (459 tests in 15 s).
- The llama.cpp pin is written in three places (`CMakeLists.txt:1125`, `setup.py:93`,
  `third_party/ggml/VERSION.txt`) plus kernel headers; CMake fetches it with `GIT_SHALLOW FALSE` (a 621 MB
  clone per fresh build directory) while setup unzips the archive of the same commit.
- `sycl/` is a second 120k-line copy of the engine, not a backend: all 51 files it shares with `src/` differ,
  58 are SYCL-only, and its `generate.cpp` (8,380 lines) is 5,800 lines apart from the CUDA one. A hand-maintained
  three-way merge script (`sycl/tools/merge_upstream.py`) is the only bridge.
- 164 engine flags are parsed and 122 documented in `usage()`: 26 are parse-only, including ones setup writes
  (`--pcie-frac`, `--pcie-mode`, `--vram-reserve-mib`, `--prompt-cache-root`, `--resident-budget-gib`,
  `--split-device`, `--eos-ids`, `--stop-eos`, `--no-fast-attn`).
- The code's cross-references point at 19 documents that are not in the repository
  (`docs/activation-contract.md`, 16 references; `docs/semantics.md`, 11; `docs/pack-format.md`, 8;
  `docs/capture-format.md`, 5; `docs/kv-streaming-design.md`, 4; `Memory/LEDGER.md`, `phase-2-correct-engine.md`,
  `aurora_s23.md` and others). A contributor cannot resolve "LEDGER L54" (`s_gemv.cu:70`) or "round 309"
  (`sampler.cu:89`).
- Stale statements: the CMake header (`CMakeLists.txt:1-9`: "CUDA targets arrive in Phase 2", "sm_80 or newer"),
  `generate.cpp:14-15` ("IT IS PHASE 2, so hit rate is h = 0"), README's RX 6800/6900 as supported while
  `cmake/hip_backend.cmake:29` warns "not validated on a real card yet", `sycl/CMakeLists.txt:13` tracking 0.1.39
  while INTEL_ARC.md has the 0.1.40 fixes, `.gitignore` and `setup.py:97` citing tools that do not exist
  (`tools/make_tiny_model.py`, `tools/make_release.py`, `tools/gpu_session.py`).

**Low**

- `CMakeLists.txt:1` starts with a UTF-8 BOM and line 3 contains mojibake; `tools/strata_tokenizer.py`, "the
  reference implementation of record", has no test file.

**What is good and should be kept**: the measured-numbers rule everywhere (README, docs, CMake and kernel
comments, `bench/results/*/README.md` with hardware tables and "limits" sections); parity tests against a
validated reference rather than against a copy of the kernel, and negative results kept on record; tests that
need a model fail rather than skip, with exit code 77 reserved for a real skip; pinned requirements, Hugging Face
revisions and the llama.cpp commit; the setup golden corpus (25 PCs) and its "recommends, never forces" rule;
fail-closed session file parsing; the constant-time key compare; issue numbers in comments; AGENTS.md's docs
style rule.

## 4. Where the time goes, and what is structural

**The chain is serial inside a window.** Layers run strictly in order on one stream (`verify.cpp:1328-1334`);
in each layer the only GPU work that overlaps the CPU pool is that layer's own VRAM hits, PCIe share and shared
expert. The paper measured the consequence: on the reference PC the host waits 14.6 ms per window for the GPU and
the GPU waits 13.4 ms for the CPU's experts (Table 5). Splitting the window into two token groups to overlap one
group's CPU work with the other's GPU work was tried and measured 7% slower (paper, finding 5): the dense weights
are read twice. The overlap that paid was the copy engine beside both (finding 9).

**The host thread is in the loop 48 times per window.** The GPU rings the doorbell at every layer even when all
ten routed experts are resident (`doorbell_publish_res` only skips copying the activation,
`src/kernels/cuda/elementwise.cu:314-337`), and the host spins on the mapped ring (`verify.cpp:1743-1770`),
plans, dispatches, and raises a flag the GPU is spinning on. Any host stall (an admission copy, a checkpoint
save, a graph capture, a `printf` that blocks on a full pipe) stalls the GPU at the next layer. The device-side
plan for all-resident layers (`STRATA_VERIFY_DEVICE_PLAN`, `verify.cpp:589-595`, `:1253-1279`) exists and is off
by default as "neutral on 1-2 GPUs"; with batch rows the host serves 2-8x the entries per layer, so it is worth
measuring again there.

**Between windows, nothing overlaps.** The next window's launch waits for the draft's sync (`mtp.cpp:1298`) and
is stream-ordered behind the commit. Tokens are written only after the whole window and the commit launch, one
unbuffered `write(2)` per token (`generate.cpp:9739-9747`, stdout unbuffered at `:1507`), and the draft is
launched after those writes (`:9754`).

**The GPU is launch-bound at T = 1-4.** `session.hpp` records the measurement that justified the per-layer graphs
(a 43-node block replayed in 1.585 ms against 2.393 ms of direct launches); the window graph removes the launch
cost, but the per-row attention launches in batch windows (section 3.1) bring part of it back.

**The CPU is bandwidth-bound for Q2_0 and arithmetic-bound for the i-quants** (paper, finding 7: 41 GB/s against
RAM's 42-52 GB/s for Q2_0; 23-26 GB/s on six cores for the i-quants). With S rows the pool computes the union of
the rows' misses, so its time grows with rows: 24 CPU experts per layer at 4 rows against 8 (BATCHING.md).

**Prompts are bound by the expert stream and the kernels, not by the attention.** Prompt speed is flat from 32K
to 128K (2,653 against 2,468 tokens/s). The fused int8 path lifted Q2_0 by 16-22% (0.1.36); the native packs
still run MMQ by default.

## 5. Taking the single stream to the next level

Ordered by expected value for the effort. "Expected" means not measured here; each item names what to measure.

1. **Make the batch commit asynchronous** as the solo commit already is (record an event at `verify.cpp:2464` the
   way `commit()` does at `:2026-2034`). Expected: the per-slot drafts, the token emission and the next window's
   staging overlap the commit graph. Measure: `strata batch:` ms per window at 2, 4 and 8 rows.
2. **Copy slot state device-to-device.** Replace the host round trip in `copy_to_slot` / `copy_from_slot` with
   `cudaMemcpyAsync` of the first `upto` cells of each pool on the verifier stream (`generate.cpp:7417-7431`).
   Expected: admission cost from seconds at long contexts to milliseconds, and `BYIELD` cheap enough to use at
   every chunk. Measure: the "slot %d takes %lld tokens (copied in %.1f ms)" line at 4K, 32K and 128K.
3. **Canonicalise the batch row layout** (sort rows by slot id; `generate.cpp:7498-7522`) so the graph key depends
   only on the active set, and bound the captured graphs in every batch mode (the LRU at `verify.cpp:2255-2280`
   exists). This is the stop-gap before the device row table of section 6.4.
4. **Batch the per-row attention launches in batch windows**: the block scores, top-k, resolve and decode
   attention already take `n` rows for one pool (`verify.cpp:1056-1066`); a pointer-array variant over per-row
   pools removes 12 x 4 x (rows - 1) launches per window, and the same for the K/V and indexer appends
   (`:965-997`). Measure: GPU stage times with `STRATA_VERIFY_PROFILE=1`.
5. **Let the adaptive tier run during batched decode**: call `adapt()` from `batch_step` between windows under the
   blocking tier (same invariant as the solo loop: no window in flight), and count distinct (window, expert)
   pairs weighted by blob bytes instead of entries. Expected: the VRAM hit rate of concurrent conversations rises
   toward the solo path's 0.72 instead of staying at the profile's 0.50. Measure: `hit_rate` in `/metrics` with
   `tools/load_test.py` at C = 4.
6. **Take the adaptive tier's wait off the critical path**: `STRATA_ADAPT_LAG=2` is already measured in the code's
   comment (`:1202-1206`), or promote the async tier; reuse the `JobThread` instead of a thread per round.
   Expected: about 1 ms per window averaged over `adapt_every = 4` on PCIe 3.0 cards.
7. **Launch the draft before the token writes, and write a window's tokens in one `fwrite`** (`:9739-9755`).
   Expected: up to 8 `write(2)` calls overlapped with the drafter's GPU time; small, free.
8. **Fold host-side sampling into the window graph**: a sampled or penalised request today runs `sample_tokens`
   again with a second stream sync (`verify.cpp:1846-1855`); the drafter already reads its parameters from
   mapped memory inside its graph (`mtp.cpp:1026`). Also move the penalty rows to mapped pinned memory copied
   inside the graph (`generate.cpp:9699`).
9. **Draft steps with `spec_min_p > 0`**: the one-step-at-a-time loop with a host spin per draft
   (`mtp.cpp:1307-1345`) can use the launch-all-and-poll branch (`:1273-1306`) and truncate below the threshold on
   the host.
10. **Smarter PCIe pick**: choose the misses with the most rows (batch) or the largest blobs rather than the last m
    in routing order (`expert_source.cpp:2494-2504`), and give the exchange buffers a device alias so evicted
    experts stay PCIe-eligible (`:2255-2271`).
11. **Next-layer expert prefetch.** `RouterLookahead` (`expert_source.cpp:1423-1530`) already applies layer l+1's
    router to layer l's input on a thread, per token, and reports its accuracy as warmed hits; today it only warms
    file pages and only in the GGUF-in-place tier. Driving the copy engine from it into a second staging bank
    (`kStagingBlobs`, `verify.hpp:431`) during layer l would widen the share the copy engine can take, which the
    paper names as the next step. Measure first: the lookahead's top-10 accuracy on a routing trace, then ms per
    window.
12. **Exchange rotation on by default where eligible** (+8.1% decode in one A/B, EXCHANGE_ROTATION.md) after the
    AMD and Windows validation it still lacks.
13. **Async residency upload**: `res_put` as a `cudaMemcpyAsync` on the adapt stream with a stream wait on the
    verifier's (`generate.cpp:4933-4936`), removing a device sync per apply and per prompt chunk refill.
14. **Startup fill**: `fill_slot_blocking` per expert (`generate.cpp:4113-4115`, up to 24,576 synchronous copies)
    as queued copies with periodic syncs, as the lent refill already does (`:8668-8680`).
15. **Slot sessions without the unused scratch and with a shared rope table** (`src/core/session.cpp:98-127`,
    `share_rope` in `qsa_state_init`): the VRAM goes back to the expert cache, which BATCHING.md shows is what the
    slots cost in speed (64 MiB per slot at 262K for the rope table alone).

**The CPU pool**

16. **Drop the post-phase re-park barrier** (`src/kernels/cpu/pool.cpp:693`, `:833`) and let sleepers rejoin
    lazily; keep the parked count for diagnostics. Measure: the `pool phases ... repark` line
    (`generate.cpp:11030-11033`) and `pool_stress`.
17. **Quantize the intermediates inside the gate/up tasks**: align the task ranges to 32-row chunks
    (`pool.cpp:734-736`, `:665-668`) and call `act_quant_q8_1` per chunk, which removes the serial step at
    `:738-740`. Measure: `ms_multi_q` (`generate.cpp:10876`).
18. **Parallelize the per-token activation quantization and build the distinct list from `job_of`** instead of
    the O(n^2) scan (`expert_source.cpp:2461-2471`, `:2582-2590`). Measure: `ms_plan` and `ms_actq`
    (`generate.cpp:10881`). Both grow with rows, so they matter more for batches.
19. **Software prefetch in the Q2_0 row kernels** (`expert.cpp:174-221`, none today), as the IQ kernels do
    (`iq_avx512.cpp:36-39`, `:164-168`: 2-3% there, 4% on AVX2 gate/up). Measure: `pool multi` MB/ms
    (`generate.cpp:10877`).
20. **Guarantee 2 MB pages for the arena** (hugetlb or THP, `pinned.cu:209-268`): the kernels record about 30
    against 44 GB/s with 4 KB pages (`iq_avx512.cpp:35`). Setup could reserve `vm.nr_hugepages` on Linux and
    say when it could not.
21. **AVX-512 VNNI in the i-quant kernels**: `iq_avx512.cpp:178` does `maddubs` plus `madd`; for the formats
    whose scale is uniform per 32 values (IQ2_XXS, IQ3_XXS, IQ3_S) `vpdpbusd` halves the integer work in a kernel
    that is decode-bound. Measure with the `native_expert_parity` bench. AMX is not a fit: the decode path's B
    operand is at most 8 columns and the i-quant cost is codebook decode, not multiplies. Lookup-table kernels
    in the T-MAC style (the paper's suggestion) remain the larger, riskier lever for the i-quants.
22. **NUMA** (two-socket or large chiplet machines only): per-node arena slices with node-local worker
    assignment; nothing exists today and nothing is measured. Measure per-node bandwidth with the `pool multi`
    counter under `numactl`.

**The prompt path**

23. **Group the MMQ path's routing on the GPU** with the fused path's count/scan/place kernels
    (`src/prefill/moe_fused.cu:108-176`), removing the 48 stream syncs per chunk (`prefill.cpp:2595`) on the
    native packs. Measure: the "host: waiting for each chunk" line (`:3343-3345`).
24. **Int8 tensor-core dense projections** (cuBLASLt IMMA with per-row-quantized activations) instead of
    dequantizing the GGUF dense weights to FP16 before every product (`gemm.cu:773-809`). It changes the
    numerics (activations are BF16-rounded today, `gemm.hpp:325`), so opt-in, sized first from the HC/GDN/QSA
    shares of `STRATA_PREFILL_TIMING`.
25. **Turn `STRATA_KV_PREFETCH=1` on from 64K** (`prefill.cpp:1695-1699`, `:1881-1907`): the measured 4-5% at
    32K-64K (KV_PREFETCH.md) waits only for validation on CUDA and HIP.
26. **Bigger chunks where VRAM allows**: throughput is proportional to the chunk until the VRAM limit
    (`bytes_needed_impl`, `:1486-1541`, about 680 KB per token); a 16K chunk on 16 GB and larger cards is the
    single largest prompt lever and needs no new kernel.

**The decode kernels**

27. **Fuse the per-layer handshake** (`verify.cpp:1254-1291`: three one-thread spin kernels, two mapped copies and
    an add per layer group) into one kernel per group: about 6 fewer graph nodes per layer group, about 600 per
    window. Measure with `STRATA_VERIFY_PROFILE` stamps 19-24.
28. **Time the upstream multi-column MMVQ layout** (`native_mmvq.cu:1147`, `:1231-1235`): the code keeps the
    exact single-column reduction shape "until the upstream layout is timed"; `mmvq_multi_parity` can time it.
29. **GDN step without barriers**: the prompt path's 4-lanes-per-column recurrence (`src/prefill/kernels.cu:
    714-745`) has no block barriers; the decode step does four per token on 48 blocks
    (`verify_kernels.cu:191-289`). `gdn_rec_parity` checks a port.
30. **Quantize the MoE input once** for the shared expert and the hit kernels (`verify.cpp:1185-1206`;
    `STRATA_VERIFY_QDEDUP` covers the native packs only today).

## 6. A design for concurrency

### 6.1 Goals and non-goals

Goals, in this order: (1) keep exactness: a sequence's greedy tokens do not depend on who else is running;
(2) first token under a second for a short prompt while others decode; (3) throughput that scales with rows on
cards that hold most experts, and degrades gracefully on cards that do not; (4) VRAM per sequence that follows its
real length; (5) the server stays a thin, crash-isolated front.

Non-goals: tensor parallelism over PCIe (the dense part is 3.5 GB per window and the link is 26 GB/s: it would
not pay), and a general-purpose engine for other models.

### 6.2 The scheduler belongs in the engine

Today the server is the scheduler and the engine is a worker with a few verbs (GEN, BGEN, STOP, BSTOP, BYIELD). The
proposal inverts that:

- The engine keeps a table of requests: id, token ids and images, sampling, `max_new`, state (queued, reading,
  decoding, done), position, the handles of its per-sequence state (section 6.3), and its draft state.
- One loop iteration = one window. The scheduler builds the window from (a) every decoding sequence (up to the row
  limit) with its drafts, and (b) a prefill budget in tokens for queued and reading sequences, chosen so that the
  decoding rows' time per token stays within a target (for example: no more than one prefill slice of S tokens
  between windows, S adapted from the measured window time). This replaces the fixed "half the chunk's time" rule.
- Preemption is a scheduler decision: a long prompt gives way whenever a shorter one is queued and the budget says
  so, without a server round trip; the part read so far stays in the sequence's own state, so there is nothing to
  copy into a slot.
- Finished sequences keep their state as a conversation cache entry (as a finished slot does today) until the pool
  needs the memory; parking to pinned RAM is what `conversation_snapshot.cpp` already does.

The protocol becomes framed messages with a request id on each: `SUBMIT id params ids`, `CANCEL id`, and from the
engine `TOKENS id n t1..tn`, `PROGRESS id read total`, `DONE id reason stats`. The server no longer needs to know
about slots at all. Keeping the engine in its own process keeps the crash isolation that `server.py` relies on; a
Unix socket or named pipe with a length-prefixed binary framing avoids the per-line parsing and lets one frame carry
every row's token for a window.

### 6.3 Per-sequence state as pools

- **K/V pages.** The QSA layers already address the K/V through a page table with 4-cell pages
  (`QsaState::page_table`, `include/strata/core/layer.hpp:209`). Making the page pool global (one per layer,
  sized for the sum of the contexts the card can hold) and the page table per sequence turns a slot's fixed
  reservation into use-proportional allocation, and gives prefix sharing between sequences for free (two
  conversations with the same system prompt share pages, copy-on-write at the first divergent page). The page size
  should grow to 16-64 cells so the table stays small at 262K; the attention kernels read the table already, so
  this is a change to allocation, not to arithmetic. KV streaming (mode 1) keeps its per-sequence host copy.
- **GDN state blocks.** 113 MB per sequence in fp32 is the irreducible cost of an active sequence; a pool of R
  blocks for the R sequences that can decode at once, with park and restore to pinned RAM for the rest, bounds it
  at R x 113 MB. Restoring a parked state is what the conversation cache measures at 50-60 ms for a short
  conversation (BATCHING.md).
- **Indexer, conv history, PLE history, draft state**: per sequence, small, pooled the same way.
- **The expert cache, the dense weights, the rope tables, the scratch buffers**: shared, as now.

The expected effect is VRAM per decoding sequence of about 113 MB plus 13 KB per token of context at INT8 instead
of 0.56 GiB at 32K, so 8 sequences averaging 8K tokens take about 1.8 GB instead of 4.5 GB. That is not measured;
it follows from the sizes in `session_bytes` (`src/core/session.cpp`) and `qsa_state_bytes`
(`src/core/layer.cpp`), and the first milestone of phase 2 is to measure it.

### 6.4 One graph per row count

The window graph should take its row binding from a small device table written before each launch: for row r, the
sequence's state base pointers (or pool indices), its position, its draft index. Every kernel that today receives a
slot's pointer as an argument indexes that table instead. Then there is one captured graph per S (as there is one
per T on the solo path) and rows can be any mix of sequences and drafts: 4 sequences with 4 drafts each in a
16-row window, or 16 sequences with no drafts, from the same graph. `commit_slot_prefixes` already commits an
accepted prefix per sequence group; it generalises to this binding.

Raising the row limit from 8 to 16 or 32 asks for row-aware dense kernels: at T <= 8 the dense projections are
bandwidth-bound GEMVs (`native_mmvq.cu`); from about 8 rows the int8 tensor-core path that the prompt experts use
(`moe_fused.cu`) should take over for the dense projections too. The kernel choice can be made per window from S.

### 6.5 A batch-aware expert tier

With S rows a layer routes to the union of their experts: measured 24 CPU experts per layer at 4 rows against 8
for 1 (BATCHING.md). Three levers, all measurable with the existing `--dump-routing` traces and
`STRATA_DECODE_TIMING=1`:

1. **Cost-based PCIe share.** Today `--pcie-frac` is a fixed fraction of the misses. An expert copied over PCIe is
   paid once and serves every row that routes to it; the CPU pays per row (bandwidth for the bytes once, arithmetic
   per row for the i-quants). The window planner (the plan sink in `verify.cpp`) knows rows per expert, so it can
   send the experts with the most rows over PCIe first and stop when the copy engine's time for the layer would
   exceed the CPU's. Expected: the CPU share of a batch window falls with rows; to be measured as ms per window
   against rows.
2. **Next-layer prefetch.** The paper names it as the next step: predict layer l+1's experts from layer l's state
   and start their copies before the router runs. The cheapest predictor needs no training: the same sequence's
   routing at layer l+1 on its previous token. Its hit rate is a one-line measurement over a routing trace
   (`tools/make_profile.py` reads them); if it is high, the copy engine runs a layer ahead and the share it can
   take grows. With several sequences the prefetch is per row, so the benefit scales.
3. **Admission policy.** The adaptive tier counts routing and swaps the most-used experts in. With several
   clients, one chatty client's experts can push out everyone else's; a per-sequence weight (or decayed counts
   per sequence) keeps the cache fair. Only worth doing once the above is measured.

### 6.6 The server

With the engine scheduling, `server.py` loses `generate_batched`, `_take_control`, the slot bookkeeping and most of
`Service.run`'s pacing. What it keeps and should gain:

- A request registry with ids, per-request metrics (queue time, time to first token, tokens per second, prompt
  tokens read from cache), exposed in `/metrics` as today.
- Admission control by budget: refuse or queue when the sum of the admitted requests' contexts exceeds what the
  pools hold, instead of discovering it in the engine.
- Fairness: per-client (API key or connection) round robin in the queue, and the shortest-prompt-first rule that
  BYIELD implements today, applied at admission.
- Tokenisation off the hot path: the tokenizer is pure Python (`tools/strata_tokenizer.py`), and every request
  encodes its whole prompt again; the time it takes on a 100K-token agent prompt has not been measured and should
  be (one `time.perf_counter()` around `encode_prompt`). Two fixes: cache the ids of the previous turn's text prefix
  per conversation, or encode in C++ (the module says a C++ port is planned).
- Threads are fine at this scale: `ThreadingHTTPServer` with one thread per stream serves tens of streams; the
  Python work per token (detokenise, parse, JSON, write) is small against a 20-50 ms window. An asyncio server is
  not what unlocks concurrency here; the engine is.

### 6.7 Several GPUs

Two paths that compose:

- **Pipeline groups** (`--batch-groups`, exists): with the engine scheduler, the groups become micro-batches that
  the scheduler assigns, and a stage can run a prefill slice while another decodes.
- **Data parallel engines.** Two engines, one per GPU, each with its own expert cache and its own scheduler, sharing
  one pinned expert arena through the MAP_SHARED file-backed arena that `PinnedArena` already supports on Linux
  (`include/strata/core/pinned.hpp:49`, `src/core/pinned_shared_test.cpp`), and a server that routes conversations
  by prefix affinity. This needs a CPU with cores to spare (two pools) and is the simplest way to double throughput
  on a machine with two cards of equal size.

### 6.8 Phases and what to measure

Each phase ends with the same harness: `tools/batch_test.py` for exactness, and a new HTTP load generator (section
7) reporting, for C = 1, 2, 4, 8 clients: median and p95 time to first token, tokens per second per client and in
total, VRAM per sequence, and the engine's `strata batch:` window breakdown. The baseline is BATCHING.md's table.

| Phase | Work | Done when |
| --- | --- | --- |
| 0 | CI for the GPU-less tests; the load generator; `STRATA_DECODE_TIMING` lines parsed into the harness | the baseline table is reproduced by the harness on the reference PC |
| 1 | engine-side request table and scheduler over the existing slots; framed protocol with ids; the server's slot logic removed | exactness tests pass; p95 first token at C=4 below BATCHING.md's "last of the round" 1.8 s with the same throughput |
| 2 | global K/V page pool, GDN block pool, device row table, one graph per S, drafts per row, penalties per row, R = 16 | VRAM per sequence measured against the formula in 6.3; 8 clients at 32K on a 12 GB card; tokens per window above 1 per row with drafts |
| 3 | cost-based PCIe share; next-layer prefetch; tensor-core dense path from 8 rows | ms per window against rows flattens on the reference PC; the 4 x 16 GB split's 360 tokens/s at 8 clients improves |
| 4 | data-parallel engines on the shared arena; pipeline micro-batches | two equal cards reach close to twice one card's throughput at C = 8 |

## 7. Contributing: build, tests, structure

### 7.1 What runs today without a GPU (measured in this session, GCC 13.3, 4 cores)

| Step | Command | Result |
| --- | --- | --- |
| configure | `cmake -S . -B build -G Ninja -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON` | 29 s (fetches llama.cpp) |
| build | `ninja -C build` | 45 s |
| C++ tests | `ctest --test-dir build -E '^(expert_parity\|pool_test)$'` | 21 of 23 pass in 21 s; the two excluded need the pack |
| server tests | `python3 -m pytest serve/` | 442 passed, 10 skipped, 175 s |
| tool tests | `STRATA_GGUF_PY=<llama.cpp>/gguf-py python3 -m pytest tools/test_*.py` | 459 passed, 15 s |

With a GPU: `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<cc>` (never omit it), `ctest` with the three
fixture-bound tests excluded, and the four model-level scripts by hand (`tools/batch_test.py`,
`tools/batch_interleave_test.py`, `tools/parking_test.py`, `tools/early_close_test.py`). None of this is written
down in one place today; it belongs in a CONTRIBUTING.md.

### 7.2 A CI that works today

All on GitHub-hosted Linux runners, no GPU:

1. The CPU-only CMake configure, build and `ctest` above, with `-DSTRATA_WERROR=ON`, and a second run of the CPU
   libraries under `-fsanitize=address,undefined`. Cache the llama.cpp checkout.
2. `pytest serve/` and `pytest tools/test_*.py`, after two small fixes: a `pytest.ini` with
   `python_files = test_*.py` (so `early_close_test.py` is not imported) and `tools/_paths.py` raising
   `unittest.SkipTest` instead of calling `sys.exit`.
3. A CUDA compile-only job in `nvidia/cuda:13.0-devel` for `75;86;89;120` (the Dockerfile already builds the
   engine without a GPU, `Dockerfile:41-100`). This is the job that catches an nvcc-only error.
4. A HIP compile-only job in a ROCm image (`tools/hip/build_windows.bat` shows the HIP build needs no GPU).
5. A Windows job for the CPU-only CMake and the Python suites (`serve/test_winjob.py` and the Windows branch of
   `pool_affinity_test` exist for a reason).
6. `clang-format --dry-run` on the host translation units and `ruff` on `serve/`, `tools/` and `setup.py`, once a
   format is chosen.

With a GPU: one self-hosted runner per backend running the full `ctest`, `tools/batch_test.py` for exactness, and
`tools/load_test.py` (added on this branch) for the throughput and first-token table at C = 1, 2, 4, 8. A tiny
model for a greedy golden test is referenced by `.gitignore` (`bench/tiny-*.gguf`, `tools/make_tiny_model.py`)
but the generator is not in the repository; publishing it would let the exactness test run on a 2 GB card.

### 7.3 Structure

- **`src/program/generate.cpp` (11,102 lines).** `main` runs from line 1503 to 11102 and defines 129 lambdas over
  dozens of mutable locals. The lambdas already name the seams: argument parsing (1503-2043), model and pack
  loading (2043-2846), the layer-split planner (2846-3635, a pure cost model that could be unit-tested on
  synthetic profiles), expert cache sizing and fill (3794-4364), graph capture and oracles (4487-5032), and the
  serve loop (5538-10094): prompt lending, stage verifiers, conversation cache (6287-6700), adaptive tier
  (6687-7059), stdin reader and watchdog (7089-7250), batch slots (7369-7617), pipeline groups (7617-7759),
  command dispatch (7759-8015), prompt reading (8200-9033), the decode loop (9033-10047); then the one-shot paths
  (10096-10974), which duplicate the serve loop's decode. Move each into a class that owns the state it captures
  (`Options` with a table-driven parser that also generates `usage()`; `ServeProtocol`; `SlotManager`;
  `ConversationCache`; `AdaptiveTier`; `PromptReader`; `DecodeLoop`), one at a time, with `tools/batch_test.py`
  and the greedy golden runs gating each step. The one-shot path then becomes `DecodeLoop` driven by a file, and
  the three copies of the decode loop and the five copies of the tier policy collapse.
- **`serve/server.py` (5,015 lines).** `make_handler` (3580-4436) is one closure returning a 30-method class with
  if/elif routing; the three streaming loops (4226-4252, 4302-4335, 4409-4435) and the three request prologues
  (`_openai`, `_responses_prepare`, `_anthropic`) are near-copies. Split into `engine.py` (process and protocol),
  `vision.py`, `tokens.py`, `security.py` (host, origin, key, own-page), `http.py` with a route table carrying
  the auth and own-page flags, one module per API, and `service.py`. The 449 HTTP-level tests make this safe.
- **`setup.py` (4,971 lines, 207 functions, no classes).** A `setup/` package (`hw/`, `models.py`, `engine/`,
  `recommend.py`, `config.py`, `start.py`, `cli.py`) behind a thin `setup.py` shim keeps `START-HERE.bat` and
  `setup.sh` unchanged and keeps the stdlib-only rule. The 23 `test_setup_*` files mock by function name, so the
  split is mechanical.
- **Environment switches.** The engine reads 260 distinct `STRATA_*` variables (357 tokens counting macros):
  about 25 trace and dump switches, 8 test hooks, 25 kill-switches for shipped optimizations (`STRATA_NO_*`,
  `*_OLD`, `*_V1`), about 150 opt-in experiments and 25 tuning numbers. `arch_defaults.cpp:18-38` sets 18 of them
  with `setenv` at startup on gfx1151, and the session file must fingerprint every one that changes arithmetic.
  Policy proposal: promote the ones setup or the docs tell users to set to flags or config keys
  (`STRATA_BATCH_MTP`, `STRATA_IQ_MT_MIN`, `STRATA_KV_ROT`, `STRATA_WDDM_BUDGET`, the gfx1151 set); retire a
  kill-switch and delete its old kernel two releases after its parity test went green; read the diagnostics
  through one struct; parse the experiments once into one `Tuning` struct passed down, which also removes the
  server's "set it in the engine's environment" indirection (`server.py:1949-1967`). Keep the rule at
  `arch_defaults.cpp:46` that the user's setting wins.
- **Reference kernels.** The naive kernels are the parity references and should stay, in a
  `src/kernels/reference/` directory excluded from the engine link.
- **`sycl/`.** Either make it a backend of the same tree (the way HIP is: a compat layer and CMake relabeling)
  or declare it a fork with its own release cadence; a 120k-line copy merged by hand will not keep up with a
  main tree that changes 50 commits a month.

### 7.4 Documents to add

- **CONTRIBUTING.md**: the three build recipes and which `ctest` names need fixtures; the Python suites and the
  `STRATA_GGUF_PY` variable; that `*_test.py` scripts need a live engine; every kernel change ships its parity run
  and every speed claim its card, model and engine version; how to add an environment switch (register it in the
  session fingerprint when it changes arithmetic, document it in the switch table); the llama.cpp pin update
  (three places) and the `sycl/` merge procedure; the issue-number convention in comments.
- **ARCHITECTURE.md**: the data path on one page with the file that implements each box (embed, 48 captured layer
  graphs, doorbell, CPU pool, head, sampler; verify window and MTP drafter; prompt chunks); a directory map; the
  three process boundaries and their contracts (server to engine over stdin/stdout, server to `strata-vision`,
  server to MCP servers); the on-disk formats (pack, session file v1, run config, `BUILD.json`, expert profile);
  the backend matrix (CUDA, HIP wave32, gfx906, SYCL) and what each compat layer does; the switch registry; the
  test map.
- **The `--serve` protocol specification.** Today the request grammar lives in `generate.cpp:8017-8066` and
  `server.py:1110`, the slot verbs in BATCHING.md, the session verbs in DETAILS.md, and the response lines
  (`READY`, `T`, `DONE`, `BT`, `BDONE`, `BADM`, `YIELDED`, `INFO`, `ERR`, `FATAL`) nowhere. One page, and a
  fuzz target for the parser once it is extracted.
- A `strata-<model>.json` schema (`serve/runconfig.py:19-40` is the only typed list today).

### 7.5 Security hardening, in order

1. A body-size cap and a read timeout on `/v1/*` (today only the control endpoints are capped).
2. Hash the prebuilt engine zips and the llama.cpp archive in `setup.py` (a `SHA256SUMS` beside each release
   asset), and make the fallback to an unpinned Hugging Face revision an explicit choice, not a warning.
3. Refuse local file paths as image sources unless a config key allows them, or require the key when the server
   is not bound to loopback.
4. Document that `strata-<model>.json` runs commands (`before_load`, `exe`, `args`) and should be writable by the
   user only; check its mode at start on Linux.
5. Keep the `Host` check on with an API key (it costs nothing) or document why it is skipped.

## Appendix A. Findings by file

| # | Severity | Where | What | Suggested fix |
| --- | --- | --- | --- | --- |
| 1 | High | `serve/server.py:1150-1172` | yield-path deadlock: control lock held while waiting for a slot; slot held while waiting for the lock; reproduced 6/6 | never wait for a slot under the lock, or refuse a `BYIELD` that leaves no free slot; add the 3-request test |
| 2 | High | `serve/server.py:950`, `:1266-1272` | `ERR` during a batch admission or solo read drains for a `BADM`/`DONE` that never comes (300 s) | mark the request done on `ERR` as `generate()` does at `:1397-1398` |
| 3 | High | `serve/server.py:909`, `:1214` | no silence watchdog in batch mode | apply `engine_silence_s` in `_control` and the slot loop |
| 4 | High | `src/program/generate.cpp:7411-7489` | admission and return-to-solo copy the K/V through pageable host memory with device syncs per QSA layer | device-to-device async copies of `upto` cells on the verifier stream |
| 5 | High | `src/core/verify.cpp:2095-2216` | one graph pair per batch row layout, unbounded without `--batch-mtp` | canonical row order now; device row table and one graph per S later |
| 6 | High | `src/program/generate.cpp:7537-7552`, `:7743` | any batch-window error ends the process for every slot | fail the row's request, keep the others |
| 7 | High | `src/program/generate.cpp:7529`, `:5446` | the adaptive expert cache never runs while slots decode | call `adapt()` from `batch_step` between windows |
| 8 | High | `src/core/verify.cpp:2119-2120`, `:514`; `generate.cpp:9699`, `:6692`, `:6829`; `src/core/mtp.cpp:290`, `:313` | unchecked CUDA calls on the window path | check and surface as the window's error |
| 9 | High | `src/prefill/prefill.cpp:2003`, `:2050`, `:2931`, `:2937`; `verify.cpp:1929` | unchecked `cudaMemcpyAsync` on the expert ring and DMA: a refused copy computes on stale bytes | check; on failure fall back to the CPU for that expert or fail the chunk |
| 10 | High | `src/program/generate.cpp:6780-6850`, `:9231-9237`; `src/core/expert_source.cpp:2251`, `:2444`, `:2468` | data races under `--pipeline-windows` between the adapt thread and the pool thread | atomics for `host_res`/usage, or a published snapshot per window |
| 11 | High | `src/core/expert_source.cpp:2169-2175` | a dropped exchange clears its override with no log: the expert is silently read from the file | log and count; keep the override until the copy is confirmed |
| 12 | High | `src/core/expert_source.cpp:2160-2164`, `:2198-2201`; `src/kernels/cpu/pool.cpp:551`, `:578`, `:601`; `moe_fused.cu:34-39`, `moe_mmq.cu:16-21`, `iq_kernels.cu:27-30`, `qsa_decode_attn.cu:528-532`, `qsa_select.cu:1276` | `abort()`/`exit(1)` inside a serving process | return errors to the window; the serve loop decides |
| 13 | High | `src/kernels/cpu/native_expert.cpp:79-96` | IQ greedy output depends on draft grouping by default (`STRATA_IQ_MT_MIN=2`) | make exactness the default or state the contract per pack |
| 14 | High | `CMakeLists.txt:152-154` | CUDA architecture forced to sm_120 when unset | default to a list, or require the flag |
| 15 | High | `.github/` | no CI | section 7.2 |
| 16 | High | `setup.py:1177-1180`, `:2313-2400`, `:1983-2055`, `:1389-1420`, `:2560-2565` | unpinned fallback on 404; no checksum on engine zips, llama.cpp archive, CUDA keyring | hashes beside release assets; explicit opt-in for the fallback |
| 17 | Medium | `src/core/verify.cpp:2464` | batch commit is synchronous | record an event as the solo commit does |
| 18 | Medium | `src/core/verify.cpp:965-997`, `:1042-1054` | per-row attention and append launches in batch windows | pointer-array variants over per-row pools |
| 19 | Medium | `src/program/generate.cpp:9965` after `:7755`/`:8808`; `:7566-7573` | `BSTOP` lost during admission; one extra token after a stop | check the stop flag after admission; stop before emitting |
| 20 | Medium | `src/program/generate.cpp:9727`, `:10796`, `:6705`, `:1202-1206` | a new thread per adaptive round; blocking event wait at lag 1 | `JobThread`; default lag 2 or the async tier |
| 21 | Medium | `src/program/generate.cpp:9699` | pageable penalty-history copy on the legacy stream | mapped pinned rows copied inside the graph |
| 22 | Medium | `src/program/generate.cpp:9984-9989`, `:8835-8838` | per-slot checkpoint copies, up to N x 118 MB of host RAM, unbudgeted | count them in the conversation cache budget |
| 23 | Medium | `src/program/generate.cpp:6726-6852`, `:10553-10645`, `:6900-6949`; `peer_experts.cpp:615-658`; `remote_expert_opt.cu:807-864` | the tier policy in five copies; the decode loop in three | one `AdaptiveTier`, one `DecodeLoop` |
| 24 | Medium | `src/core/expert_source.cpp:1010-1019` | up to 8 threads created per layer per window in the file tier | a persistent fetch pool |
| 25 | Medium | `src/core/expert_source.cpp:2442-2444`; `generate.cpp:6746-6749` | swap gain ignores blob size and counts a batch row 8 times | distinct (window, expert) weighted by bytes |
| 26 | Medium | `src/core/expert_source.cpp:2494-2504`, `:2255-2271` | PCIe share takes the last m misses in routing order; exchange buffers never eligible | cost-based pick; device alias for the exchange buffers |
| 27 | Medium | `src/core/expert_cache.cpp` vs `generate.cpp:4929` | `ExpertCache::residency_` stale after the first swap | one table, or drop the stale one |
| 28 | Medium | `serve/server.py:3987-4008`; `frontend.py:335`, `:434`, `:444`; `server.py:4206` | malformed requests drop the connection without a response | a catch-all mapping to 400/500 JSON |
| 29 | Medium | `serve/server.py:990-1008` | slot marked free after 600 s without `BDONE` | re-sync with the engine, or treat as fatal |
| 30 | Medium | `serve/server.py:3598`, `:4035-4056` | no body cap or read timeout on `/v1/*` | cap and timeout |
| 31 | Medium | `serve/server.py:2762`, `:2913`, `:2766-2770` | vision encoder and batch requests do not share the lock the comment assumes | take the lock in batch mode too, or document |
| 32 | Medium | `src/kernels/cpu/pool.cpp:693`, `:833` | redundant re-park barrier | remove; keep the count |
| 33 | Medium | `src/core/expert_source.cpp:2461-2471`, `:2582-2590`; `pool.cpp:738-740` | host-serial plan, quantization and intermediate quantization | items 17-18 of section 5 |
| 34 | Medium | `src/prefill/prefill.cpp:2589-2595` | 48 stream syncs per chunk on the MMQ path | group on the GPU |
| 35 | Medium | `src/prefill/gemm.cu:773-809` | dense weights dequantized every chunk | int8 tensor-core path, opt-in |
| 36 | Medium | `CMakeLists.txt:309-312`, `:634-917`, `:878-882`, `:1399-1410` | unused kernels in the engine library; parity executables unguarded; three tests that can never pass | reference directory; guards; mark fixture tests |
| 37 | Medium | `tools/_paths.py:21-22`; `tools/early_close_test.py` | `pytest tools/` aborts collection | `SkipTest`; `pytest.ini` file glob |
| 38 | Medium | `sycl/` | a 120k-line diverged copy | backend or fork, decided |
| 39 | Low | `src/program/generate.cpp:4933-4936`, `:8674` | synchronous residency table upload per lend, refill and apply | async copy with a stream wait |
| 40 | Low | `src/program/generate.cpp:7758`; `server.py:1122-1129` | `BYIELD` dropped outside `read_part` while the server reserved a slot | acknowledge or refuse the yield explicitly |
| 41 | Low | `serve/server.py:3172`, `:1283-1292`, `:524`, `:2930`, `:3914` | unlocked counter; stale slot lists after restart; hard-coded EOS ids; status mixing; `RecursionError` | small fixes |
| 42 | Low | `src/kernels/cpu/expert.cpp:153`, `:352-373` | namespace-scope `getenv` before the ISA check; CPUID-only feature check | move after the check; use `cpu_avx512_ok` |
| 43 | Low | `CMakeLists.txt:1-9`, `generate.cpp:14-15`, `README.md:58`, `sycl/CMakeLists.txt:13`, `.gitignore:2-4`, `setup.py:97` | stale statements and references to files not in the repository | fix in passing |
