# npschat — a small neural chatbot, three frontends

A chatbot you train yourself, with a terminal UI, a Discord bot, and a Google
Chat app sharing one engine.

Two model backends:

| Backend | What it is | Notes |
|---|---|---|
| `mini` (default) | A ~14M-param GPT (RoPE / RMSNorm / SwiGLU) with an 8k BPE subword vocabulary, both trained **from scratch** on [SODA](https://huggingface.co/datasets/allenai/soda) (~1.2M narrative-grounded dialogues, ~210M tokens) | Wants a GPU: ~6h. No pretrained weights anywhere. |
| `gpt2` | GPT-2 (124M) fine-tuned on dialogue data | Opt-in: never started automatically |

Select with `--backend` (CLI) or `NPSCHAT_BACKEND` (Discord / Google Chat).
Other datasets: `--dataset dailydialog`, or `--dataset nps` (char-level) for
vintage 2006 chat-room flavor.

**Why SODA and not something smaller.** A 14M-param model needs roughly 20
tokens per parameter ([Chinchilla](https://arxiv.org/abs/2203.15556)) — about
280M tokens. DailyDialog supplies 1.5M, i.e. **0.1 tokens/param, ~200× too
few**. That deficit is what "grammatical but irrelevant" actually looks like:
[TinyStories](https://arxiv.org/abs/2305.07759) found grammar saturates early
and cheaply, while *using the context* is the last ability to emerge and is
the most data-hungry. SODA's ~210M tokens put this model near the optimal
ratio, and its dialogues are grounded in a narrative, so turns actually
respond to each other.

**Expectations:** short, mostly grammatical small talk that generally tracks
the topic. It is not an instruction-following assistant — it cannot do
arithmetic or answer factual questions, because none of that is in the
training data. That ceiling is the model size and the corpus, not the setup.

## How it works

Both backends are causal language models over a chat stream. Dialogues are
rendered as tagged, alternating turns and terminated with a separator token:

```
A: Hey Shavon, what's up? You seem troubled.
B: Yeah, I am. I'm just having a hard time and needed someone to talk to.
A: Of course, man. I'm always here for you. What's going on?
<|endofdialog|>
```

The speaker tags teach the model that turns alternate and which side it is
answering as. The separator marks where a conversation *ends* — without it,
concatenated dialogues run together and the model learns that abruptly
switching topic is a valid reply (this was a real bug here: 12.8% of training
transitions were dialogue boundaries).

At inference the conversation is rendered the same way, ending with `B: ` so
the model continues as the bot, and generation stops at the next newline or
separator.

The engine ([engine.py](npschat_bot/engine.py)) wraps that with:

- **Conversation history** — the last few turns condition each generation
  (kept per terminal session / Discord channel / Google Chat space).
- **Relevance reranking (MMI)** — several candidate replies are sampled and
  each is scored by how much the conversation context raises its likelihood
  versus no context (`log P(reply | context) − λ·log P(reply)`, computed
  with the same model). Fluent-but-generic candidates that ignore your
  message score low; the best-scoring one is returned (shown as `rel` in
  the terminal UI).
- **Reply trimming** — generations are cut to their first sentence or two;
  sampled tails tend to wander.
- **Output filtering** — a profanity filter is applied to replies by default.
  Disable with `--unfiltered` (CLI) or `NPSCHAT_UNFILTERED=1`.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in tokens as needed
```

Datasets download automatically (NPS Chat via NLTK, DailyDialog via the
Hugging Face hub).

## Training

```sh
.venv/bin/python -m npschat_bot.train                    # mini-GPT on SODA (~6h on a GPU)
.venv/bin/python -m npschat_bot.train --dataset dailydialog   # small/fast, lower quality
.venv/bin/python -m npschat_bot.finetune                 # GPT-2 (needs >=16GB / GPU)
```

The dataset is tokenized once into a flat `uint16` file under `models/`
(`soda-train.bin`, ~420MB) and memory-mapped during training, so RAM use stays
flat regardless of corpus size. That step takes ~7 min and is cached.
Checkpoints (`models/minigpt-soda.pt`) keep the best validation loss, with the
BPE vocabulary stored inside.

Flags: `--dataset soda|dailydialog|nps`, `--steps`, `--batch-size`, `--lr`,
`--device`, `--out`, `--seed`.

### Training memory (read this on a laptop)

PyTorch's Apple-GPU backend (MPS) allocates **wired** memory — the OS cannot
swap or compress it. On an 8GB Mac a training run pins several GB and starves
everything else; a batch-64 run here took the machine to 6.2GB wired and
froze it. The trainers cap the allocator
(`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`) so an oversized run raises a clean
OOM instead of hanging the system, but the squeeze is real.

Prefer training on a separate machine with a real GPU. Any box with CUDA
works — copy the repo over, run the same command, copy
`models/minigpt-soda.pt` back:

```sh
rsync -az --exclude .venv --exclude models ./ user@host:~/chat_bot/
ssh user@host 'cd ~/chat_bot && nohup python3 -u -m npschat_bot.train > train.log 2>&1 &'
rsync -az user@host:~/chat_bot/models/minigpt-soda.pt ./models/
```

On a Jetson Orin NX (CUDA 12.6) this runs at ~10k tok/s, ~3GB VRAM at
`--batch-size 32`. Throughput is latency-bound, so larger batches buy almost
nothing — on a tight machine use `--batch-size 16`.

## Terminal chat

```sh
.venv/bin/python -m npschat_bot                         # interactive
.venv/bin/python -m npschat_bot --once "hey whats up"   # one-shot
```

Flags: `--backend mini|gpt2`, `--plain` (hide reply metadata),
`--unfiltered`, `--seed N`.

## Discord

1. Create an application at <https://discord.com/developers/applications>,
   add a **Bot**, and copy its token into `.env` as `DISCORD_BOT_TOKEN`.
2. On the Bot page, enable the **Message Content Intent** (Privileged
   Gateway Intents).
3. Invite the bot: OAuth2 → URL Generator → scope `bot` → permissions
   *Send Messages*, *Read Message History* → open the generated URL.
4. Run it:

   ```sh
   .venv/bin/python -m npschat_bot.discord_bot
   ```

The bot replies to DMs and @mentions, keeping short per-channel history. Set
`DISCORD_RESPOND_ALL=1` to reply to every message it can read.

## Google Chat

The Google Chat frontend is an HTTP app: Google POSTs events to your server
and renders the JSON it returns.

1. Run the server (default port 8080, override with `GOOGLE_CHAT_PORT`):

   ```sh
   .venv/bin/python -m npschat_bot.google_chat
   ```

2. Expose it over public HTTPS — for local development:

   ```sh
   ngrok http 8080     # or: cloudflared tunnel --url http://localhost:8080
   ```

3. In the [Google Cloud console](https://console.cloud.google.com), enable
   the **Google Chat API**, then under its **Configuration** tab set up the
   app: name/avatar/description, *Receive 1:1 messages*, *Join spaces*, and
   connection type **HTTP endpoint URL** pointing at your public URL.

4. In production, set `GOOGLE_CHAT_AUDIENCE` to your Cloud **project
   number** and `pip install google-auth` — each request's bearer token is
   then verified as coming from Google Chat. Without it the endpoint accepts
   unauthenticated requests (fine for local testing only).

Smoke-test without Google:

```sh
curl -s -X POST localhost:8080/ -H 'Content-Type: application/json' \
  -d '{"type":"MESSAGE","message":{"text":"hello there"}}'
```

## Layout

```
npschat_bot/
  corpus.py       # load + clean the NPS Chat corpus
  data.py         # dialogue dataset loaders (DailyDialog, NPS)
  model.py        # mini GPT (nanoGPT-style transformer) + BPE/char tokenizers
  train.py        # mini-GPT training -> models/minigpt-dailydialog.pt
  hf_model.py     # fine-tuned GPT-2 backend (opt-in)
  finetune.py     # GPT-2 fine-tuning -> models/gpt2-dailydialog/
  engine.py       # generation + dialogue-act model (shared brain)
  cli.py          # terminal UI (rich)
  discord_bot.py  # Discord frontend (discord.py)
  google_chat.py  # Google Chat frontend (FastAPI webhook)
```
