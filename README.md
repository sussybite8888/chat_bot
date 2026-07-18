# sodachat — a small neural network you train yourself

A tiny GPT (~1–14M parameters), trained from scratch on your own machine, put
to two uses that lean on its speed and small size:

- **A chatbot** with a terminal UI, a Discord bot, and a Google Chat app
  sharing one engine.
- **A game controller** — the same architecture, small enough to pick an
  action every frame, trained to play Snake, Pong, Dodge, and Tic-Tac-Toe. One
  agent both chats and plays (see [Playing games](#playing-games)).

Two model backends:

| Backend | What it is | Notes |
|---|---|---|
| `mini` (default) | A ~14M-param GPT (RoPE / RMSNorm / SwiGLU) with an 8k BPE subword vocabulary, both trained **from scratch** on [SODA](https://huggingface.co/datasets/allenai/soda) (~1.2M narrative-grounded dialogues, ~210M tokens) | Wants a GPU: ~6h. No pretrained weights anywhere. |
| `gpt2` | GPT-2 (124M) fine-tuned on dialogue data | Opt-in: never started automatically |

Select with `--backend` (CLI) or `SODACHAT_BACKEND` (Discord / Google Chat).
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

At inference the conversation is rendered the same way, ending with `B:` so
the model continues as the bot, and generation stops at the next newline or
separator.

Note the missing space after `B:` — that is deliberate. Byte-level BPE folds
a leading space into the following word (`" Electronic"` is a single token),
so a trailing space would tokenize as a lone space token, a sequence that
never follows `B:` in training. The model then emits word-*continuation*
fragments: `"Electronic"` comes out as `"ronic"`. This bug is invisible in
the loss and only shows up in generated text.

The engine ([engine.py](sodachat/engine.py)) wraps that with:

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
  Disable with `--unfiltered` (CLI) or `SODACHAT_UNFILTERED=1`.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in tokens as needed
```

Datasets download automatically on first use: SODA and DailyDialog from the
Hugging Face hub, NPS Chat via NLTK (`pip install nltk`, only needed for
`--dataset nps`).

## Training

```sh
.venv/bin/python -m sodachat.train                    # mini-GPT on SODA (~6h on a GPU)
.venv/bin/python -m sodachat.train --dataset dailydialog   # small/fast, lower quality
.venv/bin/python -m sodachat.finetune                 # GPT-2 (needs >=16GB / GPU)
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
ssh user@host 'cd ~/chat_bot && nohup python3 -u -m sodachat.train > train.log 2>&1 &'
rsync -az user@host:~/chat_bot/models/minigpt-soda.pt ./models/
```

On a Jetson Orin NX (CUDA 12.6) this runs at ~10k tok/s, ~3GB VRAM at
`--batch-size 32`. Throughput is latency-bound, so larger batches buy almost
nothing — on a tight machine use `--batch-size 16`.

## Terminal chat

```sh
.venv/bin/python -m sodachat                         # interactive
.venv/bin/python -m sodachat --once "hey whats up"   # one-shot
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
   .venv/bin/python -m sodachat.discord_bot
   ```

The bot replies to DMs and @mentions, keeping short per-channel history. Set
`DISCORD_RESPOND_ALL=1` to reply to every message it can read.

## Google Chat

The Google Chat frontend is an HTTP app: Google POSTs events to your server
and renders the JSON it returns.

1. Run the server (default port 8080, override with `GOOGLE_CHAT_PORT`):

   ```sh
   .venv/bin/python -m sodachat.google_chat
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

## Playing games

The same architecture, small enough to decide in ~1.6ms, also works as a game
controller. `sodachat/games/` is a general framework: a game exposes an
observation and the model predicts an action token, trained by **behaviour
cloning** (a scripted expert plays thousands of games; the model learns to
predict its move). No reinforcement learning, no pretrained weights — ~1M
parameters, trained in minutes.

Crucially, observations don't have to be bitmaps. Two modalities are built in:

| Modality | Observation | Games | Encoded by |
|---|---|---|---|
| **grid** | a 2D board of symbols | `snake`, `pong`, `dodge` | one token per cell |
| **text** | a plain-text state | `tictactoe` | character by character |

| Game | Genre | The model must… |
|---|---|---|
| `snake` | pathfinding (grid) | reach food without trapping itself |
| `pong` | tracking (grid) | move a paddle to intercept a bouncing ball |
| `dodge` | avoidance (grid) | line up with the gap in descending walls |
| `tictactoe` | symbolic (text) | play optimally from a text board — never lose |

Because a text state is the same kind of token stream the chat model uses, one
agent both converses *and* plays (see below).

```sh
.venv/bin/python -m sodachat.game_train --game snake      # train (snake/pong/dodge/tictactoe)
.venv/bin/python -m sodachat.play       --game snake      # watch a grid game, live
```

`play` shows a HUD with the score and the model's decide time — typically
~1.6ms/move, hundreds of moves per second, far faster than the frame rate.
Add `--fps 0` to let it run flat out.

### One agent: chat and play together

`sodachat.agent` is a single interface that routes each message — an intent to
play starts a game, everything else is chat:

```sh
.venv/bin/python -m sodachat.agent
```

```
you › hi there
bot › Not much, just relaxing. How about you?
you › /play snake
bot › Playing snake — I'll keep going on my own. /score, /state, /board, /watch…
you › /score                       (a command → read from the game, not the model)
bot › Score: 3.
  snake: score 3   best 5   game #2
you › /watch                       (stream the live board for a few seconds)
      ┌──── playing live ────┐
      │ · · @ o o · · * · ·  │   score 4   best 5   game #2
      └──────────────────────┘
you › nice moves!                  (plain text → the chat model)
bot › Thanks! I'm trying.
you › /stop
bot › Stopped — best score was 5.
```

The interface has one rule: **plain text goes to the model**, and **game
control is `/commands`** — `/play`, `/stop`, `/watch`, the exact, deterministic
readouts `/score`, `/state`, `/board` (straight from the game object), and
`/model`. `/help` lists them.

`/model` shows the loaded models (params, architecture, training) and **switches
which model powers the agent**: `/model specialist` (default) uses a separate
model per task — the chat model, the reader, and a per-game player — while
`/model unified` routes chat, reading, *and* game moves through a single 30M
model trained on the whole mixture (all chat datasets + the reader task + the
games). You can flip between them live and feel the trade-off: the specialists
read the score exactly and move in ~1ms; the one model is more general but
currently weaker at exact numeric reads and ~10× slower per move.

Grid games (Snake, Pong, Dodge) **run continuously on their own** in a
background thread once started — stepping and auto-restarting while you type — so
`/score` always reflects the current moment. `/watch` streams the animated board
for a few seconds (typing and a repainting board can't share a terminal without
a full TUI, so the animation is on demand; `sodachat.play` is the standalone
full-speed view). Tic-Tac-Toe is turn-based — type a cell number to move.

### The model can read the game

Ask about the game in plain English and a small **reader model** answers by
reading the live state — "what's the score?" → *"you have 7 points."* The chat
model can't do this (it was never trained to reference a score), so
[reader.py](sodachat/reader.py) is a ~0.9M-param model trained to *read*: given
the state written as fields (`score 7 length 4 food up`) and a question, it
locates the field the question names and copies its value.

It genuinely reads rather than memorizes — trained with the fields in random
order and values spanning 0–99, and **verified on scores held out of training**
(17, 33, 54, 76, 91), which it reports correctly despite never seeing them. It's
also trained to stay silent on chit-chat and on fields a game doesn't have (ask
Snake "whose turn?"), so those fall through to the chat model. During a game,
plain-text questions run through the reader against the live state; `/score`
remains for an exact, instant readout.

### Controlling and talking at the same time (one model, two heads)

The agent above *routes* between a chat model and game policies. A tighter
form of the same idea is a **single model with two output heads** that fire on
the same forward pass — control and text simultaneously, not either/or:

```sh
.venv/bin/python -m sodachat.narrate train    # ~a few minutes on CPU
.venv/bin/python -m sodachat.narrate play     # watch it play Snake AND narrate
```

`MultiHeadGPT` ([model.py](sodachat/model.py)) is one shared transformer trunk
with two heads:

- an **action head** — a linear layer over the moves, read from the trunk's
  hidden state, that decides where to go;
- the **LM head** (tied to the embeddings) that generates a running commentary.

Each tick, the board is rendered as text and passed through the trunk once; the
action head picks the move while the LM head narrates it:

```
· · @ · · · · · · ·
· · o · · · * · · ·
· · o o · · · · · ·        💬 "food is up and right, turning up"
· · · · · · · · · ·        score 3   move: up
```

The move is ready from that first forward pass (~1.6 ms); the words are then
generated token by token from the LM head (~20–70 ms), so the action never
waits on the narration. The action head plays a real game (Snake avg ~22,
versus ~26 for the single-purpose model — a small cost for the shared trunk
also learning to talk), and the commentary stays consistent with the move
because both read the same encoding of the board.

Training is multi-task: for each frame the action head is supervised by the
scripted expert and the LM head by templated commentary derived from the game
state, with the two losses summed over one sequence — so the shared trunk
learns a representation that serves both. The commentary is trained-from-scratch
narration, so it's simple and game-flavoured, not open-ended chat; the point is
that both outputs come from one model at once.

### Consistent latency (why real-time control works)

For real-time control, *worst-case* latency matters more than the average — a
single slow frame stutters or misses a deadline. `--bench` reports the full
per-move distribution:

```sh
.venv/bin/python -m sodachat.play --game snake --bench
```

Measured on CPU (default), 20000 moves, GC paused as in play:

| p50 | p99 | p99.9 | over 30 fps budget |
|---|---|---|---|
| 2.4 ms | 3.7 ms | 11.5 ms | 0.025% of frames |

**The model's compute is consistent** — 99% of moves land within ~1.5 ms of the
median, and the standard deviation is ~0.1 ms. That steadiness is engineered:
the model is warmed up before the loop (so kernel compilation isn't an
in-game outlier), the input tensor is reused, the cyclic garbage collector is
paused during play (its pauses were a systematic multi-ms spike source), and
it runs single-threaded on CPU. The GPU is counter-intuitively worse for a
model this small — async kernel-launch variance gives it a much heavier tail
(occasional tens-of-ms spikes) — so play defaults to CPU. Because the board is
fixed-size, the sequence length, and thus the work per move, is constant.

**The rare tail is the OS, not the model.** On a general-purpose OS, ~0.02% of
frames are preempted by other processes and overrun their budget; the absolute
max swings from ~5 ms to ~75 ms between runs purely from scheduling noise. Two
things make this a non-issue: the fixed-timestep loop *absorbs* a slow frame
(it resyncs to the next deadline instead of spiralling, so one late frame in
thousands is imperceptible), and at typical rates (10–30 fps) the frame budget
is many times the p99 anyway. Hard-real-time guarantees would need process
priority pinning or an RTOS, which is out of scope for a terminal game.

**Adding your own game** is one file: subclass `Game`, set `MODALITY`
(`"grid"` or `"text"`), the action list, and implement `reset` / `observe` /
`step` plus a scripted `expert` for the training data. Decorate it with
`@register` and it's immediately trainable, playable, and available in the
agent — nothing in the tokenizer, trainer, or UI is game-specific. See
[games/snake.py](sodachat/games/snake.py) (grid) and
[games/tictactoe.py](sodachat/games/tictactoe.py) (text) for the two patterns.

## Layout

```
sodachat/
  corpus.py       # load + clean the NPS Chat corpus
  data.py         # dialogue dataset loaders (SODA, DailyDialog, NPS)
  model.py        # mini GPT (nanoGPT-style transformer) + BPE/char tokenizers
  train.py        # mini-GPT training -> models/minigpt-soda.pt
  hf_model.py     # fine-tuned GPT-2 backend (opt-in)
  finetune.py     # GPT-2 fine-tuning -> models/gpt2-dailydialog/
  engine.py       # chat generation + MMI relevance reranking
  cli.py          # terminal chat UI (rich)
  discord_bot.py  # Discord chat frontend (discord.py)
  google_chat.py  # Google Chat frontend (FastAPI webhook)
  agent.py        # unified interface: chat (plain text) + /commands for games
  reader.py       # small model that reads game state to answer questions
  game_train.py   # behaviour-cloning trainer for game control
  play.py         # real-time terminal UI for grid games (rich.Live)
  games/          # pluggable games: core framework + snake/pong/dodge/tictactoe
```
