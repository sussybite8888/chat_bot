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
- **Nucleus sampling + repetition penalty** — reply candidates are sampled with
  top-p (`0.95`) instead of a bare top-k, a softer tail cut, plus a mild CTRL-
  style repetition penalty (`1.15`) over the tokens generated so far, which keeps
  the model off the self-looping continuations small LMs fall into. The same
  knobs the GPT-2 backend already uses; applied only to chat, not the low-
  temperature reader/game paths. See `warp_logits` in [blocks.py](sodachat/blocks.py).
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
.venv/bin/python -m sodachat                          # agent: chat + games (default)
.venv/bin/python -m sodachat.cli                       # plain chat, no games
.venv/bin/python -m sodachat.cli --once "hey whats up" # one-shot
```

`sodachat.cli` flags: `--backend mini|gpt2`, `--plain` (hide reply metadata),
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

The same architecture, small enough to decide a move in real time, also works as
a game controller. `sodachat/games/` is a general framework: a game exposes an
observation and the model predicts an action token, trained by **behaviour
cloning** (a scripted expert plays thousands of games; the model learns to
predict its move). No reinforcement learning, no pretrained weights — ~1.8M
parameters reading a 20×20 board, trained in minutes on a GPU.

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
.venv/bin/python -m sodachat.gui                          # same, in a desktop window
.venv/bin/python -m sodachat.games.versus                 # play snake against the bot (multiplayer)
```

`play` shows a HUD with the score and the model's decide time — on the 20×20
boards, roughly ~16ms/move on a laptop CPU (tens of moves per second), still far
faster than the frame rate. Add `--fps 0` to let it run flat out.

`gui` opens a pygame window with tabs for all four games: grid games play
themselves continuously (space pauses, `+`/`-` changes speed), and tic-tac-toe
is interactive — click a cell to play X against the model.

**Multiplayer snake — you vs. the bot.** `games.versus` (or `/duel` in the
agent) puts two snakes on one board, both racing for the same food: you steer
one with WASD / the arrow keys, the bot steers the other in real time. The bot
needs *no new model* — each tick the two-snake board is folded into the ordinary
*single-snake* view the solo model already reads, with your snake drawn as plain
body cells, i.e. one more wall to avoid. Since a second snake is out of the solo
model's training distribution, the same masking idea that keeps it from playing
an *illegal* move is extended to keep it from playing a *suicidal* one: it
follows the model's move unless that move would crash this tick, then steps clear
via the scripted greedy. Run into a wall, yourself, or either body and you're
out; head-on, the longer snake lives; last snake standing wins.

### One agent: chat and play together

`sodachat.agent` is a single interface that routes each message — an intent to
play starts a game, everything else is chat. It's what `python -m sodachat`
runs by default:

```sh
.venv/bin/python -m sodachat
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
control is `/commands`** — `/play`, `/stop`, `/watch`, `/duel` (play snake
against the bot, live), the exact, deterministic readouts `/score`, `/state`,
`/board`, `/model`, and `/stats` (generation speed: tok/s, ms/reply, frequency).
Images are part of the conversation: **drop an image file into the chat**
(or `/see <image>`) and the vision specialist says what it shows — a
handwritten digit or a photo subject like a dog or a cat (works in any mode).
The agent remembers what it saw, so a plain-text follow-up like *"what was in
the picture?"* gets answered from it — the same pattern as the reader
answering questions about the live game state — and the exchange lands in the
chat history, so the chat model can keep talking about it. `/help` lists all
commands. Each reply also shows its speed inline.

`/model` shows the loaded models (params, architecture, training) and **switches
which model powers the agent**, live:

- `/model expert` (default once trained) — **one model whose game and chat
  weights are largely separate** (task-routed experts; see below). Chat and
  reading flow through a *text* expert + the LM head; game moves flow through a
  *game* expert + a dedicated action head that picks a move in a **single forward
  pass**. Because the two tasks no longer share their feed-forward weights,
  teaching it to play well stopped eroding its chat — the fix for the problems
  the unified and instruct models had below. Moves are goal-conditioned: each
  game uses its natural goal, or set one with `/goal` to steer it mid-game.
- `/model specialist` — a separate model per task (chat model, reader,
  per-game player). Best raw quality: exact reads, ~1ms moves.
- `/model unified` — one 30M *dense* model trained on the whole mixture (all chat
  datasets + reader + games). One set of weights juggles everything, so it chats
  and reads well but plays weakly (games were a small slice of its training).
- `/model instruct` — an earlier VLA-style post-train of the unified model
  ([instruct.py](sodachat/instruct.py)): pad-loads its weights and adds
  instruction-conditioning. Kept for comparison — it demonstrates goal-following
  but its single shared FFN meant post-training for games regressed reading and
  play. The expert model is that idea done right.

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

The move is ready from that first forward pass; the words are then
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

### One model, separated game and chat weights (task-routed experts)

The unified model put *everything* — chat, reading, four games — through one set
of weights, and games lost: they were a small slice of the data, so it chatted
well but played weakly. The obvious fix (post-train it harder on games, as
[instruct.py](sodachat/instruct.py) did) made it worse, because a single shared
feed-forward network can't learn to play without overwriting what made it chat.
That's **catastrophic interference**, and it's the real reason for splitting the
model.

`ExpertGPT` ([expert.py](sodachat/expert.py)) keeps *one* model but stops the
game and chat parts from sharing so much. Attention, embeddings, and norms stay
shared (they're task-general), but **each block's feed-forward network is split
into two experts — a text expert and a game expert — and every token is routed
to its task's expert**:

```
token ─▶ shared attention ─▶ ┌─ text token  ─▶ TEXT expert ─┐ ─▶ shared norm ─▶ ┌ LM head (chat/read)
                             └─ game token  ─▶ GAME expert ─┘                   └ action head (moves)
```

Chat tokens and game tokens flow through *different* FFN weights, so the
gradients from learning to play never touch the chat expert — and vice versa.
Same idea as a Mixture-of-Experts, but routed by **task** (a per-token tag)
rather than a learned gate, so it's deterministic and adds no routing cost. It's
still one network with one warm-start: [expert.py](sodachat/expert.py) copies the
unified model's shared weights and *duplicates* its single FFN into both experts,
so each starts competent and then specializes.

Two objectives train it at once: an **LM loss** on every token (routed to its
expert) keeps chat and reading sharp, and an **action loss** on the game expert's
`<|act|>` positions teaches board→move. Moves come off the action head in a
single forward pass (no token-by-token generation), and they're
**goal-conditioned** — the same board yields a different move for "eat the food"
versus "go to the top left corner", the VLA property, folded into the one model
instead of a separate post-train. It's the default in the agent (`/model expert`)
once trained:

```sh
python -m sodachat.expert train --device cuda   # warm-starts from unified.pt
python -m sodachat.expert eval --game snake      # mean score over episodes
python -m sodachat.expert vla                     # zero-shot instruction-following probe
```

**Testing the instruction-following without training anything** — the `sandbox`
game ([games/sandbox.py](sodachat/games/sandbox.py)) is a bare movement grid (one
agent, one target) that exists purely to probe VLA transfer. The model *never
trains on it*, but its moves (`up/down/left/right`) and goals ("go to the top
left corner", "eat the food") are the ones it learned on Snake, and the board
uses the same `model_board()` glyphs (agent `@`, target `*` — Snake's head and
food) — so a goal-conditioned model obeys instructions on it zero-shot. `expert
vla` measures the follow-rate; you can also `/play sandbox` in the agent and
steer it live with `/goal`. Measured on the final model: directional goals
100%, corner goals 100%, target goals ~67% — instruction-following that
generalizes to a game outside the training set.

### Specialists the expert can load (new capabilities as plug-ins)

The task-routed split enables one more trick: **new capabilities as plug-in
experts**. A *specialist* is a fresh FFN expert per block plus its own output
head and special tokens, trained with the entire shared trunk **frozen** — so
by construction its training cannot disturb chat, reading, or play (those
weights never receive a gradient). It ships as a small standalone checkpoint,
and `ExpertLM` auto-loads every `models/specialist-*.pt` at startup, grafting
each one on as a new routed expert (`attach_specialist` in
[expert.py](sodachat/expert.py)).

The first specialist is **image recognition**
([vision.py](sodachat/vision.py)): handwritten digits (MNIST) *and* everyday
photo subjects (CIFAR-10 — airplane, automobile, bird, cat, deer, dog, frog,
horse, ship, truck), 20 labels total. Images are rendered the way game boards
are — as a glyph grid the tokenizer already reads. Any image is pooled to at
most 16×16, and each cell becomes one character: a gray ramp `.:+#` where the
cell is colorless, or a hue letter `r y g c b m` (uppercase = bright) where it
has real color — so the model keeps the color signal that separates sky from
fur. Dense rows keep an image at ~75–280 BPE tokens:

```
<|img|>
BBBBBBBBBBBBBBBB      ← bright blue sky
::+rrr+::+##+:::
:+rrrrr+:+##+:::      ← red fuselage
gggggggggggggggg      ← grass
<|cls|>
```

The label reads off a 20-way class head at the trailing `<|cls|>` in a
**single forward pass**, exactly how game moves read off the action head at
`<|act|>`. The whole document routes to the vision expert, which is seeded
from the *game* expert (glyph boards are the closest diet to a rendered
image) and then specializes. Digits train with random polarity, so a photo of
a pen-and-paper digit works without preprocessing; CIFAR trains with mirror
augmentation. Expect digits to be near-perfect and object labels to be a good
guess rather than an oracle — CIFAR-10 through a 16×16 glyph grid and a
frozen text trunk is genuinely hard.

```sh
python -m sodachat.vision train    # needs models/expert.pt; trunk stays frozen
python -m sodachat.vision eval     # held-out MNIST test accuracy
python -m sodachat.vision demo     # print a few rendered digits + predictions
```

Once trained, the expert picks it up automatically — `/model` in the agent
lists it under the expert, and dropping an image into the chat (or
`/see <image-file>`) recognizes it, with plain-text follow-ups ("what was in
the picture?") answered from what it saw — and `ExpertLM.classify("vision",
doc)` (or `vision.classify(lm, pixels)`) does the same in code. Because a specialist's
new tokens claim vocabulary ids at training time, specialists trained from
the same base attach in the order they were trained; a specialist also
records a fingerprint of the trunk it was trained against, so a stale one
fails to load with a clear message instead of silently misfiring.

### A second specialist: code (language ID + completion)

The same plug-in mechanism adds a **code** specialist
([code.py](sodachat/code.py)): a fresh expert + head, trained with the trunk
frozen, that (1) names a snippet's programming language and (2) continues it.
The corpus is [CodeSearchNet](https://huggingface.co/datasets/code_search_net)
— function bodies across six languages (python, java, javascript, php, ruby,
go), subsampled to a few thousand each. A snippet becomes a tagged doc the
same way an image does:

```
<|code|>
def hello(name):
    return f"hi {name}"
<|cls|>
```

The language reads off a 6-way class head at the trailing `<|cls|>` in a
single forward pass — same mechanism as the vision label and the game move.
The completion path is different: it reuses the *shared, frozen* LM head
(tied to the embeddings) and generates token by token with every token routed
to the code expert. So classification is what this specialist actually learns
(and it learns it well — expect high accuracy, like vision's digits), while
completion is bounded by what the pretrained text trunk already knows. Treat
continuations as autocomplete-flavoured, not a real code model — a ~14M trunk
frozen at its dialogue diet can only do so much with source code.

```sh
python -m sodachat.code train    # needs models/expert.pt; trunk stays frozen
python -m sodachat.code eval     # held-out accuracy, per language
python -m sodachat.code demo     # print a few snippets + predicted language
python -m sodachat.code complete --file snippet.py
```

Once trained, the expert loads it automatically — it shows up under the
expert in `/model`, and dropping a source file into the chat (or
`/code <file>`) names its language. Add `complete` (`/code <file> complete`)
and it continues the file; a plain-text follow-up like *"what language was
that?"* is answered from what it read — the same pattern as `/see`.

### A reasoning specialist: thinking step by step

Every specialist so far answers in one shot. The **reasoning** specialist
([reason.py](sodachat/reason.py)) is the first one trained to *show its work* —
given a question it writes the intermediate steps, then commits to an answer:

```
<|reason|> Natalia sold clips to 48 of her friends in April, and then she sold
half as many clips in May. How many clips did Natalia sell altogether?
<|think|> Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
<|answer|> 72 <|end|>
```

Same plug-in shape as the others — one fresh FFN expert per block, trunk frozen,
three new tag tokens — but with three differences that matter:

**Prompt-masked loss.** The question is context, not a prediction target, so the
loss is taken only over the `<|think|>`/`<|answer|>` span. The corpus therefore
ships as *two* parallel binaries, token ids and a per-token supervision mask —
the same trick [expert.py](sodachat/expert.py) uses to carry task ids and action
targets alongside the tokens. Train on the question too (what a plain packed LM
stream does) and most of the gradient goes into learning to write questions.

**An answer *token*.** `<|answer|>` is a token, not a phrase, so the final answer
is recovered by splitting on token ids rather than by parsing prose — which
matters because the tokenizer drops special tokens when it decodes, and because
at this scale the reasoning often wanders before it lands.

**A much larger corpus.** The earlier specialists each trained on one dataset of
a few thousand examples (MNIST, CIFAR-10, ~24k CodeSearchNet snippets). Reasoning
doesn't survive that diet — step-by-step derivation has to be seen in bulk and in
many phrasings. So the corpus is streamed and interleaved from **eight public
datasets**, all permissively licensed:

| dataset | license | rows | what it adds |
|---|---|---|---|
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | CC-BY-4.0 | 22M | the bulk: plain step-by-step prose with the answer in its own field |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (`main` + `socratic`) | MIT | 7.5k×2 | gold grade-school word problems, in plain and self-questioning style |
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | MIT | 395k | bulk short step-by-step math |
| [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) (CoT only) | MIT | 262k | terse multi-choice chains |
| [orca-math-word-problems](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | MIT | 200k | conversational worked solutions |
| [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Apache-2.0 | 860k | competition math, `\boxed{}` answers |
| [AQuA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat) | Apache-2.0 | 97k | quantitative multiple choice with rationales |
| [StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA) | MIT | 1.6k | non-arithmetic implicit multi-hop yes/no |

That builds to **~1.6M reasoning traces / 400M tokens** — three orders of magnitude
more examples than any earlier specialist. Sources are read with `streaming=True`
and tokenized straight into the cached `.bin` stream, so the corpus costs ~1.2 GB
on disk where the raw parquet would be over 12 GB, and never lands on the training
box whole.

The size is chosen to match the **training schedule**, not to be as large as
possible: the default run sees 20k × 24 × 512 = 246M tokens, so a 400M-token corpus
means each example is seen ~0.6 times and the model never gets to memorize one.
These sources could supply billions of tokens; the rest would simply never be read.

**What was measured and rejected**, so nobody re-litigates it — every one of these
looked good on its dataset card:

| rejected | why |
|---|---|
| [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) (6.9M) | solutions are `<llm-code>` Python blocks. A 97% keep-rate that is really 97% code — `codegen`'s job, and the wrong output shape here |
| [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) (896k) | only 22% survives: olympiad proofs whose `answer` field is the word "proof" |
| [orca-agentinstruct-1M](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1) (1M) | the best hope for non-math breadth: keeps **0%** of `analytical_reasoning`, 2–12% elsewhere, and the survivors are quantitative anyway |
| [CoT-Collection](https://huggingface.co/datasets/kaist-ai/CoT-Collection) (1.8M) | the one set that would have fixed the math skew — script-only, so unloadable under `datasets` 5.x, with no parquet mirror |
| [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) (3.2M) | R1 traces average ~20k characters against this window's ~1.2k |
| camel-ai/math, facebook/natural_reasoning | non-commercial licenses |

So the mix is **~99.9% quantitative**. StrategyQA is the only non-arithmetic source
that is both permissively licensed and loadable, and it has 1,603 usable rows. In
practice this is a *quantitative* reasoner — ask it a general-knowledge yes/no
question and you get confident nonsense, because nothing in its diet looks like
that. Fixing it needs a large non-math CoT corpus that doesn't currently exist in
loadable, permissive form, not a reweighting of this one.

```sh
python -m sodachat.reason data     # build/inspect the corpus cache only
python -m sodachat.reason train    # needs models/expert.pt; trunk stays frozen
python -m sodachat.reason eval     # held-out perplexity + GSM8K exact match
python -m sodachat.reason demo     # a few questions + chains of thought
python -m sodachat.reason think --question "..."
```

**The held-out split is keyed on problem *identity*, not question text.** A 1-in-40
hash of the question would look disjoint and still leak badly here: MetaMathQA
augments each source problem into many rephrasings, and ~89k of MathInstruct's CoT
rows are AQuA-RAT problems that differ only in how the choices are spelled
(`Options: A)21` vs `Answer Choices: (A) 21`). Either way the *same problem* would
land in train under one phrasing and in val under another, and the held-out
perplexity would be quietly scoring memorization. So the split hashes a normalized
identity — choice list stripped, and MetaMathQA's `original_question` in place of
its rephrasing — which keeps every variant of a problem on one side.

### Measured: what 4x the data actually bought

Both models scored on the *same* held-out stream, and on the same 200 GSM8K **test**
problems (never trained on):

| corpus | traces | steps | val ppl | GSM8K exact match |
|---|---|---|---|---|
| 100M tokens | 412k | 5,500 | 5.0 | 4/200 = **2.0%** |
| 400M tokens | 1.42M | 20,000 | **3.5** | 3/200 = **1.5%** |

**Perplexity improved a lot; answer accuracy did not move.** 4 hits versus 3 on the
same 200 problems is a one-problem difference — noise, not a regression, and not an
improvement either. Nor can it be resolved by measuring harder: at ~2% accuracy,
separating 1.5% from 2.0% needs thousands of problems, and GSM8K's test split only
has 1,319.

The gain is real but it is in *fluency of reasoning*, not correctness. At matched
compute (step 5500, same val stream) the larger corpus was already ahead, 5.0 → 4.7,
so some of it is data diversity rather than the extra steps. And the chains genuinely
got better structured — it now picks the right operations and often computes them
correctly:

```
Q: A shirt costs $15 and jeans cost twice as much. How much do both cost?
   "the jeans cost 2 * 15 = $30. The total for both is $15 + $30 = $45."   ✓ correct
Q: If a train travels 60 miles in 2 hours, what is its average speed?
   "the average speed is 60 / 2 = 30 miles per hour"                       ✓ correct
   answer: "$30 / 30 = 1.5$ miles per hour"        ← mangled a right answer
Q: 3 boxes of 7 pencils, gives away 5. How many left?
   "3 * 7 = 21 pencils"                                                    ✓ correct
   "21 - 5 = 12 pencils left"                      ← right operation, bad arithmetic
```

Two failure modes remain, and they are what exact-match punishes: arithmetic slips,
and a spurious *extra operation inside the answer slot* after the right value was
already derived. The second is a data artifact worth fixing — prose answers still
reach the corpus from orca-math (~3.5% of it), which teaches the model that reasoning
may continue past `<|answer|>`.

**The honest conclusion: the binding constraint is the frozen ~14M trunk, not the
corpus size.** More data made it a better model *of* reasoning text and did not make
it better at arithmetic. Going from 400M to 4B tokens would be expected to do the
same again. Getting real accuracy needs a bigger or unfrozen trunk, or a tool the
model can call to do the arithmetic — not more tokens.

**Read the score honestly.** `eval` reports GSM8K *test* exact-match next to
perplexity, because perplexity flatters a model like this and accuracy doesn't. The
table above is why both are printed. The trunk is a ~14M-parameter model frozen on a
chat/game diet with a dialogue BPE vocabulary: it learns the *shape* of reasoning
— the format, the moves, the arithmetic patois — far better than it learns to be
right. Expect fluent-looking derivations with wrong totals.

Once trained, the expert loads it automatically: `/think <question>` works it out
in the agent, and questions in plain chat that clearly want working-out ("how
many are left if I have 12 apples and eat 3?") route here on their own. The
trigger is deliberately conservative — chat is the default, and hijacking small
talk to "reason" about it reads worse than missing a word problem.

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
`step` plus a scripted `expert` for the training data. A grid game also sets
`GLYPHS` (pretty Unicode for the terminal) and `MODEL_GLYPHS` (one ASCII byte
per cell value for the board the LM reads — `.` empty, `@` the thing you
control, `#` an obstacle, `*` the objective; see below). Decorate it with
`@register` and it's immediately trainable, playable, and available in the
agent — nothing in the tokenizer, trainer, or UI is game-specific. See
[games/snake.py](sodachat/games/snake.py) (grid) and
[games/tictactoe.py](sodachat/games/tictactoe.py) (text) for the two patterns.

**A board for the eye and a board for the model.** The terminal board
(`render()`, `GLYPHS`) uses box-drawing/block glyphs that look good but cost
2–3 bytes each — a 20×20 board tokenizes to ~800 subwords, which overflowed the
expert's context and truncated the goal header right off the front, and it drew
snake's head and body with the *same* glyph (they differed only by colour,
which the text board drops), so the LM couldn't even see where its head was. So
the LM movers read `model_board()` instead: one ASCII byte per cell
(`MODEL_GLYPHS`), ~100 tokens for a 20×20 board, every value distinct. The
convention (`@` controlled entity, `*` objective) is shared across games, so
snake's `@`/`*` and the sandbox probe's `@`/`*` line up and instruction-
following transfers. The specialist per-game models are unaffected — they
encode cell *values* directly (one token per cell, the table above).

## Layout

The package has a **MODEL MAP** at the top of [sodachat/__init__.py](sodachat/__init__.py)
listing every model with its architecture, inference class, and checkpoint. The
short version: `blocks.py` is the shared toolkit (no single model owns it),
`model.py` holds the base `MiniGPT` that most models reuse, and each model's
data/training/inference lives in its own file below.

```
sodachat/
  corpus.py       # load + clean the NPS Chat corpus
  data.py         # dialogue dataset loaders (SODA, DailyDialog, NPS)
  blocks.py       # SHARED toolkit: RMSNorm/RoPE/attention/SwiGLU/Block, GPTConfig,
                  #   tokenizers, pick_device, pad_load — every model builds on these
  model.py        # base decoder LM (MiniGPT) + chat model (MiniChatLM) + checkpoint I/O
  train.py        # chat-model training -> models/minigpt-soda.pt
  unified.py      # one 30M dense model on the whole mixture -> models/unified.pt
  instruct.py     # VLA-style instruction post-train -> models/unified-instruct.pt
  expert.py       # task-routed experts: 1 model, separate game/chat FFNs -> models/expert.pt
  vision.py       # image-recognition specialist (MNIST digits + CIFAR-10 objects),
                  #   a frozen-trunk expert add-on -> models/specialist-vision.pt
  code.py         # code specialist (language ID + completion), CodeSearchNet,
                  #   a frozen-trunk expert add-on -> models/specialist-code.pt
  codegen.py      # code-generation specialist (next-token LM over code),
                  #   a frozen-trunk expert add-on -> models/specialist-codegen.pt
  reason.py       # reasoning specialist (step-by-step then answer), 7 public CoT
                  #   datasets -> models/specialist-reason.pt
  narrate.py      # multi-head model (MultiHeadGPT): action head + LM commentary in one pass
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
                  #   + sandbox (a no-train VLA test grid)
                  #   + versus (multiplayer snake: you vs. the bot, reusing the solo model)
```
