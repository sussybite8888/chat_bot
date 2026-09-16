# sodachat — a small neural network you train yourself

A tiny GPT (~1–14M parameters), trained from scratch on your own machine, put
to two uses that lean on its speed and small size:

- **A chatbot** with a terminal UI, a Discord bot, a Google Chat app, and a
  browser page sharing one engine. The bots share one *loaded* copy of it too,
  through a small API server that holds the models (see
  [One model, many bots](#one-model-many-bots)); the browser page skips the
  server entirely and runs the model client-side via ONNX Runtime Web (see
  [In the browser](#in-the-browser-onnx-runtime-web)).
- **A game controller** — the same architecture, small enough to pick an
  action every frame, trained to play Snake, Pong, Dodge, and Tic-Tac-Toe. One
  agent both chats and plays (see [Playing games](#playing-games)).

Two model backends:

| Backend | What it is | Notes |
|---|---|---|
| `mini` (default) | A ~14M-param GPT (RoPE / RMSNorm / QK-norm / squared-ReLU FFN / logit softcap) with an 8k BPE subword vocabulary, both trained **from scratch** on [SODA](https://huggingface.co/datasets/allenai/soda) (~1.2M narrative-grounded dialogues, ~210M tokens) blended with [Pre-1929 Books](https://huggingface.co/datasets/common-pile/pre_1929_books_filtered) prose and your own `data/`, under [Muon](https://kellerjordan.github.io/posts/muon/) + a warmup-stable-decay schedule | Wants a GPU: ~24h. No pretrained weights anywhere. |
| `gpt2` | GPT-2 (124M) fine-tuned on dialogue data | Opt-in: never started automatically |

Select with `--backend` (CLI) or `SODACHAT_BACKEND` (Discord / Google Chat).
Other datasets: `--dataset dailydialog`, or `--dataset nps` (char-level) for
vintage 2006 chat-room flavor.

**Why SODA and not something smaller.** A 14M-param model needs *at minimum*
roughly 20 tokens per parameter ([Chinchilla](https://arxiv.org/abs/2203.15556))
— about 280M tokens. DailyDialog supplies 1.5M, i.e. **0.1 tokens/param, ~200×
too few**. That deficit is what "grammatical but irrelevant" actually looks like:
[TinyStories](https://arxiv.org/abs/2305.07759) found grammar saturates early
and cheaply, while *using the context* is the last ability to emerge and is
the most data-hungry. SODA's ~210M tokens clear that bar in a single pass, and
its dialogues are grounded in a narrative, so turns actually respond to each
other.

The default schedule then runs **~4 epochs** of it, ~60 tokens/param. Chinchilla
is a *training*-compute-optimal ratio — it answers "best loss per GPU-hour
spent training", which is the wrong question for a model that gets trained once
and then run forever. Repeating a corpus up to ~4 times is worth nearly as much
as fresh data ([Muennighoff et al.](https://arxiv.org/abs/2305.16264)), so the
extra epochs cost wall-clock and nothing else. Use `--steps` to trade quality
back for time.

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
- **Reply trimming** — generations are cut to whole sentences; sampled tails
  tend to wander. How much is one setting, `--reply-length short|medium|long`
  (CLI), `SODACHAT_REPLY_LENGTH` (Discord / Google Chat) or `/length` (agent
  REPL), which moves the generation budget and the trim together — raising the
  trim alone only re-cuts a reply the model was already stopped from finishing.
  At `long` the trim stops being the binding constraint (2% of replies, against
  38% at `medium`); past that the limit is the model's own turn length, since
  SODA turns are short.
- **Personality** — the model has no system prompt (it is a from-scratch
  `A:`/`B:` dialogue LM, so "be cheerful" is not an instruction it can
  follow), so a persona is built from the things that do reach it: a short
  example exchange prepended to the conversation, the sampling temperature,
  the MMI λ above, and a light post-process. `--persona` (CLI),
  `SODACHAT_PERSONA` (Discord / Google Chat) or `/persona` (agent REPL and
  any channel). See [Personality](#personality).
- **Output filtering** — a profanity filter is applied to replies by default.
  Disable with `--unfiltered` (CLI) or `SODACHAT_UNFILTERED=1`.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in tokens as needed
```

Datasets download automatically on first use: SODA, DailyDialog and Pre-1929
Books from the Hugging Face hub, NPS Chat via NLTK (`pip install nltk`, only
needed for `--dataset nps`).

## Training

```sh
.venv/bin/python -m sodachat.train                    # mini-GPT on SODA (~24h on a GPU)
.venv/bin/python -m sodachat.train --dataset dailydialog   # small/fast, lower quality
.venv/bin/python -m sodachat.finetune                 # GPT-2 (needs >=16GB / GPU)
```

Each corpus is tokenized once into a flat `uint16` file under `models/`
(`soda+books+local-dialog-train.bin`, ~420MB, and one file per stream beside it)
and memory-mapped during training, so RAM use stays flat regardless of corpus
size. That step takes ~15 min and is cached. Checkpoints
(`models/minigpt-soda.pt`) keep the best validation loss, with the BPE
vocabulary stored inside.

Flags: `--dataset soda|dailydialog|nps`, `--steps`, `--batch-size`, `--lr`,
`--device`, `--out`, `--seed`, `--data-dir`, `--no-local-data`,
`--books-tokens`, `--local-end-weight`, `--local-start-weight`,
`--local-ramp-frac`.

### Books: the English the dialogue corpora assume

A chat corpus teaches turn-taking, not English. SODA's turns are short, modern
and machine-written, and a model fed nothing else is fluent in chat and thin
everywhere else — which at 77M parameters mostly shows up as a small vocabulary
and collapsing syntax on anything longer than a sentence.

So the run also trains on [Pre-1929
Books](https://huggingface.co/datasets/common-pile/pre_1929_books_filtered)
(Common Pile v0.1): ~130k US books published before 1929, in the public domain
since 2024, OCR'd by the Internet Archive for HathiTrust. Long-form edited prose,
and permissively licensed — the same bar the rest of the corpora clear.

It is 26 gzipped shards, ~19.5GB, so a run takes a slice off the front rather
than the lot: `--books-tokens` (default 150M, `0` disables) is a budget, and the
shards stream straight off the wire — decompressed line by line, nothing but the
tokens written to disk. The last shard is held out, so validation prose is books
the run never saw.

The text is OCR of printed pages, which needs undoing before it is English:
lines are hard-wrapped at the column width (feed that in raw and the model
learns to break a line every seventy characters), words are split across line
ends with hyphens, and every book carries title pages, running heads, page
numbers and an index. `data.clean_book_text` rejoins the paragraphs and
`_is_prose` drops the furniture — a paragraph of fewer than three words, under
80% letters, or without a single lowercase character is page furniture, not
prose. What survives is cut into ~4000-character passages on paragraph
boundaries, each an untagged document like your own files.

### Your own training data (`data/`)

Everything above trains on datasets fetched from Hugging Face. To train on your
own files instead, drop them in [data/](data/) — plain text and source, no
loader to write and no conversion step. The **file extension** decides which
model a file feeds:

```
data/
  text/     .txt .md .rst ...        -> the chat model     (sodachat.train)
  code/     .py .js .ts .go .rb .php .java
                                     -> the code generator (sodachat.codegen)
```

```sh
.venv/bin/python -m sodachat.localdata          # what would be picked up, and what's skipped
.venv/bin/python -m sodachat.train              # text files mixed into the dialogue stream
.venv/bin/python -m sodachat.codegen train      # source files mixed into CodeSearchNet
```

Text files become untagged documents, each terminated by `<|endofdialog|>` so
the document-boundary attention mask (below) keeps one file from bleeding into
the next; ~8% are held out for validation. The tokenized cache is fingerprinted
against the folder, so editing a file re-tokenizes on the next run rather than
silently training on the old copy.

**Your data gets the end of the run.** Dialogues, books and `data/` are
tokenized to separate files and each batch is drawn from all three, which means
the blend is a dial rather than a consequence of how many bytes each one has.
`data/` is the corpus the model is ultimately for and also the smallest by
orders of magnitude, so it is scheduled: it keeps its natural token share for
the bulk of training and then ramps to `--local-end-weight` (default 5% of each
batch) over the final `--local-ramp-frac` (default 0.2) of the run — the same
stretch the WSD schedule spends decaying the learning rate. That is the cheapest
voice a small corpus can buy. The weights barely move once the rate has decayed,
so what arrives during the decay is what the finished model sounds like, while
the big corpora stay in the mixture throughout and the ending is a shift in
emphasis rather than a fine-tune that forgets.

The trade is repetition: a few hundred KB of your own writing, at 5% of a batch
for 32k steps, is thousands of passes over the same files, and a model will
memorize what it sees that often. The startup log prints exactly how many passes
your `data/` works out to and warns past 100, so tune `--local-end-weight` to it
rather than the other way round. Validation is deliberately *not* measured at
the training blend — it stays at the natural one all run, so "best val loss so
far" compares like with like, and the per-stream losses are logged next to it
(`val 2.417 (dialog 2.31 books 3.02 local 2.88)`). Source files go in under the same language headers (`# python`,
`// javascript`) and the same machine-generated-code filter as the rest of the
codegen corpus; an extension outside the list above is skipped rather than fed
in untagged. Details and the full skip list: [data/README.md](data/README.md).

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

`sodachat.cli` flags: `--backend mini|gpt2`, `--reply-length short|medium|long`,
`--persona <name>` / `--personas` (list them), `--plain` (hide reply metadata),
`--unfiltered`, `--seed N`.

## Personality

How the bot *sounds* is a named persona, and changing it costs nothing —
no retraining, no reload, no restart:

```sh
.venv/bin/python -m sodachat.cli --personas               # what's available
.venv/bin/python -m sodachat.cli --persona deadpan        # plain chat
.venv/bin/python -m sodachat --persona grumpy             # the agent
```

```
/persona            # in the agent, a Discord DM or a Google Chat room: list
/persona cheerful   # ...and switch, mid-conversation
```

In the chat frontends the starting persona is `SODACHAT_PERSONA` in `.env`;
`/persona` then changes it per channel, so one room can be deadpan while
another is cheerful.

Built in: `neutral` (the model as trained, the default), `cheerful`,
`deadpan`, `curious`, `grumpy`, `lowkey`.

**How it works.** There is no system prompt to write a personality into — the
chat model is a from-scratch `A:`/`B:` dialogue LM. A persona is instead four
knobs that do reach the model ([sodachat/persona.py](sodachat/persona.py)):

| knob | what it does |
| --- | --- |
| `primer` | an example exchange prepended to the conversation, so the model's freshest evidence of how `B` talks is `B` talking that way |
| `temperature` | how far the sampler strays from the safe reply — deadpan low, chaos high |
| `mmi_lambda` | the relevance/genericness trade-off in the reranker; raise it and only a reply to *this* message will do |
| `lowercase`, `closers` | a light post-process, because a model this size cannot be talked into a verbal tic |

The primer sits at the front of the prompt, which is also what gets trimmed
first when a long conversation overflows the context window — so primers stay
to a couple of exchanges and a persona fades rather than fights for room. It
is deliberately kept out of the MMI null baseline (`null_prompt`), which is
"what would this model say with no context at all"; priming both sides would
cancel the persona back out of the score.

**Writing your own.** [personas.json](personas.json) in the repo root is
merged over the built-ins (reuse a built-in name to override it) and re-read
when it changes, so an edit lands on the next message:

```json
{
  "pirate": {
    "description": "nautical, overfamiliar",
    "primer": [
      ["how's it going?", "ahoy! fair winds today, matey, and no complaints from me."],
      ["i had a rough day", "arr, rough seas happen. sit ye down and tell the tale."]
    ],
    "temperature": 0.85,
    "mmi_lambda": 0.7,
    "lowercase": false,
    "closers": [", matey.", " arr."],
    "closer_chance": 0.3
  }
}
```

Only `description` is required; every other field falls back to the neutral
default. `SODACHAT_PERSONAS` points at a different file.

A caveat worth setting expectations with: this is a 14M-parameter model
trained on SODA. A persona reliably moves tone, length and register — it will
not turn the bot into a different character, and the further a persona is from
the training distribution (the pirate above), the more it leans on the closers
to do the work.

## One model, many bots

Every frontend used to load its own checkpoints. Running the Discord bot and the
Google Chat app meant **two** copies of a ~190MB model set, two warm-ups, and
two processes holding weights for conversations that were never going to decode
at the same instant anyway. So the models moved into a server of their own and
the bots became clients of it:

```
    discord_bot ─┐
    google_chat ─┼─ HTTP ─> sodachat.api ─> Rooms ─> SodaAgent per room ─> one model
    your own    ─┘
```

Start the models:

```sh
.venv/bin/python -m sodachat.api          # http://127.0.0.1:8765
```

Then point as many bots at it as you like — one line of environment each:

```sh
export SODACHAT_API_URL=http://127.0.0.1:8765
.venv/bin/python -m sodachat.discord_bot
.venv/bin/python -m sodachat.google_chat
```

Measured on the training box, agent mode on CPU: the server is **881MB** RSS and
takes ~40s to warm up; a bot alongside it is **51MB** and ready in ~8s. A second
bot adds another 51MB instead of another copy of the models. Nothing but the
server imports PyTorch — `sodachat.client` and `sodachat.transport` have no model
dependency, and importing the package no longer pulls one in either.

**Leave `SODACHAT_API_URL` unset and nothing changes**: the bot loads the models
into its own process, exactly as before. That is still the right answer for one
bot on one machine. `client.py` picks between the two and hands the frontend the
same `reply()` either way, so neither bot has a branch for it.

### What the server is

`Rooms` ([rooms.py](sodachat/rooms.py)) — one `SodaAgent` per room over one copy
of the checkpoints — behind five endpoints:

| | |
|---|---|
| `GET /healthz` | whether the models are loaded (unauthenticated, for probes) |
| `GET /v1/info` | device, mode, and which attachments it will accept |
| `POST /v1/reply` | `{room, text, attachments?, reset?}` → `{text, seconds, actions}` |
| `POST /v1/watch` | `{room, text}` → `{actions}` for a message the bot wasn't sent — a reaction, usually nothing. Runs no model |
| `GET /v1/rooms` | the live conversations |
| `POST /v1/rooms/reset` | forget one room (or all of them) |

Rooms are namespaced by the frontend that owns them — `discord:123`,
`googlechat:spaces/AAA` — because one server now holds every bot's
conversations, and two transports' ids are not from the same space. Each still
gets its own history, its own running game, its own persona.

Attachments ride along as base64 in the JSON: the bot downloads the file from
its own transport, the server writes it into that room's directory and names it
in the message, and the agent then sees exactly what a dropped file looks like in
a terminal. Same code path as before, one machine further away.

Generation stays serialized on `Rooms`' lock — one model, one decode at a time —
so concurrent requests queue rather than contend. Four at once on CPU came back
in 3.7/7.8/11.9/16.3s, which is four ~4s replies in a row and no failures.

### What a room is allowed to read — and write

The agent reads files: `/see photo.png` classifies an image, `/code app.py` names
a language, and a path sitting in a message is picked up and handled by what it
*is*. In a terminal that is exactly right — the person typing owns the machine,
and `/see ~/Desktop/cat.png` is no more access than `cat` would be.

Through a bot it is not, and a *shared* server makes it worse: whoever can
message any bot could otherwise walk the host filesystem — `/code /etc/passwd`,
`/see ~/.ssh/...`, `/code .env complete` reading back a token — and the reply
hands over what it found. So file access is a policy the frontend picks
([files.py](sodachat/files.py)), not something the agent decides:

| | reads | writes |
|---|---|---|
| `FileAccess.nothing()` | nothing — **the default**, so a frontend that forgets to think about this is closed rather than open | nothing |
| `FileAccess.rooted(dir)` | only what's under `dir`, symlinks resolved | the same directory |
| `FileAccess.anywhere()` | anything the process can; passed explicitly by the terminal agent, and by nothing else | **only its workspace** |

Each room gets a directory of its own and its agent is rooted there, and that
directory is also where it may **write**: `/write notes.txt ...`, `/append`,
`/read`, `/rm`, `/files` — which the model can reach for itself, so "remember
that" can become a file rather than a promise. The attachments of the message
being answered are staged there and deleted when the turn ends; what the agent
writes itself stays until the room is reset. So a room can read what it sent
and what it has written down, and nothing else: not another room's files, not
the server's, not yours. `/v1/info` reports `"file_access": "per-room sandbox
(read and write)"`, so the other end can check the promise.

**Reading and writing are separate policies, and writing is always the narrower
one.** A terminal's user owns the machine, so `/code ~/notes.md` there is no
more access than `cat` — but the model can now *start* an operation rather than
only answer one, and a sampled `[[rm ~/.ssh/id_rsa]]` is nobody's idea of a
feature. So even the terminal agent writes in one place and one place only:
`workspace/` in the repo, created on first use, where you can go and look at
what it kept.

The workspace is quota'd — 256 KB a file, 4 MB and 64 files a room — because a
model that gets stuck in a loop writing files is a thing that happens, and a
full disk on the box holding the models takes every room down with it.

Everything a prober would try is refused, and refused the same way — an answer
that said "no such file" for one path and "not allowed" for another would be an
oracle for what exists on the host:

```
/code /etc/passwd                 → I can only read files sent to me in this conversation.
/code ~/.ssh/id_rsa               → (the same)
/code ../../../../etc/passwd      → (the same)
/code sodachat/api.py             → (the same)
/see /dev/zero                    → (the same)
/code mine.py     (just uploaded) → That's python — I'm 99% sure.

/write ../../../tmp/pwned x       → I can only write inside my own folder.
/write /etc/passwd x              → (the same)
/write ~/x                        → I can't write to a home directory.
/write link.txt x   (a symlink planted in the sandbox, pointing out)
                                  → I can only write inside my own folder.
/rm somedir                       → only files, never a directory tree
/write notes.txt hello            → wrote notes.txt (5 B).
```

Checked at the policy level too: another room's directory, `..` in every
spelling, `~`, a symlink planted inside the sandbox pointing at `/etc/hosts`, a
directory, a device file, and hostile upload filenames (`../../../../tmp/pwned.py`
lands as `pwned.py` inside the room and nowhere else). A bare path in ordinary
text just isn't recognized and the message routes to chat — someone saying
"check app.py" in a channel wants conversation, not a lecture about sandboxes.

Talk to it without a bot:

```sh
curl -s localhost:8765/v1/info
curl -s -X POST localhost:8765/v1/reply -H 'Content-Type: application/json' \
  -d '{"room":"dev","text":"hey there"}'
# -> {"room":"dev","text":"hi!","seconds":1.4,"actions":[{"name":"react","arg":"👋"}]}
#    `actions` is what the *frontend* should do in the room (see Tools, below)

.venv/bin/python -m sodachat.client --room dev "write me a python function"
.venv/bin/python -m sodachat.client --room dev --file cat.png "what is this?"
```

**Securing it.** The default bind is `127.0.0.1`, because this endpoint runs a
model for whoever reaches it. To serve bots on other machines, set
`SODACHAT_API_KEY` — every route but `/healthz` then requires it, as
`X-API-Key: <key>` or `Authorization: Bearer <key>`, and the clients send it —
and bind with `--host 0.0.0.0` (the server warns if you do the second without
the first). Run it behind TLS if it leaves the machine.

## Discord

1. Create an application at <https://discord.com/developers/applications>,
   add a **Bot**, and copy its token into `.env` as `DISCORD_BOT_TOKEN`.
2. On the Bot page, enable the **Message Content Intent** (Privileged
   Gateway Intents).
3. Invite the bot: OAuth2 → URL Generator → scope `bot` → permissions
   *Send Messages*, *Read Message History*, *Add Reactions* → open the
   generated URL. Everything the bot does by default needs only those; the
   tools that want *Manage Messages*, *Manage Nicknames*, *Manage Roles* or
   *Manage Channels* are off until you say otherwise (see
   [Tools](#tools-reacting-pinning-and-calling-its-own-commands)).
4. Run it:

   ```sh
   # with the models in their own process (see One model, many bots)
   SODACHAT_API_URL=http://127.0.0.1:8765 .venv/bin/python -m sodachat.discord_bot

   # or on its own, models loaded here
   .venv/bin/python -m sodachat.discord_bot
   ```

The bot replies to DMs and @mentions. Set `DISCORD_RESPOND_ALL=1` to reply to
every message it can read.

**It runs the whole agent, not just chat.** A channel gets what the terminal
gets: the [routing specialist](#a-routing-specialist-deciding-which-of-them-answers-you)
picks which capability answers each message, and `/help`, `/play snake`,
`/gen`, `/think`, `/route`, `/model`, `/persona` all work. Three things are
specific to a chat room:

* **Post an image and it gets looked at**; post a `.py`/`.js`/… and it gets
  read. This is the room equivalent of dropping a file into the terminal — the
  bot downloads the attachment, the models' side writes it into that channel's
  own directory (the only place that channel can read from, see
  [What a room is allowed to read](#what-a-room-is-allowed-to-read)), the agent
  recognizes it as a path, and the vision or code specialist takes it. Follow-ups work the same way too (*"what was in that
  picture?"*). Files it can't use get a reason rather than silence, and it asks
  the models what they accept before spending the bandwidth
  ([client.py](sodachat/client.py)).
* **Each channel is its own conversation** — its own history, its own running
  game, its own memory of the last image seen — keyed `discord:<channel id>`
  ([rooms.py](sodachat/rooms.py)). A second channel costs 0 MB, measured, and a
  second *bot* costs 51MB rather than a second copy of the models.
* **Boards and code are fenced, long replies are split.** A 20×20 snake board or
  a `/help` table is column-aligned text that a proportional font destroys, so
  the block paragraphs (and only those) go in a code fence, and anything over
  Discord's 2000-character limit is split without leaving a fence unclosed. That
  is the transport's business, not the model's, so it happens here
  ([transport.py](sodachat/transport.py)).
* **A turn can do something, not just say something** — react to your message,
  pin it, run one of its own commands. That is the next section.

Generation runs off the event loop, one reply at a time, so the gateway
heartbeat and the typing indicator keep going while the model writes. Set
`SODACHAT_AGENT=0` where the models load for the old plain-chat behaviour.

### Tools: reacting, pinning, and calling its own commands

Everything above comes out as words. A server is not only words — a message can
be reacted to, pinned, split into a thread — and a bot that can only post
paragraphs sits outside half of how a channel actually talks. So a turn now
produces two things: the text to post, and the **acts** it wants taken
([actions.py](sodachat/actions.py)).

```
you ›  lol that snake run was cursed
bot ›  😂  (a reaction on your message)
       "it went where the food was. mostly."
```

| tool | what it does | on by default |
|---|---|---|
| `react` / `unreact <emoji>` | react to a message (*Add Reactions*) | **yes** |
| `say <text>` | post a message of its own | **yes** |
| `delete` | take back the last thing **it** said | **yes** |
| `dm <text>` | answer privately instead | no |
| `pin` / `unpin` | pin the message being answered (*Manage Messages*) | no |
| `thread <name>` | start a thread on it (*Create Public Threads*) | no |
| `nick <name>` | rename **itself** here (*Change Nickname*) | no |
| `rename <name>` | rename whoever it is answering (*Manage Nicknames*) | no |
| `role` / `unrole <name>` | give or take a role, by name (*Manage Roles*) | no |
| `topic <text>` | set the channel topic (*Manage Channels*) | no |
| `slowmode <seconds>` | set the channel's slowmode (*Manage Channels*) | no |
| `status <text>` | what it is shown as playing | no |

Turn the rest on per bot, in `.env`:

```sh
DISCORD_ALLOWED_ACTIONS=react,unreact,say,delete,pin,rename   # or: all, or: none
```

`rename`, `role` and `unrole` act on **whoever the bot is answering** — the
model cannot pick someone else, because mentions reach it as the word
`@someone` with the id scrubbed out ([discord_text.py](sodachat/discord_text.py)),
which is exactly why that scrubbing is there. Typed by hand the mention *is*
resolved, so `/rename @bob stinky` renames bob and `/role @bob regular` gives
them the role. `role` is the one tool here that can hand out power: Discord
stops a bot assigning anything above its own role, and it ships off.

**What is deliberately missing: kicking, banning, timeouts, and deleting other
people's messages.** There is no environment variable for them — the refusal is
the point. Those are punishments and demolition rather than conversation, and a
14M-parameter model that picks its words by sampling has no business holding
that end of the stick. `delete` takes back only what the bot itself said.

**The models never touch Discord.** They run in the API server, which holds no
gateway connection — so a turn *names* an act and the bot process performs it
([`perform`](sodachat/discord_bot.py)). That split is why the same reply works
in Google Chat (which has no app-callable reactions: the acts are dropped and
the words stand alone) and in the terminal, where `/react 🔥` prints what a
channel would have done. A frontend performs only what
`DISCORD_ALLOWED_ACTIONS` lists, and an act that fails — missing permission, a
thread that already exists, an emoji this server doesn't have — is logged with
the reason and never costs the reply.

**Where the act comes from.** Two paths, and the second one is the honest one:

* **The model asks.** Anything it generates is scanned for a call in double
  brackets — `[[react :kekw:]]`, `[[pin]]`, and also `[[play snake]]`, because
  it has the same `/command` list you do. The call is lifted out of the text
  (you never see the brackets) and the command's output joins the reply. The
  from-scratch model does not write these yet; the plumbing is here so that
  *training* it to is the only missing step, and the instruct/gpt2 backends can
  already stumble into one.
* **Nothing asked, so a trigger table picks.** `pick_reaction` reads the
  incoming message — laughter, thanks, a greeting, a *"rip, it's down"* — and
  falls back to whatever the [routing specialist](#a-routing-specialist-deciding-which-of-them-answers-you)
  labelled the message (`reason` → 🤔, `codegen` → 💻, `game` → 🎮). This is
  the same arrangement routing itself was in before `route.py` was trained:
  hand-written rules standing in for a specialist nobody has trained yet, kept
  as a table of (pattern → slot) pairs so a reaction specialist could replace
  `pick_reaction` and nothing else. **Most messages get no reaction**, on
  purpose — a reaction under every line means nothing.

It can also **keep files**. `/write notes.txt ...`, `/append`, `/read`, `/rm`
and `/files` work on a folder of the channel's own, and the model reaches for
them through the same `[[write notes.txt ...]]` call — so "remember that" can
become a file instead of a promise. That folder is the channel's sandbox, it is
quota'd, and it goes away when the room is reset: see
[What a room is allowed to read — and write](#what-a-room-is-allowed-to-read--and-write).

Which emoji a slot picks is the **persona**'s business, like everything else
about how the bot sounds ([Personality](#personality)): deadpan answers a joke
with 🙂, ragebaiter with 💀, and a persona that overrides nothing uses the
default table. `/tools` lists what is available and whether the model may reach
for it; `/tools off` leaves it to you and your `/react`.

### It reacts to messages you didn't send it

The bot answers DMs and @mentions. It *watches* everything else: a channel is
mostly people talking to each other, and a bot that only ever reacts to things
said at it is a bot standing in the corner.

```
someone ›  the deploy is down again
   bot   ›  😔        (no reply — just a reaction)
```

That path is built to be boring, because it runs over every message the bot can
read:

* **Nothing is generated.** `Rooms.watch` is `pick_reaction` and a persona — one
  pass over a trigger table, no forward pass, no generation lock. The channel's
  conversation is neither read nor added to, so watching a room never changes
  what it says next when you *do* talk to it.
* **Only reactions can happen unprompted** (`WATCH_ACTS`), whatever the models
  ask for. A bot that starts posting in conversations it wasn't in is a
  different and much worse thing to be.
* **One channel, one reaction, then quiet** for `DISCORD_REACT_COOLDOWN`
  seconds (default 60) — checked in the bot before the request is sent, so a
  busy channel costs nothing at all. `DISCORD_WATCH=0` turns the whole thing
  off.

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

Like the Discord bot, this runs the full agent: routing, `/commands`, games, and
one agent per Chat space — sharing the models with every other bot when
`SODACHAT_API_URL` points at the [API server](#one-model-many-bots), and loading
them here when it doesn't. Two differences from Discord, both forced by the
platform:

* **Attachments aren't read.** Google Chat sends a *reference* to an uploaded
  file, and fetching it needs the Chat API with service-account credentials —
  where Discord hands over a URL the bot can already use. So an upload arrives
  as an empty message with metadata, and the app says that rather than ignoring
  it. `/see <path>` and `/code <path>` can't stand in for it either: a room only
  reads files that were sent to it (see
  [What a room is allowed to read](#what-a-room-is-allowed-to-read)), so this
  frontend has no path to the vision and code specialists at all.
* **One event, one reply**, so there is nowhere to put overflow: a reply past
  4096 characters is cut and marked, instead of split across messages.

Smoke-test without Google:

```sh
curl -s -X POST localhost:8080/ -H 'Content-Type: application/json' \
  -d '{"type":"MESSAGE","message":{"text":"hello there"}}'

curl -s -X POST localhost:8080/ -H 'Content-Type: application/json' \
  -d '{"type":"MESSAGE","message":{"text":"write me a python function that sorts a list"}}'
```

## In the browser (ONNX Runtime Web)

The models also run **client-side**, with no Python and no server round-trip:
export the graphs to ONNX, and [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)
runs them in the page. After the download, the tab works with the network off.

```sh
pip install onnx onnxruntime onnxscript   # export-time only
npm install && npm run vendor             # onnxruntime-web -> web/vendor/

python -m sodachat.export_onnx            # checkpoints -> web/models/*.onnx
python -m sodachat.web --open             # http://127.0.0.1:8000
```

Every model in [the map](sodachat/__init__.py) that has a text head is exported,
and the page offers a panel per capability — chat, a specialist's classifier,
a specialist's generator, and Snake played through the action head:

| Graph | Params | fp32 | What the page does with it |
|---|---|---|---|
| `chat.onnx` | 13.7M | 55 MB | the `mini` chat model, with MMI reranking |
| `expert-text.onnx` | 30.8M | 124 MB | the expert's chat/read expert |
| `expert-game.onnx` | 30.8M | 124 MB | board → move off the action head |
| `expert-route.onnx` | 30.8M | 124 MB | which capability should answer |
| `expert-code.onnx` | 30.8M | 124 MB | language ID off a 6-way head |
| `expert-vision.onnx` | 30.8M | 124 MB | 20-way image label |
| `expert-codegen.onnx` | 30.8M | 124 MB | code continuation |
| `expert-reason.onnx` | 30.8M | 124 MB | step-by-step, then an answer |

Export a subset when you don't want all of it — `--models chat`, or
`--tasks text,route` — and `--quantize int8` cuts each file to roughly a
quarter at some quality cost. The chat UI needs only `chat.onnx`.

**Why the graphs look the way they do.** Three things had to change on the way
out, all of them forced by the browser, and all explained at the top of
[export_onnx.py](sodachat/export_onnx.py):

- **A KV cache.** `MiniGPT.generate` re-runs the whole context per token, which
  is fine on a GPU and hopeless in WASM — a reply is 12 candidates × 48 tokens,
  and at ~3 GFLOP per uncached pass that is minutes of arithmetic. The exported
  graphs take the attention cache in and hand it back grown, so a token costs
  ~30 MFLOP. The prompt is identical across a reply's 12 candidates, so the
  engine prefills once and *forks* that cache per candidate.
- **Explicit attention.** `F.scaled_dot_product_attention` has no counterpart in
  the ORT web build, so the export spells it out as matmul/softmax.
- **One graph per task, not one routed graph.** `RoutedFFN` picks an expert per
  token, but at inference a whole sequence carries one task — so each graph is
  exported with a single expert's weights baked in. Fusing all seven would put
  117M params of experts in every download to use 17M of them.

**The browser half is a port, not a wrapper.** `web/js/` re-implements the
byte-level BPE tokenizer, the sampler (`blocks.warp_logits`), the MMI ranking
(`engine.py`), and Snake's board — because none of that lives in the ONNX graph.
A port that is *nearly* right is a model being fed prompts it never trained on,
so the agreement is tested rather than assumed:

```sh
python -m sodachat.export_onnx           # --check: graphs vs PyTorch, on by default
python tools/dump_tokenizer_cases.py && npm run check:tokenizer
python tools/dump_engine_cases.py  && npm run check:model
```

which covers, respectively: every exported graph against the module it came
from (max |Δlogit| ~2e-5, prefill *and* single-token steps off a cache — and the
export **fails** rather than warns if that drifts); 44 encode/decode cases
against the Python tokenizer; and the JavaScript against PyTorch end to end —
greedy ids exactly, log-probs, classifier confidences and action-head moves to
within fp32 noise.

```
44/44 cases match
chat reply: "Not much, just hung out at home. What about you?" (rel 0.43, 12 candidates)
expert-text reply: "I went to school and then came home." (rel 0.98, 12 candidates)
26/26 checks pass
```

Those run the browser's code under Node, which leaves out what the browser
itself adds — module resolution, WASM instantiation, cross-origin isolation,
the DOM. `npm run check:browser` closes that gap by driving the page in a real
Chrome (optional: needs `puppeteer-core` and a browser, and skips cleanly
without them). It is worth having; both bugs it caught on its first run were
invisible to everything above:

```
chat            webgpu · 4 threads     Chat → "Not much, just ran some errands. You?"
expert-text     webgpu · 4 threads     Chat → "Just hung out at home. You know, the usual stuff."
expert-game     webgpu · 4 threads     Play snake → 20-row board, score 0
expert-code     webgpu · 4 threads     Classify → python · 100.0% confident
expert-codegen  webgpu · 4 threads     Generate → 95 tokens
expert-reason   webgpu · 4 threads     Generate → 117 tokens
expert-route    webgpu · 4 threads     Classify → reason · 100.0% confident
expert-vision   webgpu · 4 threads     Classify → airplane · 70.3% confident
8 models OK
```

`sodachat.web` is a static file server and nothing else — it sends the COOP/COEP
headers that let onnxruntime-web use threads, and the right MIME type for
`.wasm`. WebGPU is used where the browser has it, falling back to WASM.

### Deploying it

`npm run build` packages `web/` into `dist/`, ready to upload anywhere that
serves static files:

```sh
npm run build                        # dist/, models split at 25MB
npm run build -- --models chat       # ship one model instead of all eight
npm run build -- --chunk-size 90MB   # a host with a larger cap
npm run build -- --no-chunk          # leave the .onnx files whole

python -m sodachat.web --root dist   # serve the build locally
```

**Models are split, because hosts cap file size** — Cloudflare Pages at 25 MiB,
which every expert graph is five times over. Each `.onnx` becomes numbered
`.partNNN` files listed in the manifest, and `fetchModel` in
[model.js](web/js/model.js) streams them back into one buffer sized from the
manifest up front. It is the *same bytes*, not an approximation:

```
  chat               55.2 MB  3 parts        largest part: 25,000,000 bytes
  expert-text       124.0 MB  5 parts        sha256 of the parts, concatenated,
  expert-game       124.0 MB  5 parts        equals the original .onnx for all 8
  ...
  52 files, 948.4 MB total
```

A missing or truncated part would otherwise reach onnxruntime as a corrupt
protobuf and be reported as a parse error far from the cause, so the loader
checks the total first and says what it found:

```
Could not load chat: chat.onnx: expected 55232197 bytes across 2 part(s), got 50000000
```

The build also writes a `_headers` file (read by Cloudflare Pages and Netlify,
ignored elsewhere) carrying the same COOP/COEP that `sodachat.web` sends, plus
per-file immutable caching for the chunks and the WASM runtime. Those two
headers are worth keeping: without cross-origin isolation the browser withholds
`SharedArrayBuffer`, onnxruntime-web drops to one thread, and the page is
several times slower with nothing on screen to explain why. On a host that
cannot set headers at all (GitHub Pages), it still works — single-threaded.

Verify a build the same way as the dev tree, by pointing the browser check at it:

```sh
python -m sodachat.web --root dist --port 8736 &
SODACHAT_URL=http://127.0.0.1:8736/ npm run check:browser
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
chat history, so the chat model can keep talking about it. `/persona` changes
how the bot sounds without restarting anything (see
[Personality](#personality)). `/help` lists all commands. Each reply also shows
its speed inline.

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

Actually *writing* code is a third specialist ([codegen.py](sodachat/codegen.py)),
LM-trained on the same six languages, and two details of its diet decide whether
the output reads like code a person wrote:

**Every snippet trains under a language header** — an ordinary comment line
(`// javascript`, `# python`), so still no new tokens. Six languages packed into
one undifferentiated stream leaves `function foo(` as likely to continue in PHP
or Java as in JavaScript, and mixing in a local corpus that was 68% C by bytes
put `#endif` in the middle of JavaScript. The header turns the language into
something generation can *ask for* rather than guess.

**Machine-generated source is filtered out** of both corpora. Minified, bundled,
transpiled and obfuscated code is syntactically valid and licence-clean, so no
other filter rejects it — but it is exactly what teaches a model to write mangled
one-letter names and helper-call soup, which is how a small model ends up
emitting something that looks obfuscated. The shape heuristics (dense lines,
mostly 1-2 character identifiers) apply only to JavaScript and TypeScript: Go
names its receivers `p` and its buffers `b`, and reading that as mangling threw
away 41% of the Go corpus.

Sampling, on the other hand, turned out not to matter — worth recording, because
the opposite is intuitive. Code repeats itself by nature (the loop counter and
the accumulator recur every other line), so the repetition penalty looks like a
suspect: penalize what you just wrote and the sampler has to reach for a fresh
name. Measured over 56 samples × 7 prompts, it isn't one. The share of 1-2
character identifiers is flat across penalties 1.0 / 1.1 / 1.15 (0.176 / 0.174 /
0.176), while degenerate looping falls off steeply (repeated 4-grams 0.024 →
0.023 → 0.009). Mangled output is a training-data problem; 1.15 is simply where
a 17M expert stops looping.

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

### A routing specialist: deciding which of them answers you

Five specialists is four too many for a chain of `if`s. Until now the agent
worked out what a plain-text message wanted with hand-written triggers — a regex
per capability, tried in a fixed order — and the *order* was load-bearing: the
reasoning check had to sit ahead of the code-Q&A check, whose word list contains
`in` and would otherwise swallow every word problem. Each trigger only fires on
the phrasings somebody thought of, and each new capability means re-deriving the
whole ordering by hand.

That cascade is a classifier with hand-tuned weights, so
[route.py](sodachat/route.py) trains the real one — same frozen-trunk plug-in
shape as the rest, a sixth specialist whose job is choosing between the other
five:

```
<|route|>
what language was that file again?
<|dest|>                                -> code (0.94)
```

The destination reads off a 6-way class head at the trailing `<|dest|>` in a
single forward pass — the same mechanism as the vision label, the code language
and the game move. Six destinations, exactly the ones the agent can hand a bare
message to:

| route | goes to | trained from |
|---|---|---|
| `chat` | the chat model (the default) | DailyDialog + SODA + NPS Chat utterances |
| `reason` | `reason.think` | GSM8K, Orca-Math, MetaMathQA, AQuA-RAT questions + generated short arithmetic |
| `codegen` | `codegen.generate` | [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) tasks + CodeSearchNet docstrings, wrapped in request phrasings |
| `code` | the code specialist's last read | templated questions about a file just read |
| `vision` | the vision specialist's last look | templated questions about an image just seen |
| `game` | the reader | templated live-game questions + `reader`'s own question list |

Unlike vision and code, this expert is seeded from the **text** expert rather
than the game one: a user message is ordinary dialogue, not a glyph grid. Its
`<|dest|>` marker is private rather than the shared `<|cls|>` on purpose — when
two specialists share a marker token, whichever attaches first owns its
embedding row and the later one's trained row is discarded, and a classifier
reading its head at exactly that position is the one place that quirk really
bites.

**Three of the six classes are synthesized, and that is the honest weak point.**
There is no public dataset of "messages people send a chatbot, labelled by which
subsystem should answer", so `code`, `vision` and `game` are a few dozen
skeletons crossed with slot values and roughened (fillers, punctuation, casing
that comes and goes) — much easier than real messages. Two things keep the
reported numbers from flattering that. The held-out split holds out whole
**skeletons**, one in six by digest order, so a validation message's *shape* was
never trained on rather than merely its slot values. And `route.PROBE` is ~50
hand-written messages that came from no template at all — that number, not the
held-out one, is the one to believe.

**Chat is the class to protect**, because the two error directions don't cost
the same: missing a request just means a chat reply, while a false positive
answers small talk with generated code. So inference thresholds at 60% and falls
back to chat below it, and `eval` reports the **chat leakage rate** — how much
ordinary conversation gets confidently routed away — as the headline safety
number.

**The old triggers didn't go away.** They run when the router leaves a message
in chat, and they are the whole story until the specialist is trained — so an
agent without `specialist-route.pt` routes exactly as it did before, and
`/route <message>` says so instead of pretending.

```sh
python -m sodachat.route train    # needs models/expert.pt; trunk stays frozen
python -m sodachat.route eval     # held-out + hand-written probe accuracy
python -m sodachat.route demo     # the probe set, message by message
python -m sodachat.route ask "what language was that?"
```

**Measured** — 40.5k training messages, best-held-out checkpoint, both numbers at
the 60% threshold:

| | held-out (unseen skeletons) | probe (hand-written) |
|---|---|---|
| macro accuracy | **95.5%** | **80.6%** |
| chat leakage | 1.6% | **0%** (14/14 stay in chat) |
| per class | chat 98%, codegen 99%, code 99%, vision 98%, reason 89%, game 89% | chat 100%, game 100%, reason 88%, code 67%, vision 67%, codegen 63% |

The 15-point gap between the two columns is the synthesized-class tax, and it is
the honest number: held-out `code` and `vision` score ~98% on phrasings the
generator produced, ~67% on phrasings a person wrote. Training longer makes that
worse, not better — a 4,000-step run peaked on held-out accuracy at step **500**
and then drove probe accuracy from 81% down to 69% while the held-out number sat
still, which is what fitting a template generator looks like from the outside.
The default schedule is short for that reason, and only the best held-out
checkpoint is kept.

**The misses degrade the right way.** Eight of the 48 probe messages are
misrouted; run end to end through the agent, the old triggers recover **three**
of them. *"Work out how many minutes there are in a fortnight"* routes to chat at
53%, under the threshold, and the reasoning trigger picks it up. *"Build me a
class that wraps an http client with retries"* routes to chat and the codegen
trigger catches it. *"Can you code up a debounce helper in js"* is the
interesting one: the router confidently says `code`, but with no file read yet
that handler declines rather than inventing an answer, and codegen takes it —
and gets the language right.

The other five get a chat reply, which is the cheap failure. What did *not*
happen anywhere in the probe set is the expensive one: no ordinary message was
confidently routed away from chat, so nothing answered small talk with generated
code. That asymmetry is what the threshold buys, and it is the reason the router
is worth switching on at 80% rather than waiting for 95%.

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
  data.py         # corpus loaders (SODA, DailyDialog, NPS, Pre-1929 Books) + the
                  #   training text format (tagged turns, plain documents)
  localdata.py    # your own plaintext data in data/: prose -> train.py,
                  #   source -> codegen.py
  blocks.py       # SHARED toolkit: RMSNorm/RoPE/QK-norm/attention/SwiGLU/ReLU2MLP/
                  #   Block, GPTConfig, tokenizers, pick_device, pad_load — every
                  #   model builds on these
  optim.py        # SHARED training toolkit: Muon, the Muon/AdamW parameter split,
                  #   the warmup-stable-decay schedule
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
  route.py        # routing specialist: which capability answers a message,
                  #   a frozen-trunk expert add-on -> models/specialist-route.pt
  narrate.py      # multi-head model (MultiHeadGPT): action head + LM commentary in one pass
  hf_model.py     # fine-tuned GPT-2 backend (opt-in)
  finetune.py     # GPT-2 fine-tuning -> models/gpt2-dailydialog/
  engine.py       # chat generation + MMI relevance reranking
  persona.py      # named personalities: primer turns + sampling + styling
  cli.py          # terminal chat UI (rich)
  export_onnx.py  # export the models to ONNX for the browser -> web/models/
  web.py          # static server for web/ (COOP/COEP, wasm MIME types)
  rooms.py        # the model side of a chat room: one agent per room over one set
                  #   of checkpoints, attachment staging, per-room reset
  api.py          # MASTER API SERVER: holds the models, serves every bot
  client.py       # the other end — remote (HTTP) or in-process, same reply()
  files.py        # which files the agent may open AND write: per-room sandbox for
                  #   bots, unrestricted reads only for the terminal the user owns,
                  #   writes always confined to one workspace + quota'd
  transport.py    # chat plumbing with no model imports: code fencing, message
                  #   splitting, attachments, the env switches — what a bot needs
  actions.py      # what a turn can DO in a room besides talk: the tool vocabulary
                  #   (react/pin/thread/nick), the [[call]] parser, the reaction
                  #   picker — named by the models, performed by the frontend
  discord_bot.py  # Discord chat frontend (discord.py), running the full agent
  google_chat.py  # Google Chat frontend (FastAPI webhook), running the full agent
  agent.py        # unified interface: chat (plain text) + /commands for games;
                  #   plain text is dispatched by the routing specialist
  reader.py       # small model that reads game state to answer questions
  game_train.py   # behaviour-cloning trainer for game control
  play.py         # real-time terminal UI for grid games (rich.Live)
  games/          # pluggable games: core framework + snake/pong/dodge/tictactoe
                  #   + sandbox (a no-train VLA test grid)
                  #   + versus (multiplayer snake: you vs. the bot, reusing the solo model)

personas.json     # YOUR personalities, merged over the built-in ones

workspace/        # what the TERMINAL agent writes with /write (gitignored,
                  #   created on first use). A bot writes in its room's sandbox
                  #   instead, which lives in a temp dir and goes with the room.

data/             # YOUR plaintext training data (optional, see data/README.md)
  text/           #   prose, mixed into the chat model's stream
  code/           #   source, mixed into the code generator's corpus

web/              # the browser frontend — static, no build step
  index.html      #   chat / classify / generate / play panels
  js/tokenizer.js #   byte-level BPE, ported from blocks.BPETokenizer
  js/model.js     #   ONNX session: KV cache, sampling, log-probs, heads
  js/engine.js    #   the chat engine — candidates + MMI ranking (engine.py)
  js/snake.js     #   Snake, ported from games/snake.py, for the action head
  js/app.js       #   page wiring; picks panels from the manifest's `kind`
  models/         #   generated by export_onnx.py (gitignored)
  vendor/ort/     #   onnxruntime-web, copied in by `npm run vendor` (gitignored)

tools/            # build + parity checks for the above
  vendor_ort.mjs        # copy onnxruntime-web out of node_modules
  build_dist.mjs        # package web/ -> dist/, splitting models to fit host caps
  dump_tokenizer_cases.py / check_tokenizer.mjs   # JS tokenizer == Python
  dump_engine_cases.py   / check_web_engine.mjs   # JS runtime  == PyTorch
  check_browser.mjs     # the page in a real Chrome (optional, skips if absent)
```
