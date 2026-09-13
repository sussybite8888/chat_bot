"""Terminal chat UI."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from .engine import BACKENDS, REPLY_LENGTHS, ChatEngine
from .persona import BUILT_INS as PERSONAS
from .persona import describe as describe_personas
from .persona import resolve_persona


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sodachat",
        description="Chat with a small GPT trained from scratch on dialogue data.",
    )
    parser.add_argument("--once", metavar="MESSAGE", help="reply to one message and exit")
    parser.add_argument(
        "--unfiltered",
        action="store_true",
        help="disable the profanity filter on replies (on by default, since a "
        "generative model can produce anything)",
    )
    parser.add_argument("--plain", action="store_true", help="hide reply metadata")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=None,
        help="mini: from-scratch GPT trained on DailyDialog (default); "
        "gpt2: fine-tuned GPT-2, opt-in (needs a beefier machine to train)",
    )
    parser.add_argument(
        "--reply-length",
        choices=sorted(REPLY_LENGTHS),
        default=None,
        help="how long replies may run (default: medium, or "
        "SODACHAT_REPLY_LENGTH). Sets the generation budget and the "
        "sentence/character trim together.",
    )
    parser.add_argument(
        "--persona",
        metavar="NAME",
        default=None,
        # Not `choices=`: a persona can also come from personas.json, which
        # is read when the name is resolved rather than when --help is built.
        help="personality of the replies: "
        + " | ".join(PERSONAS)
        + " (default: neutral, or SODACHAT_PERSONA). Add your own in "
        "personas.json; switch mid-chat with /persona.",
    )
    parser.add_argument(
        "--personas",
        action="store_true",
        help="list the available personas (including your own) and exit",
    )
    args = parser.parse_args(argv)

    try:
        persona = resolve_persona(args.persona)
    except ValueError as e:
        parser.error(str(e))

    console = Console()
    if args.personas:  # before the model load: this question doesn't need it
        # markup=False: a custom persona's line carries the file it came from,
        # and rich reads "[/path/to/personas.json]" as a closing tag.
        console.print(describe_personas(persona), markup=False)
        return

    console.print(
        "[dim]loading model (a missing model is trained on first run — that "
        "one-time step can take a while)...[/]"
    )
    engine = ChatEngine(
        filtered=not args.unfiltered,
        seed=args.seed,
        backend=args.backend,
        reply_length=args.reply_length,
        persona=persona,
    )
    history: list[str] = []

    def respond(message: str) -> None:
        reply = engine.reply(message, history=history)
        history.extend([message, reply.text])
        line = f"[bold magenta]bot ›[/] {reply.text}"
        if not args.plain:
            meta = reply.source
            if engine.persona.name != "neutral":
                meta += f" · {engine.persona.name}"
            if reply.score:
                meta += f" · rel {reply.score:.2f}"
            line += f"  [dim]({meta})[/]"
        console.print(line)

    if args.once is not None:
        respond(args.once)
        return

    console.print(
        Panel.fit(
            f"Chatting via the [bold]{engine.backend}[/] backend, "
            f"persona [bold]{engine.persona.name}[/].\n"
            "Type [bold]/persona[/] to change how it sounds, "
            "[bold]/quit[/] (or Ctrl-D) to leave.",
            border_style="cyan",
            title="sodachat",
        )
    )
    while True:
        try:
            message = console.input("[bold cyan]you ›[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if message.strip().lower() in {"/quit", "/exit", "/q"}:
            break
        # The only command plain chat has: swap the personality without
        # restarting (the agent REPL has the full /persona, this is the same
        # switch). Bare /persona lists what's on offer.
        if message.strip().split(" ")[0].lower() in {"/persona", "/personality"}:
            _, _, name = message.strip().partition(" ")
            if name.strip():
                try:
                    engine.persona = resolve_persona(name.strip())
                except ValueError as e:
                    console.print(f"[red]{e}[/]")
                    continue
            console.print(describe_personas(engine.persona), markup=False,
                          style="dim")
            continue
        respond(message)
    console.print("[dim]bye![/]")


if __name__ == "__main__":
    main()
