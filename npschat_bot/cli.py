"""Terminal chat UI."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from .engine import BACKENDS, ChatEngine


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="npschat", description="Chat with a GPT trained on the NPS Chat corpus."
    )
    parser.add_argument("--once", metavar="MESSAGE", help="reply to one message and exit")
    parser.add_argument(
        "--unfiltered",
        action="store_true",
        help="disable the profanity filter (the corpus is raw 2006 chat-room text)",
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
    args = parser.parse_args(argv)

    console = Console()
    console.print(
        "[dim]loading model (a missing model is trained on first run — that "
        "one-time step can take a while)...[/]"
    )
    engine = ChatEngine(
        filtered=not args.unfiltered, seed=args.seed, backend=args.backend
    )
    history: list[str] = []

    def respond(message: str) -> None:
        reply = engine.reply(message, history=history)
        history.extend([message, reply.text])
        line = f"[bold magenta]bot ›[/] {reply.text}"
        if not args.plain:
            meta = reply.source
            if reply.score:
                meta += f" · rel {reply.score:.2f}"
            line += f"  [dim]({meta})[/]"
        console.print(line)

    if args.once is not None:
        respond(args.once)
        return

    console.print(
        Panel.fit(
            f"Chatting via the [bold]{engine.backend}[/] backend.\n"
            "Type [bold]/quit[/] (or Ctrl-D) to leave.",
            border_style="cyan",
            title="npschat",
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
        respond(message)
    console.print("[dim]bye![/]")


if __name__ == "__main__":
    main()
