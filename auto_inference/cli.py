"""
Command-line interface for Auto-Inference.

Usage examples::

    # Single prompt
    auto-inference generate meta-llama/Meta-Llama-3-8B-Instruct "Hello!"

    # Interactive chat session
    auto-inference chat meta-llama/Meta-Llama-3-8B-Instruct

    # System info
    auto-inference info

    # Search HuggingFace Hub
    auto-inference search "mistral 7b"
"""

import argparse
import sys
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-inference",
        description="Auto-Inference: unified LLM inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ---- generate ----
    gen = subparsers.add_parser("generate", help="Generate text from a single prompt")
    gen.add_argument("model", help="HuggingFace model ID or local path")
    gen.add_argument("prompt", help="Input prompt (wrap in quotes)")
    gen.add_argument("--backend", default="transformers",
                     choices=["transformers", "unsloth", "vllm", "llamacpp"],
                     help="Inference backend (default: transformers)")
    gen.add_argument("--max-new-tokens", type=int, default=512)
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--top-k", type=int, default=50)
    gen.add_argument("--repetition-penalty", type=float, default=1.0)
    gen.add_argument("--no-sample", action="store_true",
                     help="Use greedy decoding instead of sampling")
    gen.add_argument("--stream", action="store_true",
                     help="Stream output token by token")
    gen.add_argument("--load-in-4bit", action="store_true",
                     help="Load model in 4-bit (transformers backend only)")
    gen.add_argument("--load-in-8bit", action="store_true",
                     help="Load model in 8-bit (transformers backend only)")
    gen.add_argument("--device", default="auto",
                     help="Device: auto, cuda, cpu, mps (transformers only)")
    gen.add_argument("--trust-remote-code", action="store_true")

    # ---- chat ----
    chat = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat.add_argument("model", help="HuggingFace model ID or local path")
    chat.add_argument("--backend", default="transformers",
                      choices=["transformers", "unsloth", "vllm", "llamacpp"])
    chat.add_argument("--system", default=None,
                      help="System prompt for the conversation")
    chat.add_argument("--max-new-tokens", type=int, default=512)
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.add_argument("--stream", action="store_true",
                      help="Stream assistant responses")
    chat.add_argument("--load-in-4bit", action="store_true")
    chat.add_argument("--load-in-8bit", action="store_true")
    chat.add_argument("--device", default="auto")
    chat.add_argument("--trust-remote-code", action="store_true")

    # ---- info ----
    subparsers.add_parser("info", help="Print system and device information")

    # ---- search ----
    search = subparsers.add_parser("search", help="Search HuggingFace Hub for models")
    search.add_argument("query", help="Search query (e.g. 'llama 3 instruct')")
    search.add_argument("--task", default="text-generation",
                        help="HuggingFace task filter (default: text-generation)")
    search.add_argument("--limit", type=int, default=10)

    # ---- estimate ----
    est = subparsers.add_parser("estimate", help="Estimate VRAM for a model")
    est.add_argument("model", help="HuggingFace model ID")
    est.add_argument("--dtype", default="float16",
                     choices=["float32", "float16", "bfloat16", "int8", "int4", "nf4"],
                     help="Target dtype for loading (default: float16)")

    return parser


def _load_backend(args: argparse.Namespace):
    """Build and return the configured backend."""
    from auto_inference.core import AutoInference

    kwargs: dict = {}

    if args.backend == "transformers":
        kwargs["device"] = args.device
        kwargs["trust_remote_code"] = getattr(args, "trust_remote_code", False)
        kwargs["load_in_4bit"] = getattr(args, "load_in_4bit", False)
        kwargs["load_in_8bit"] = getattr(args, "load_in_8bit", False)

    print(f"Loading '{args.model}' with backend='{args.backend}' …", file=sys.stderr)

    return AutoInference.from_pretrained(
        model_name=args.model,
        backend=args.backend,
        max_new_tokens=getattr(args, "max_new_tokens", 512),
        temperature=getattr(args, "temperature", 0.7),
        do_sample=not getattr(args, "no_sample", False),
        top_p=getattr(args, "top_p", 0.95),
        top_k=getattr(args, "top_k", 50),
        repetition_penalty=getattr(args, "repetition_penalty", 1.0),
        **kwargs,
    )


def cmd_generate(args: argparse.Namespace) -> None:
    backend = _load_backend(args)

    if args.stream and hasattr(backend, "generate_stream"):
        for chunk in backend.generate_stream(args.prompt):
            print(chunk, end="", flush=True)
        print()
    else:
        print(backend(args.prompt))


def cmd_chat(args: argparse.Namespace) -> None:
    backend = _load_backend(args)
    history: List[dict] = []
    system_prompt: Optional[str] = args.system

    print("Interactive chat session. Type 'exit' or 'quit' to end.")
    print("Type 'clear' to reset conversation history.")
    if system_prompt:
        print(f"System: {system_prompt}")
    print("-" * 40)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            history.clear()
            print("[Conversation cleared]")
            continue

        history.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)

        if args.stream and hasattr(backend, "chat_stream"):
            full_response = ""
            for chunk in backend.chat_stream(history, system_prompt=system_prompt):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()
        else:
            full_response = backend.chat(history, system_prompt=system_prompt)
            print(full_response)

        history.append({"role": "assistant", "content": full_response})


def cmd_info(_args: argparse.Namespace) -> None:
    from auto_inference.utils import print_system_info
    from auto_inference.core import AutoInference

    print_system_info()
    print()
    print("Available backends:", ", ".join(AutoInference.available_backends()))


def cmd_search(args: argparse.Namespace) -> None:
    from auto_inference.utils import list_hf_models

    print(f"Searching HuggingFace Hub for: '{args.query}' (task={args.task})\n")
    results = list_hf_models(args.query, task=args.task, limit=args.limit)

    if not results:
        print("No results found.")
        return

    col_w = max(len(r["model_id"]) for r in results) + 2
    print(f"{'Model ID':<{col_w}} {'Downloads':>12}  {'Likes':>8}")
    print("-" * (col_w + 24))
    for r in results:
        print(f"{r['model_id']:<{col_w}} {r['downloads']:>12,}  {r['likes']:>8,}")


def cmd_estimate(args: argparse.Namespace) -> None:
    from auto_inference.utils import estimate_model_memory

    gib = estimate_model_memory(args.model, dtype=args.dtype)
    if gib is None:
        print(f"Could not retrieve parameter count for '{args.model}'.")
        print("Check that the model ID is correct and the Hub is reachable.")
    else:
        print(f"Estimated VRAM for '{args.model}' at {args.dtype}: ~{gib:.2f} GiB")


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "generate": cmd_generate,
        "chat": cmd_chat,
        "info": cmd_info,
        "search": cmd_search,
        "estimate": cmd_estimate,
    }

    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
