#!/usr/bin/env python3
import argparse

from tunelm.inference import dumps, generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validated Strudel with a TuneLM checkpoint")
    parser.add_argument("prompt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    print(
        dumps(
            generate(
                args.prompt,
                args.checkpoint,
                temperature=args.temperature,
                validate=not args.no_validate,
            )
        )
    )


if __name__ == "__main__":
    main()

