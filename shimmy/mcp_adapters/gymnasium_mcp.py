"""Run a Gymnasium environment over MCP."""

import argparse
import importlib
import json

import gymnasium

from shimmy.mcp_adapters.gymnasium_interface import GymnasiumMCPAdapter


def try_load_json(value: str):
    """Attempt to load value as json."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class GymEnvironmentArgument(argparse.Action):
    """Process keyword arguments that should passthrough to ``gym.make``."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Add arbitrary named arguments into ``namespace.environment_kwargs``."""
        kwargs = getattr(namespace, "environment_kwargs", None)
        if kwargs is None:
            kwargs = namespace.environment_kwargs = {}
        if self.dest in kwargs:
            raise KeyError(f"Duplicate environment argument: {self.dest}")
        kwargs[self.dest] = values


def main(argv: list[str] | None = None) -> None:
    """Start MCP, forwarding extra flags and JSON kwargs to gymnasium.make."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog="Extra --key value or --key=value flags become environment kwargs. "
        "Values use JSON types when valid, otherwise strings; bare flags mean true. "
        "Hyphens become underscores. Duplicate keys raise KeyError.",
    )
    parser.add_argument("-i", "--import", dest="imports", default="")
    parser.add_argument(
        "-t", "--transport", default="stdio", help="MCP transport (default: stdio)"
    )
    parser.add_argument(
        "-r",
        "--render-mode",
        action=GymEnvironmentArgument,
        help="Gymnasium render mode",
    )
    parser.add_argument(
        "--kwargs",
        type=json.loads,
        default={},
        help="JSON object of gymnasium.make arguments",
    )
    parser.add_argument("env_id")
    _, unknown = parser.parse_known_args(argv)
    for flag in dict.fromkeys(token.split("=", 1)[0] for token in unknown):
        if (
            flag.startswith("-")
            and flag != "--"
            and isinstance(try_load_json(flag), str)
        ):
            parser.add_argument(
                flag,
                dest=flag.lstrip("-").replace("-", "_"),
                type=try_load_json,
                nargs="?",
                const=True,
                action=GymEnvironmentArgument,
            )
    args = parser.parse_args(argv)
    if not isinstance(args.kwargs, dict):
        parser.error("--kwargs must be a JSON object")
    kwargs = getattr(args, "environment_kwargs", {})
    duplicates = kwargs.keys() & args.kwargs.keys()
    if duplicates:
        raise KeyError(
            f"Duplicate environment arguments: {', '.join(sorted(duplicates))}"
        )
    kwargs.update(args.kwargs)
    for module in args.imports.split(","):
        if module.strip():
            importlib.import_module(module.strip())
    env = gymnasium.make(args.env_id, **kwargs)
    try:
        GymnasiumMCPAdapter(env).run(transport=args.transport)
    finally:
        env.close()


if __name__ == "__main__":
    main()
