"""Expose a single Gymnasium environment through FastMCP."""

import json
from functools import cache
from threading import RLock
from typing import Any, cast

import gymnasium
import numpy as np

from shimmy.mcp_adapters.spaces import (
    decode,
    describe,
    encode,
    image_content,
    image_layout,
    json_value,
)

try:
    from fastmcp import FastMCP
    from mcp.types import TextContent
    from importlib import import_module
    import_module("PIL.Image")
except ImportError as e:
    raise ImportError("MCP support requires: pip install 'shimmy[mcp]'") from e


class GymnasiumMCPAdapter:
    """Wrap an environment with MCP tools, resources, and episode guidance."""

    def __init__(
        self,
        env: gymnasium.Env,
        name: str | None = None,
        *,
        max_image_bytes: int = 5_000_000,
        **kwargs,
    ):
        """Create the server.

        Args:
            env: Environment owned by the caller.
            name: Optional server name override.
            max_image_bytes: Maximum base64 bytes per image.
            **kwargs: Forwarded to FastMCP.
        """
        if max_image_bytes <= 0:
            raise ValueError("max_image_bytes must be positive")
        self.env = env
        self.max_image_bytes = max_image_bytes
        self._lock = RLock()
        self._closed = False
        self.total_reward = 0.0
        env_id = getattr(env.spec, "id", None) or "unknown"
        # TODO: improve this default instruction
        kwargs.setdefault(
            "instructions",
            "Read gymnasium://observation_space and gymnasium://action_space, reset, "
            "then step with a JSON action until terminated or truncated. sample_action "
            "provides valid actions. Image observations need to be check via the "
            "`render` tool'. Close when finished.",
        )
        self.mcp = FastMCP(
            name if name is not None else f"Gymnasium[{env_id}]", **kwargs
        )
        for method in (
            self.reset,
            self.step,
            self.render,
            self.close,
            self.sample_action,
        ):
            self.mcp.tool()(method)
        self._resource("spec", lambda: env.spec.to_json() if env.spec else "{}")
        self._resource(
            "metadata",
            lambda: json.dumps(
                json_value({**env.metadata, "render_mode": env.render_mode}),
                allow_nan=False,
            ),
        )
        self._resource(
            "observation_space",
            lambda: json.dumps(
                describe(env.observation_space, images=True), allow_nan=False
            ),
        )
        self._resource(
            "action_space",
            lambda: json.dumps(describe(env.action_space), allow_nan=False),
        )
        self.mcp.prompt()(self.play_episode)

    def _resource(self, name, factory):
        @cache
        def read() -> str:
            with self._lock:
                return factory()

        self.mcp.resource(
            f"gymnasium://{name}", name=name, mime_type="application/json"
        )(read)

    def __getattr__(self, name):
        """Delegate server methods to FastMCP."""
        return getattr(self.mcp, name)

    def _observation(self, observation, **fields):
        result = {
            "observation": encode(self.env.observation_space, observation, images=True),
            **json_value(fields),
        }
        return [
            TextContent(type="text", text=json.dumps(result, allow_nan=False)),
        ]

    def reset(self, seed: int | None = None, options: dict | None = None) -> Any:
        """Start an episode and return its observation and info."""
        with self._lock:
            observation, info = self.env.reset(seed=seed, options=options)
            self._closed = False
            self.total_reward = 0.0
            return self._observation(
                observation, info=info, total_reward=self.total_reward
            )

    def step(self, action: Any) -> Any:
        """Apply one JSON action and return the transition with render hints for images."""
        with self._lock:
            try:
                action = decode(self.env.action_space, action)
            except ValueError as exc:
                return {"error": True, "message": str(exc)}
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.total_reward += float(reward)
            return self._observation(
                observation,
                reward=reward,
                total_reward=self.total_reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )

    def render(self) -> Any:
        """Return the current render as text, images, or status metadata."""
        with self._lock:
            value = self.env.render()
            if isinstance(value, str):
                return TextContent(type="text", text=value)
            is_list_expected = self.env.render_mode and self.env.render_mode.endswith("_list")
            frames = value if is_list_expected else [value]
            assert frames is not None, f"render_mode of '{self.env.render_mode}' should not produce None"
            if all(isinstance(elem, str) for elem in frames):
                return TextContent(type="text", text=str(frames))
            content = [
                TextContent(
                    type="text", text=json.dumps({"render_mode": self.env.render_mode})
                )
            ]
            for frame in frames:
                layout = (
                    image_layout(gymnasium.spaces.Box(0, 255, frame.shape, np.uint8))
                    if isinstance(frame, np.ndarray) and frame.dtype == np.uint8
                    else None
                )
                if layout:
                    content.append(
                        image_content(cast(np.typing.NDArray[np.uint8], frame), layout, self.max_image_bytes)
                    )
                else:
                    content.append(
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "render_mode": self.env.render_mode,
                                    "data": json_value(frame),
                                },
                                allow_nan=False,
                            ),
                        )
                    )
            return content

    def close(self) -> dict:
        """Close the environment."""
        with self._lock:
            if not self._closed:
                self.env.close()
                self._closed = True
            return {"ok": True}

    def sample_action(self) -> Any:
        """Sample a valid JSON action."""
        with self._lock:
            return encode(self.env.action_space, self.env.action_space.sample())

    def play_episode(self, goal: str | None = None) -> str:
        """Explain how to play one episode."""
        return f"{self.mcp.instructions}\n" + (
            f"Goal: {goal}" if goal else "Play one episode."
        )
