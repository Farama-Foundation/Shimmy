"""Exercise the MCP adapter through its codecs and in-process client."""

import asyncio
import base64
import io
import json
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from shimmy.mcp_adapters import GymnasiumMCPAdapter
from shimmy.mcp_adapters import gymnasium_mcp as cli
from shimmy.mcp_adapters.spaces import (
    MIMETYPE_FIELD,
    decode,
    describe,
    encode,
    image_content,
    image_layout,
)

fastmcp = pytest.importorskip("fastmcp")
Image = pytest.importorskip("PIL.Image")

SPACE_CASES = [
    gym.spaces.Discrete(4, start=2),
    gym.spaces.Box(-1, 1, (), dtype=np.float32),
    gym.spaces.Box(-1, 1, (2, 3), dtype=np.float32),
    gym.spaces.MultiBinary((2, 3)),
    gym.spaces.MultiDiscrete([2, 3]),
    gym.spaces.Text(10),
    gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Text(5))),
    gym.spaces.Dict(a=gym.spaces.Discrete(2), b=gym.spaces.MultiBinary(3)),
    gym.spaces.Sequence(gym.spaces.Dict(a=gym.spaces.Discrete(2))),
    gym.spaces.Sequence(gym.spaces.Dict(a=gym.spaces.Discrete(2)), stack=True),
    gym.spaces.Graph(gym.spaces.Box(-1, 1, (2,)), gym.spaces.Discrete(3)),
    gym.spaces.Graph(gym.spaces.Discrete(3), None),
]
if hasattr(gym.spaces, "OneOf"):
    SPACE_CASES.append(gym.spaces.OneOf((gym.spaces.Discrete(2), gym.spaces.Text(5))))


@pytest.mark.parametrize("space", SPACE_CASES)
def test_space_roundtrip(space):
    """All supported spaces preserve single samples over JSON."""
    space.seed(17)
    value = space.sample()
    encoded = json.loads(json.dumps(encode(space, value), allow_nan=False))
    assert space.contains(decode(space, encoded))
    assert encode(space, decode(space, encoded)) == encoded
    json.dumps(describe(space), allow_nan=False)


@pytest.mark.parametrize("value", [1.5, True, "1", -1, 256, [1]])
def test_invalid_action(value):
    """Reject coercions and out-of-space values with a useful path."""
    space = gym.spaces.Dict(a=gym.spaces.Discrete(2))
    with pytest.raises(ValueError, match=r"action.a"):
        decode(space, {"a": value})


class Pixels(gym.Env):
    """Small deterministic environment for image and state tests."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array", "rgb_array_list", "ansi"]}
    render_mode = "rgb_array"

    def __init__(self):
        """Initialize image spaces and counters."""
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            image=gym.spaces.Box(0, 255, (8, 9, 3), np.uint8),
            count=gym.spaces.Discrete(100),
        )
        self.steps = 0
        self.closes = 0

    def reset(self, *, seed=None, options=None):
        """Return the current frame."""
        super().reset(seed=seed)
        return {"image": self.render(), "count": self.steps}, {}

    def step(self, action):
        """Advance the counter."""
        self.steps += 1
        obs, info = self.reset()
        return obs, 1.0, False, False, info

    def render(self):
        """Return RGB pixels."""
        return np.full((8, 9, 3), self.steps, dtype=np.uint8)

    def close(self):
        """Record cleanup."""
        self.closes += 1


def test_client():
    """Exercise registration, transitions, static resources, and native images."""

    async def run():
        env = Pixels()
        adapter = GymnasiumMCPAdapter(env)
        assert adapter.name == "Gymnasium[unknown]"
        async with fastmcp.Client(adapter.mcp) as client:

            async def read(uri):
                result = await client.read_resource(uri)
                return getattr(result, "contents", result)[0].text

            tools = await client.list_tools()
            assert {t.name for t in getattr(tools, "tools", tools)} == {
                "reset",
                "step",
                "render",
                "close",
                "sample_action",
            }
            prompts = await client.list_prompts()
            assert getattr(prompts, "prompts", prompts)[0].name == "play_episode"
            result = await client.call_tool("reset", {})
            payload = json.loads(result.content[0].text)
            assert payload["total_reward"] == 0
            assert payload["observation"]["image"] == "Check image with `render` tool"
            assert payload["observation"]["count"] == 0
            assert len(result.content) == 1
            rendered = await client.call_tool("render", {})
            assert rendered.content[0].type == "text"
            block = rendered.content[1]
            assert (
                block.type == "image" and getattr(block, MIMETYPE_FIELD) == "image/png"
            )
            pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(block.data))))
            np.testing.assert_array_equal(pixels, env.render())
            await client.call_tool("step", {"action": 0})
            stepped = await client.call_tool("step", {"action": 0})
            assert env.steps == 2
            assert json.loads(stepped.content[0].text)["total_reward"] == 2
            assert (
                json.loads(stepped.content[0].text)["observation"]["image"]
                == "Check image with `render` tool"
            )
            assert len(stepped.content) == 1
            invalid = await client.call_tool("step", {"action": 0.5})
            assert json.loads(invalid.content[0].text)["error"]
            assert env.steps == 2
            reset = await client.call_tool("reset", {})
            assert json.loads(reset.content[0].text)["total_reward"] == 0
            assert (await client.call_tool("render", {})).content[1].type == "image"
            first = await read("gymnasium://metadata")
            env.metadata = {"changed": True}
            assert await read("gymnasium://metadata") == first
            resources = await client.list_resources()
            for resource in getattr(resources, "resources", resources):
                json.loads(await read(resource.uri))
            await client.call_tool("close", {})
        assert env.closes == 1

    asyncio.run(run())


def test_cartpole():
    """Run a standard environment without a renderer dependency."""
    adapter = GymnasiumMCPAdapter(gym.make("CartPole-v1"))
    try:
        assert adapter.name == "Gymnasium[CartPole-v1]"
        adapter.reset(seed=0)
        action = adapter.sample_action()
        assert adapter.env.action_space.contains(action)
        assert "reward" in json.loads(adapter.step(action)[0].text)
    finally:
        adapter.close()


@pytest.mark.parametrize("shape", [(8, 9), (8, 9, 1), (8, 9, 3), (8, 9, 4), (3, 8, 9)])
def test_image_layouts(shape):
    """Encode grayscale, RGB, RGBA, and channel-first images losslessly."""
    space = gym.spaces.Box(0, 255, shape, np.uint8)
    value = space.sample()
    assert encode(space, value, images=True) == "Check image with `render` tool"
    img_layout = image_layout(space)
    assert img_layout is not None
    block = image_content(value, img_layout, 5_000_000)
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(block.data))))
    expected = np.moveaxis(value, 0, -1) if shape == (3, 8, 9) else value.squeeze()
    np.testing.assert_array_equal(decoded, expected)
    with pytest.raises(ValueError, match="max_image_bytes"):
        image_content(value, img_layout, 1)
    assert encode(space, value) == value.tolist()


def test_non_image_and_bounds():
    """Keep tensors as JSON and represent infinite bounds explicitly."""
    space = gym.spaces.Box(-np.inf, np.inf, (2, 2), np.float32)
    assert encode(space, np.zeros((2, 2), np.float32), images=True) == [[0, 0], [0, 0]]
    assert describe(space)["low"] == [["-Infinity"] * 2] * 2


def test_custom_space():
    """Use a custom space's codec and explain unsupported output."""

    class Custom(gym.Space):
        def contains(self, value):
            return value == "valid"

    space = Custom()
    assert encode(space, "valid") == "valid"
    assert decode(space, "valid") == "valid"
    with pytest.raises(ValueError, match="Custom"):
        encode(space, object())


@pytest.mark.parametrize("mode", ["rgb_array_list", "ansi", "human", None])
def test_render_modes(mode, monkeypatch):
    """Preserve frame order, text, and empty render status."""
    env = Pixels()
    env.render_mode = mode
    frames = [env.render(), env.render() + 1]
    output = frames if mode == "rgb_array_list" else "scene" if mode == "ansi" else None
    monkeypatch.setattr(env, "render", lambda: output)
    adapter = GymnasiumMCPAdapter(
        env, name="Custom", instructions="Custom instructions"
    )
    assert adapter.name == "Custom"
    assert "Custom instructions" in adapter.play_episode("win")
    result = adapter.render()
    if mode == "rgb_array_list":
        for block, frame in zip(result[1:], frames):
            np.testing.assert_array_equal(
                np.asarray(Image.open(io.BytesIO(base64.b64decode(block.data)))), frame
            )
        assert len(result) == 3
    elif mode == "ansi":
        assert result.text == "scene"
    else:
        assert json.loads(result[1].text) == {"render_mode": mode, "data": None}


def test_cli(monkeypatch):
    """Import registration modules before make and close on server failure."""
    calls = []
    env = Pixels()
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: calls.append(name))
    monkeypatch.setattr(
        cli.gymnasium, "make", lambda name: (calls.append(name), env)[1]
    )

    class Server:
        def __init__(self, value):
            assert value is env

        def run(self, transport):
            assert transport == "stdio"
            raise RuntimeError("stopped")

    monkeypatch.setattr(cli, "GymnasiumMCPAdapter", Server)
    with pytest.raises(RuntimeError, match="stopped"):
        cli.main(["-i", " first, ,second ", "Test-v0"])
    assert calls == ["first", "second", "Test-v0"]
    assert env.closes == 1


@pytest.mark.parametrize("transport_flag", ["-t", "--transport"])
@pytest.mark.parametrize("render_flag", ["-r", "--render-mode"])
def test_cli_arguments(monkeypatch, transport_flag, render_flag):
    """Forward typed environment arguments and select the MCP transport."""
    calls = {}
    env = Pixels()

    def make(name, **kwargs):
        calls.update(name=name, kwargs=kwargs)
        return env

    class Server:
        def __init__(self, value):
            assert value is env

        def run(self, **kwargs):
            calls["run"] = kwargs

    monkeypatch.setattr(cli.gymnasium, "make", make)
    monkeypatch.setattr(cli, "GymnasiumMCPAdapter", Server)
    cli.main(
        [
            "--gravity",
            "-9.8",
            "Test-v0",
            transport_flag,
            "http",
            render_flag,
            "rgb_array",
            "--max-episode-steps=12",
            "--enabled",
            "--disable-env-checker",
            "true",
            "--label",
            "hello",
            "--weights",
            "[1, 2]",
            "--optional",
            "null",
            "--kwargs",
            '{"config": {"size": 3}}',
        ]
    )
    assert calls == {
        "name": "Test-v0",
        "kwargs": {
            "gravity": -9.8,
            "render_mode": "rgb_array",
            "max_episode_steps": 12,
            "enabled": True,
            "disable_env_checker": True,
            "label": "hello",
            "weights": [1, 2],
            "optional": None,
            "config": {"size": 3},
        },
        "run": {"transport": "http"},
    }
    assert env.closes == 1


@pytest.mark.parametrize(
    "arguments",
    [
        ["--size", "3", "--kwargs", '{"size": 4}'],
        ["--max-episode-steps=3", "--kwargs", '{"max_episode_steps": 4}'],
        ["-r", "rgb_array", "--kwargs", '{"render_mode": "human"}'],
        ["--size=3", "--size=4"],
    ],
)
def test_cli_duplicate_kwargs(arguments):
    """Reject collisions before constructing an environment."""
    with pytest.raises(KeyError, match="Duplicate environment argument"):
        cli.main(["Test-v0", *arguments])


@pytest.mark.parametrize("value", ["[]", "null", "1", "invalid"])
def test_cli_invalid_kwargs(value):
    """Require a JSON object for --kwargs."""
    with pytest.raises(SystemExit):
        cli.main(["Test-v0", "--kwargs", value])
