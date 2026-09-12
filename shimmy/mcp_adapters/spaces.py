"""Single-sample JSON codecs for Gymnasium spaces."""

import base64
import io
import json
from importlib.metadata import version
from typing import Any, cast

import gymnasium
import numpy as np
from gymnasium import spaces
from mcp.types import ImageContent
from PIL import Image

MCP_MAJOR_VERSION = int(version("mcp").split(".")[0])
if MCP_MAJOR_VERSION < 2:
    MIMETYPE_FIELD = "mimeType"
else:
    MIMETYPE_FIELD = "mime_type"


def json_value(value: Any) -> Any:
    """Convert NumPy values to strict JSON values."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    json.dumps(value, allow_nan=False)
    return value


def image_layout(space: spaces.Space) -> str | None:
    """Identify uint8 images, preferring channels-last for ambiguous shapes."""
    if not isinstance(space, spaces.Box) or space.dtype != np.uint8:
        return None
    shape = space.shape
    if not all(shape):
        return None
    if len(shape) == 2:
        return "HW"
    if len(shape) == 3:
        if shape[-1] in (1, 3, 4):
            return "HWC"
        if shape[0] in (1, 3, 4):
            return "CHW"
    return None


def image_content(value: np.ndarray, layout: str, limit: int) -> Any:
    """Encode an image as native MCP content within the base64 byte limit."""
    if layout == "CHW":
        value = np.moveaxis(value, 0, -1)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    buffer = io.BytesIO()
    Image.fromarray(value).save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    if len(data) > limit:
        raise ValueError(f"Encoded image exceeds max_image_bytes={limit}")
    kwargs: dict[str, str] = {MIMETYPE_FIELD: "image/png"}
    return ImageContent(type="image", data=data, **kwargs)


def encode(space: spaces.Space, value: Any, images: bool = False) -> Any:
    """Encode one sample, referring image observations to render."""
    if images and image_layout(space):
        return "Check image with `render` tool"
    if isinstance(space, spaces.Dict):
        return {k: encode(s, value[k], images) for k, s in space.spaces.items()}
    if isinstance(space, spaces.Tuple):
        return [encode(s, v, images) for s, v in zip(space.spaces, value)]
    if isinstance(space, spaces.Sequence):
        values = (
            gymnasium.vector.utils.iterate(space.stacked_feature_space, value)
            if space.stack
            else value
        )
        return [encode(space.feature_space, v, images) for v in values]
    if isinstance(space, spaces.OneOf):
        index, item = value
        return {
            "index": int(index),
            "value": encode(space.spaces[index], item, images),
        }
    if isinstance(space, spaces.Graph):
        assert space.edge_space is not None
        return {
            "nodes": [encode(space.node_space, v, images) for v in value.nodes],
            "edges": (
                None
                if value.edges is None
                else [encode(space.edge_space, v, images) for v in value.edges]
            ),
            "edge_links": json_value(value.edge_links),
        }
    try:
        result = json_value(space.to_jsonable([value])[0])
        if not space.contains(space.from_jsonable([result])[0]):
            raise ValueError("round trip failed")
        return result
    except (TypeError, ValueError, KeyError, IndexError, NotImplementedError) as exc:
        raise ValueError(
            f"Unsupported or invalid {type(space).__name__}: {exc}"
        ) from exc


def decode(space: spaces.Space, value: Any, path: str = "action") -> Any:
    """Decode and validate an action without silently truncating integers."""
    try:
        if isinstance(space, spaces.Dict):
            if not isinstance(value, dict) or value.keys() != space.spaces.keys():
                raise ValueError("expected exactly the declared keys")
            result = {
                k: decode(s, value[k], f"{path}.{k}") for k, s in space.spaces.items()
            }
        elif isinstance(space, spaces.Tuple):
            if not isinstance(value, list) or len(value) != len(space.spaces):
                raise ValueError("expected an array of the declared length")
            result = tuple(
                decode(s, v, f"{path}[{i}]")
                for i, (s, v) in enumerate(zip(space.spaces, value))
            )
        elif isinstance(space, spaces.Sequence):
            if not isinstance(value, list):
                raise ValueError("expected an array")
            items = [
                decode(space.feature_space, v, f"{path}[{i}]")
                for i, v in enumerate(value)
            ]
            result = (
                gymnasium.vector.utils.concatenate(
                    space.feature_space,
                    items,
                    gymnasium.vector.utils.create_empty_array(
                        space.feature_space, n=len(items)
                    ),
                )
                if space.stack
                else tuple(items)
            )
        elif isinstance(space, spaces.OneOf):
            if not isinstance(value, dict) or set(value) != {"index", "value"}:
                raise ValueError("expected index and value")
            index = value["index"]
            if not isinstance(index, int) or not 0 <= index < len(space.spaces):
                raise ValueError("invalid subspace index")
            result = (
                index,
                decode(space.spaces[index], value["value"], f"{path}.value"),
            )
        elif isinstance(space, spaces.Graph):
            if not isinstance(value, dict) or set(value) != {
                "nodes",
                "edges",
                "edge_links",
            }:
                raise ValueError("expected nodes, edges, and edge_links")
            nodes = decode(
                spaces.Sequence(space.node_space, stack=True),
                value["nodes"],
                f"{path}.nodes",
            )
            edges = value["edges"]
            if edges is not None:
                if space.edge_space is None:
                    raise ValueError("edges are not supported")
                edges = decode(
                    spaces.Sequence(space.edge_space, stack=True),
                    edges,
                    f"{path}.edges",
                )
            links = value["edge_links"]
            if links is not None:
                links = np.asarray(links)
                if links.size == 0:
                    links = np.empty((0, 2), dtype=np.int64)
                if links.dtype.kind not in "iu":
                    raise ValueError("edge_links must be integers")
            result = spaces.GraphInstance(nodes, edges, links)
        elif isinstance(
            space,
            (spaces.Box, spaces.MultiDiscrete, spaces.MultiBinary, spaces.Discrete),
        ):
            raw = np.asarray(value)
            if raw.dtype.kind not in "iuf" or not np.all(np.isfinite(raw)):
                raise ValueError("expected finite numeric values")
            if np.issubdtype(space.dtype, np.integer):
                bounds = np.iinfo(cast(np.integer, space.dtype))
                if (
                    np.any(raw != np.floor(raw))
                    or np.any(raw < bounds.min)
                    or np.any(raw > bounds.max)
                ):
                    raise ValueError("expected representable integers")
            result = np.asarray(value, dtype=space.dtype)
            if result.shape != space.shape:
                raise ValueError(f"expected shape {space.shape}, got {result.shape}")
            if isinstance(space, spaces.Discrete):
                result = result[()]
        else:
            result = space.from_jsonable([value])[0]
        if not space.contains(
            result  # pyright: ignore[reportArgumentType,reportGeneralTypeIssues]
        ):
            raise ValueError("value is outside the space")
        return result
    except (
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        OverflowError,
        NotImplementedError,
    ) as exc:
        raise ValueError(
            f"{path}: expected {space!r}; received {str(value)[:160]} ({type(value).__name__}): {exc}"
        ) from exc


def describe(space: spaces.Space, images: bool = False) -> dict:
    """Describe a space using JSON-safe bounds and nested space definitions."""
    result: dict[str, Any] = {"type": type(space).__name__, "repr": repr(space)}
    for field in (
        "shape",
        "dtype",
        "n",
        "nvec",
        "start",
        "min_length",
        "max_length",
        "stack",
    ):
        value = getattr(space, field, None)
        if value is not None:
            result[field] = str(value) if field == "dtype" else json_value(value)
    if isinstance(space, spaces.Box):
        schema = {
            "type": "integer" if np.issubdtype(space.dtype, np.integer) else "number"
        }
        for length in reversed(space.shape):
            schema = {
                "type": "array",
                "minItems": length,
                "maxItems": length,
                "items": schema,
            }
        result["json_schema"] = schema
        for field in ("low", "high"):
            value = getattr(space, field)
            result[field] = np.where(
                np.isfinite(value),
                value.astype(object),
                np.where(value < 0, "-Infinity", "Infinity"),
            ).tolist()
        layout = image_layout(space) if images else None
        result.update(content_type="image" if layout else "json")
        if layout:
            result.update(mime_type="image/png", channel_order=layout)
            result.update(
                observation_value="Check image with `render` tool", image_tool="render"
            )
    if isinstance(space, spaces.Dict):
        result["spaces"] = {k: describe(s, images) for k, s in space.spaces.items()}
    elif isinstance(space, (spaces.Tuple, spaces.OneOf)):
        result["spaces"] = [describe(s, images) for s in space.spaces]
    elif isinstance(space, spaces.Sequence):
        result["feature_space"] = describe(space.feature_space, images)
    elif isinstance(space, spaces.Graph):
        result["node_space"] = describe(space.node_space, images)
        result["edge_space"] = (
            None if space.edge_space is None else describe(space.edge_space, images)
        )
    elif isinstance(space, spaces.Text):
        result["charset"] = sorted(space.character_set)
    return result
