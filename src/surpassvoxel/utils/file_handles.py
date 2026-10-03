import yaml
import orjson
import logging
import threading
from pathlib import Path
from typing import Any


def _normal_path(path: str | Path) -> Path:
    return Path(path).resolve()


def load_json(path: str | Path) -> Any:
    path = _normal_path(path)
    return orjson.loads(path.read_bytes())


def save_json(data: Any, path: str | Path, *, indent: bool = True) -> None:
    path = _normal_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    option = orjson.OPT_INDENT_2 if indent else 0
    path.write_bytes(orjson.dumps(data, option=option))


def loads_json(data: str) -> Any:
    return orjson.loads(data)


def saves_json(data: Any, *, indent: bool = True) -> str:
    option = orjson.OPT_INDENT_2 if indent else 0
    return orjson.dumps(data, option=option).decode("utf-8")


def load_yaml(path: str | Path) -> dict:
    path = _normal_path(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_yaml(data: Any, path: str | Path) -> None:
    path = _normal_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


# ================================


_logging_init: threading.Lock = threading.Lock()
_logging_files: dict[str, logging.Logger] = dict()


def get_log(path: str | Path | None, *, init_level: int = logging.INFO) -> logging.Logger:
    NULL_LOGGER_NAME = "surpassvoxel.null"
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
    LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

    global _logging_files
    if path is None:
        logger = logging.getLogger(NULL_LOGGER_NAME)
        logger.propagate = False
        logger.setLevel(init_level)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        return logger

    path = _normal_path(path)
    key = str(path)

    with _logging_init:
        logger = _logging_files.get(key)
        if logger is not None:
            return logger

        path.parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(f"surpassvoxel.{key}")
        logger.propagate = False
        logger.setLevel(init_level)
        logger.handlers.clear()

        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        logger.addHandler(handler)

        _logging_files[key] = logger

    return logger


_keys_init: threading.Lock = threading.Lock()
_keys_files: dict[str, tuple[Path, Any]] = dict()


def get_config(name: str, *, path: str | Path | None = None) -> Any:

    global _keys_files
    if path is None:
        with _keys_init:
            cached = _keys_files.get(name)
            if cached is None:
                raise ValueError(f"keys {name!r} is not registered")
            return cached[1]

    path = _normal_path(path)

    with _keys_init:
        cached = _keys_files.get(name)
        if cached is not None:
            loaded_path, data = cached
            if loaded_path != path:
                raise ValueError(
                    f"keys {name!r} is already loaded from {loaded_path}, "
                    f"it cannot be loaded from {path}"
                )
            return data

        try:
            data = load_yaml(path)
        except Exception as exc:
            raise ValueError(f"failed to load keys {name!r} from {path}: {exc}") from exc

        _keys_files[name] = (path, data)

    return data
