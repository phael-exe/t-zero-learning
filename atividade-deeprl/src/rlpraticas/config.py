"""Carregamento de configurações YAML com herança (`base:`) e overrides via CLI."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def deep_merge(parent: dict, child: dict) -> dict:
    """Mescla `child` sobre `parent` recursivamente, sem alterar os originais."""
    merged = copy.deepcopy(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def set_by_path(cfg: dict, dotted_key: str, value: Any) -> None:
    """Define cfg["a"]["b"] = value a partir de "a.b"."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise KeyError(f"'{part}' não é uma seção em '{dotted_key}'")
    node[parts[-1]] = value


def _load_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(_load_file(path.parent / base), cfg)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    """Lê o YAML, resolve `base:` e aplica overrides no formato chave.sub=valor."""
    cfg = _load_file(Path(path))
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override inválido: {item!r}. Use chave.subchave=valor")
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    for required in ("experiment", "algo", "env_id", "total_timesteps"):
        if required not in cfg:
            raise KeyError(f"Configuração sem a chave obrigatória '{required}'")
    return cfg
