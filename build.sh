#!/usr/bin/bash

uvx cibuildwheel --platform linux --output-dir=dist
uv run python -m build --sdist