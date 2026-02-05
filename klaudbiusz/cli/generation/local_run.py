"""Single app generation without Dagger (runs locally)."""

import os
from datetime import datetime
from pathlib import Path

import fire
from dotenv import load_dotenv

from cli.generation.codegen import ClaudeAppBuilder
from cli.generation.codegen_multi import LiteLLMAppBuilder

load_dotenv()


def _find_databricks_cli() -> str | None:
    """Auto-detect Databricks CLI binary with aitools/apps-mcp support.

    MCP mode is faster than skills mode because tools are always available.
    Skills require on-demand loading which adds latency per tool invocation.

    Prefers local development builds that have experimental aitools command.
    """
    # Prefer local dev build with aitools support (check these first)
    dev_candidates = [
        Path.home() / "cli" / "cli",  # Local CLI repo build
        Path.home() / "databricks-cli" / "cli",  # Alternative build location
    ]

    for path in dev_candidates:
        if path.exists() and os.access(path, os.X_OK):
            return str(path)

    # Fall back to installed versions (may not have aitools)
    import shutil

    databricks_path = shutil.which("databricks")
    if databricks_path:
        return databricks_path

    return None


# Default MCP args for Databricks CLI aitools command
# "experimental aitools mcp" or "experimental apps-mcp mcp" (apps-mcp is alias)
DEFAULT_MCP_ARGS = ["experimental", "aitools", "mcp"]


def run(
    prompt: str,
    app_name: str | None = None,
    backend: str = "claude",
    model: str | None = None,
    mcp_binary: str | None = None,
    mcp_args: list[str] | None = None,
    output_dir: str | None = None,
    no_mcp: bool = False,
) -> dict[str, str | None]:
    """Run app generation locally (no Dagger).

    Args:
        prompt: The prompt describing what to build
        app_name: Optional app name (default: timestamp-based)
        backend: Backend to use ("claude" or "litellm", default: "claude")
        model: LLM model (required if backend=litellm)
        mcp_binary: Path to Databricks CLI (auto-detected if not specified)
        mcp_args: Args for MCP server (default: experimental aitools mcp)
        output_dir: Directory to store generated apps (default: ./app)
        no_mcp: Force skills mode even if MCP binary is available

    Usage:
        # Claude backend - auto-detects Databricks CLI for faster MCP execution
        uv run python -m cli.generation.local_run "build dashboard"

        # Force skills mode (slower, for testing)
        uv run python -m cli.generation.local_run "build dashboard" --no_mcp

        # Explicit CLI path
        uv run python -m cli.generation.local_run "build dashboard" --mcp_binary=~/cli/cli

        # LiteLLM backend - requires MCP
        uv run python -m cli.generation.local_run "build dashboard" --backend=litellm --model=gemini/gemini-2.5-pro
    """
    # Auto-detect Databricks CLI if not specified (MCP is faster than skills)
    if mcp_binary is None and not no_mcp:
        mcp_binary = _find_databricks_cli()
        if mcp_binary:
            # Use default MCP args if not specified
            if mcp_args is None:
                mcp_args = DEFAULT_MCP_ARGS
            print(f"🚀 Using MCP mode (faster): {mcp_binary} {' '.join(mcp_args)}")
        else:
            print("⚠️  Databricks CLI not found, falling back to skills mode (slower)")
    elif no_mcp:
        mcp_binary = None
        print("📜 Using skills mode (--no_mcp flag)")
    else:
        if mcp_args is None:
            mcp_args = DEFAULT_MCP_ARGS
        print(f"🚀 Using MCP mode: {mcp_binary} {' '.join(mcp_args)}")

    if backend == "litellm":
        if not model:
            raise ValueError("--model is required when using --backend=litellm")
        if not mcp_binary:
            raise ValueError("--mcp_binary is required for litellm backend (use --mcp_binary or ensure Databricks CLI is available)")

    if app_name is None:
        app_name = f"app-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    resolved_output_dir = Path(output_dir) if output_dir else Path("./app")

    match backend:
        case "claude":
            builder = ClaudeAppBuilder(
                app_name=app_name,
                output_dir=str(resolved_output_dir),
                mcp_binary=mcp_binary,
                mcp_args=mcp_args,
            )
            metrics = builder.run(prompt)
            app_dir = metrics.get("app_dir")
        case "litellm":
            assert model is not None  # already validated above
            builder = LiteLLMAppBuilder(
                app_name=app_name,
                model=model,
                mcp_binary=mcp_binary,
                mcp_args=mcp_args,
                output_dir=str(resolved_output_dir),
            )
            result = builder.run(prompt)
            app_dir = result.app_dir
        case _:
            raise ValueError(f"Unknown backend: {backend}. Use 'claude' or 'litellm'.")

    print(f"\n{'=' * 80}")
    if app_dir:
        print("Generation complete:")
        print(f"  App: {app_dir}")
    else:
        print("No app generated (agent may have just answered without creating files)")
    print(f"{'=' * 80}\n")

    return {"app_dir": app_dir}



if __name__ == "__main__":
    fire.Fire(run)
