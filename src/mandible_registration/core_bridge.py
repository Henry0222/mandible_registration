from __future__ import annotations

from pathlib import Path


def ensure_registration_core() -> Path:
    """Verify the installed, versioned public registration integration API."""

    try:
        import auto_alignment.integration as integration
    except ModuleNotFoundError as exc:
        if exc.name not in {"auto_alignment", "auto_alignment.integration"}:
            raise
        raise RuntimeError("请在当前解释器中安装已交付的通用配准 2.0.0 wheel。") from exc
    if (
        integration.INTEGRATION_API_VERSION != 1
        or integration.GENERAL_MODEL_REGISTRATION_VERSION != "2.0.0"
    ):
        raise RuntimeError("通用配准版本与本次已验证接口不一致，请核对安装包。")
    return Path(integration.__file__).resolve().parent
