from __future__ import annotations

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        source_dir = self._resolve_core_source()
        force_include = build_data.setdefault("force_include", {})
        assert isinstance(force_include, dict)

        if self.target_name == "sdist":
            force_include[str(source_dir)] = "core/src/query_autocomplete"
            return

        force_include[str(source_dir)] = "query_autocomplete"
        force_include_editable = build_data.setdefault("force_include_editable", {})
        assert isinstance(force_include_editable, dict)
        force_include_editable[str(source_dir)] = "query_autocomplete"

    def _resolve_core_source(self) -> Path:
        candidates = [
            Path(self.root).parent / "core" / "src" / "query_autocomplete",
            Path(self.root) / "core" / "src" / "query_autocomplete",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate

        joined = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Unable to locate query_autocomplete sources for packaging. Checked: {joined}")
