from __future__ import annotations

from importlib import metadata
import hashlib
from pathlib import Path
import shutil
import zipfile


VERSION = "1.0.0"
QT_DISTRIBUTIONS = (
    "PySide6",
    "PySide6-Essentials",
    "PySide6-Addons",
    "shiboken6",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_qt_licenses(output: Path) -> int:
    copied = 0
    for distribution_name in QT_DISTRIBUTIONS:
        distribution = metadata.distribution(distribution_name)
        package_name = distribution.metadata["Name"] or distribution_name
        destination_root = output / f"{package_name}-{distribution.version}"
        for relative in distribution.files or ():
            parts = Path(str(relative)).parts
            lowered = [part.lower() for part in parts]
            if "licenses" not in lowered:
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file():
                continue
            index = lowered.index("licenses")
            destination = destination_root.joinpath(*parts[index + 1 :])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1
    if copied == 0:
        raise RuntimeError("未找到 PySide6 / Qt 许可证文件")
    return copied


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    dist = project / "dist"
    release = dist / f"MandibleRegistration-v{VERSION}"
    executable = release / f"MandibleRegistration-v{VERSION}.exe"
    executable.resolve(strict=True)

    shutil.copy2(project / "PORTABLE_README.txt", release / "使用说明.txt")
    shutil.copy2(
        project / "THIRD_PARTY_NOTICES.md",
        release / "THIRD_PARTY_NOTICES.md",
    )
    copied = collect_qt_licenses(release / "PySide6-LICENSES")

    archive = dist / f"MandibleRegistration-v{VERSION}-win64.zip"
    if archive.is_file():
        archive.unlink()
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as output:
        for source in sorted(path for path in release.rglob("*") if path.is_file()):
            output.write(source, source.relative_to(release.parent).as_posix())

    sidecar = archive.with_name(f"{archive.name}.sha256.txt")
    sidecar.write_text(
        f"{sha256_file(archive)}  {archive.name}\n",
        encoding="ascii",
    )
    print(f"Packaged {archive.name} with {copied} Qt license files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
