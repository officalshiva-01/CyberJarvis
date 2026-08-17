from pathlib import Path


PROJECT_ROOT = Path.home() / "CyberJarvis"


def list_project_files():
    """Return files and directories inside the CyberJarvis project."""

    if not PROJECT_ROOT.exists():
        return {
            "error": "CyberJarvis project directory does not exist."
        }

    items = []

    for path in sorted(PROJECT_ROOT.iterdir()):
        items.append({
            "name": path.name,
            "type": "directory" if path.is_dir() else "file"
        })

    return items


def read_project_file(filename: str):
    """Read a text file only from the CyberJarvis project directory."""

    requested_path = (PROJECT_ROOT / filename).resolve()

    # Security: prevent escaping CyberJarvis/
    try:
        requested_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return {
            "error": "Access denied: file is outside the CyberJarvis project."
        }

    if not requested_path.exists():
        return {
            "error": f"File not found: {filename}"
        }

    if not requested_path.is_file():
        return {
            "error": f"{filename} is not a regular file."
        }

    try:
        content = requested_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        return {
            "filename": filename,
            "content": content
        }

    except Exception as error:
        return {
            "error": str(error)
        }


if __name__ == "__main__":
    print("Project files:")
    print(list_project_files())

    print("\nTesting main.py:")
    result = read_project_file("main.py")

    if "error" in result:
        print(result["error"])
    else:
        print(result["content"][:1000])
