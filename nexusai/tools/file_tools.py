"""File Tools — File system operations (read, write, search, list, etc.)."""

from __future__ import annotations
import os
import glob
import shutil
from pathlib import Path
from nexusai.tools.base import BaseTool, tool


class FileTool(BaseTool):
    name = "file"
    description = "File system operations — read, write, search, copy, move, delete files and directories"
    category = "filesystem"

    @tool(description="Read the contents of a file")
    def read_file(self, path: str, encoding: str = "utf-8") -> str:
        with open(path, "r", encoding=encoding) as f:
            return f.read()

    @tool(description="Write content to a file (creates parent dirs if needed)")
    def write_file(self, path: str, content: str, encoding: str = "utf-8") -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)
        return f"Written {len(content)} bytes to {path}"

    @tool(description="Append content to a file")
    def append_file(self, path: str, content: str) -> str:
        with open(path, "a") as f:
            f.write(content)
        return f"Appended {len(content)} bytes to {path}"

    @tool(description="List files and directories at a path")
    def list_dir(self, path: str = ".", recursive: bool = False) -> list:
        p = Path(path)
        if recursive:
            return [str(f) for f in p.rglob("*")]
        return [str(f) for f in p.iterdir()]

    @tool(description="Search for files matching a glob pattern")
    def glob_search(self, pattern: str, root: str = ".") -> list:
        return glob.glob(os.path.join(root, pattern), recursive=True)

    @tool(description="Search for text within files")
    def search_in_files(self, query: str, path: str = ".", extension: str = "*") -> list:
        results = []
        p = Path(path)
        for f in p.rglob(f"*.{extension}" if extension != "*" else "*"):
            if f.is_file():
                try:
                    content = f.read_text(errors="ignore")
                    for i, line in enumerate(content.splitlines(), 1):
                        if query.lower() in line.lower():
                            results.append({"file": str(f), "line": i, "content": line.strip()})
                except Exception:
                    pass
        return results

    @tool(description="Get file information (size, modified, etc.)")
    def file_info(self, path: str) -> dict:
        p = Path(path)
        stat = p.stat()
        return {
            "path": str(p.resolve()), "name": p.name, "extension": p.suffix,
            "size_bytes": stat.st_size, "is_file": p.is_file(), "is_dir": p.is_dir(),
            "modified": stat.st_mtime, "created": stat.st_ctime,
        }

    @tool(description="Copy a file or directory")
    def copy(self, source: str, destination: str) -> str:
        if os.path.isdir(source):
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        return f"Copied {source} -> {destination}"

    @tool(description="Move/rename a file or directory")
    def move(self, source: str, destination: str) -> str:
        shutil.move(source, destination)
        return f"Moved {source} -> {destination}"

    @tool(description="Delete a file or directory")
    def delete(self, path: str) -> str:
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(path)
        else:
            p.unlink()
        return f"Deleted {path}"

    @tool(description="Create a directory (including parents)")
    def mkdir(self, path: str) -> str:
        Path(path).mkdir(parents=True, exist_ok=True)
        return f"Created directory {path}"

    @tool(description="Check if a file or directory exists")
    def exists(self, path: str) -> bool:
        return Path(path).exists()

    @tool(description="Read a file as lines with line numbers")
    def read_lines(self, path: str, start: int = 1, end: int = -1) -> list:
        with open(path) as f:
            lines = f.readlines()
        if end == -1:
            end = len(lines)
        return [{"line": i, "content": l.rstrip()} for i, l in enumerate(lines[start - 1 : end], start)]

    @tool(description="Replace text in a file")
    def replace_in_file(self, path: str, old: str, new: str) -> str:
        content = Path(path).read_text()
        count = content.count(old)
        updated = content.replace(old, new)
        Path(path).write_text(updated)
        return f"Replaced {count} occurrences in {path}"

    @tool(description="Get the size of a directory tree")
    def dir_size(self, path: str) -> dict:
        total = 0
        file_count = 0
        for p in Path(path).rglob("*"):
            if p.is_file():
                total += p.stat().st_size
                file_count += 1
        return {"total_bytes": total, "total_mb": round(total / 1048576, 2), "file_count": file_count}

    @tool(description="Get the current working directory")
    def cwd(self) -> str:
        return os.getcwd()

    @tool(description="Create a temporary file with content")
    def temp_file(self, content: str, suffix: str = ".txt") -> str:
        import tempfile
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    @tool(description="Read binary file as base64")
    def read_binary_base64(self, path: str) -> str:
        import base64
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    @tool(description="Get file hash (MD5, SHA256)")
    def file_hash(self, path: str, algorithm: str = "sha256") -> str:
        import hashlib
        h = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @tool(description="Watch a directory for changes (one-shot check)")
    def dir_diff(self, path: str, snapshot: dict = None) -> dict:
        current = {}
        for p in Path(path).rglob("*"):
            if p.is_file():
                current[str(p)] = p.stat().st_mtime
        if snapshot is None:
            return {"snapshot": current, "changes": []}
        added = set(current) - set(snapshot)
        removed = set(snapshot) - set(current)
        modified = {f for f in set(current) & set(snapshot) if current[f] != snapshot[f]}
        return {"snapshot": current, "changes": {"added": list(added), "removed": list(removed), "modified": list(modified)}}
