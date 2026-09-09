"""Image, PDF, Docker, Database, Email, Archive, Text, API, System, Calendar Tools."""
from __future__ import annotations
import os, json, subprocess, time, datetime, re, sqlite3, zipfile, tarfile, platform
from pathlib import Path
from nexusai.tools.base import BaseTool, tool

class ImageTool(BaseTool):
    name = "image"; description = "Image processing — resize, convert, metadata"; category = "media"
    @tool(description="Get image info")
    def info(self, path: str) -> dict:
        try:
            from PIL import Image; img = Image.open(path); return {"size": img.size, "mode": img.mode, "format": img.format, "path": path}
        except ImportError: return {"error": "Pillow not installed"}
    @tool(description="Resize an image")
    def resize(self, path: str, width: int, height: int, output: str = None) -> str:
        from PIL import Image; img = Image.open(path); img = img.resize((width, height)); out = output or path; img.save(out); return f"Resized to {width}x{height}: {out}"
    @tool(description="Convert image format")
    def convert(self, path: str, output: str) -> str:
        from PIL import Image; Image.open(path).save(output); return f"Converted: {output}"
    @tool(description="Create thumbnail")
    def thumbnail(self, path: str, size: int = 128, output: str = None) -> str:
        from PIL import Image; img = Image.open(path); img.thumbnail((size, size)); out = output or f"thumb_{Path(path).name}"; img.save(out); return f"Thumbnail: {out}"
    @tool(description="Rotate image")
    def rotate(self, path: str, degrees: float, output: str = None) -> str:
        from PIL import Image; img = Image.open(path).rotate(degrees, expand=True); out = output or path; img.save(out); return f"Rotated {degrees}°: {out}"

class PDFTool(BaseTool):
    name = "pdf"; description = "PDF operations — read, extract, merge"; category = "document"
    @tool(description="Extract text from PDF")
    def extract_text(self, path: str) -> str:
        try:
            from pypdf import PdfReader; reader = PdfReader(path); return "\n".join(page.extract_text() or "" for page in reader.pages)[:10000]
        except ImportError: return "pypdf not installed"
    @tool(description="Get PDF metadata")
    def metadata(self, path: str) -> dict:
        try:
            from pypdf import PdfReader; reader = PdfReader(path); return {"pages": len(reader.pages), "metadata": {k: str(v) for k, v in (reader.metadata or {}).items()}}
        except ImportError: return {"error": "pypdf not installed"}
    @tool(description="Extract text from specific pages")
    def extract_pages(self, path: str, start: int = 0, end: int = -1) -> str:
        from pypdf import PdfReader; reader = PdfReader(path); end = end if end > 0 else len(reader.pages); return "\n".join(reader.pages[i].extract_text() or "" for i in range(start, min(end, len(reader.pages))))

class DockerTool(BaseTool):
    name = "docker"; description = "Docker container management"; category = "infrastructure"
    def _run(self, cmd: str) -> str:
        r = subprocess.run(f"docker {cmd}", shell=True, capture_output=True, text=True, timeout=60); return r.stdout.strip() if r.returncode == 0 else f"Error: {r.stderr.strip()}"
    @tool(description="List containers")
    def ps(self, all: bool = False) -> str: return self._run(f"ps {'--all' if all else ''}")
    @tool(description="List images")
    def images(self) -> str: return self._run("images")
    @tool(description="Run a container")
    def run(self, image: str, command: str = "", flags: str = "--rm") -> str: return self._run(f"run {flags} {image} {command}")
    @tool(description="Build an image")
    def build(self, path: str = ".", tag: str = "nexusai-build") -> str: return self._run(f"build -t {tag} {path}")
    @tool(description="Stop a container")
    def stop(self, container: str) -> str: return self._run(f"stop {container}")
    @tool(description="Remove a container")
    def rm(self, container: str) -> str: return self._run(f"rm {container}")
    @tool(description="Docker compose up")
    def compose_up(self, path: str = ".", detach: bool = True) -> str: return self._run(f"compose -f {path}/docker-compose.yml up {'-d' if detach else ''}")
    @tool(description="Docker compose down")
    def compose_down(self, path: str = ".") -> str: return self._run(f"compose -f {path}/docker-compose.yml down")
    @tool(description="View container logs")
    def logs(self, container: str, tail: int = 50) -> str: return self._run(f"logs --tail {tail} {container}")

class DatabaseTool(BaseTool):
    name = "database"; description = "Database operations — SQLite, PostgreSQL, MySQL"; category = "data"
    @tool(description="Execute SQL on SQLite database")
    def sqlite_query(self, db_path: str, query: str) -> list:
        conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row; cur = conn.execute(query); rows = [dict(r) for r in cur.fetchall()]; conn.commit(); conn.close(); return rows
    @tool(description="Create SQLite table")
    def sqlite_create_table(self, db_path: str, table: str, columns: dict) -> str:
        cols = ", ".join(f"{k} {v}" for k, v in columns.items())
        conn = sqlite3.connect(db_path); conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols})"); conn.commit(); conn.close(); return f"Created table {table}"
    @tool(description="Insert data into SQLite")
    def sqlite_insert(self, db_path: str, table: str, data: dict) -> str:
        keys = ", ".join(data.keys()); placeholders = ", ".join("?" * len(data))
        conn = sqlite3.connect(db_path); conn.execute(f"INSERT INTO {table} ({keys}) VALUES ({placeholders})", list(data.values())); conn.commit(); conn.close(); return "Inserted 1 row"
    @tool(description="List SQLite tables")
    def sqlite_tables(self, db_path: str) -> list: return self.sqlite_query(db_path, "SELECT name FROM sqlite_master WHERE type='table'")

class EmailTool(BaseTool):
    name = "email"; description = "Email composition and sending"; category = "communication"
    @tool(description="Send email via SMTP")
    def send(self, to: str, subject: str, body: str, smtp_server: str = None, smtp_port: int = 587, username: str = None, password: str = None) -> str:
        import smtplib; from email.mime.text import MIMEText
        smtp_server = smtp_server or os.environ.get("SMTP_SERVER", "smtp.gmail.com"); username = username or os.environ.get("SMTP_USER", ""); password = password or os.environ.get("SMTP_PASS", "")
        msg = MIMEText(body); msg["Subject"] = subject; msg["From"] = username; msg["To"] = to
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as s: s.starttls(); s.login(username, password); s.send_message(msg); return f"Email sent to {to}"
        except Exception as e: return f"Failed: {e}"
    @tool(description="Compose email (returns MIME)")
    def compose(self, to: str, subject: str, body: str, from_addr: str = "nexusai@local") -> str:
        from email.mime.text import MIMEText; msg = MIMEText(body); msg["Subject"] = subject; msg["From"] = from_addr; msg["To"] = to; return msg.as_string()

class ArchiveTool(BaseTool):
    name = "archive"; description = "Archive operations — ZIP, TAR, GZIP"; category = "filesystem"
    @tool(description="Create ZIP archive")
    def create_zip(self, output: str, files: list) -> str:
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files: z.write(f, Path(f).name)
        return f"Created {output} with {len(files)} files"
    @tool(description="Extract ZIP archive")
    def extract_zip(self, path: str, dest: str = ".") -> str:
        with zipfile.ZipFile(path) as z: z.extractall(dest); return f"Extracted {len(z.namelist())} files to {dest}"
    @tool(description="List ZIP contents")
    def list_zip(self, path: str) -> list:
        with zipfile.ZipFile(path) as z: return z.namelist()
    @tool(description="Create TAR archive")
    def create_tar(self, output: str, files: list, compress: str = "gz") -> str:
        mode = f"w:{compress}" if compress else "w"
        with tarfile.open(output, mode) as t:
            for f in files: t.add(f, Path(f).name)
        return f"Created {output}"
    @tool(description="Extract TAR archive")
    def extract_tar(self, path: str, dest: str = ".") -> str:
        with tarfile.open(path) as t: t.extractall(dest, filter="data"); return f"Extracted to {dest}"

class TextTool(BaseTool):
    name = "text"; description = "Text processing — NLP, regex, formatting"; category = "text"
    @tool(description="Word count and text statistics")
    def word_count(self, text: str) -> dict:
        words = text.split(); chars = len(text); lines = text.count("\n") + 1; sentences = len(re.split(r'[.!?]+', text))
        return {"words": len(words), "characters": chars, "lines": lines, "sentences": sentences, "paragraphs": text.count("\n\n") + 1}
    @tool(description="Find text with regex")
    def regex_find(self, text: str, pattern: str) -> list: return re.findall(pattern, text)
    @tool(description="Replace text with regex")
    def regex_replace(self, text: str, pattern: str, replacement: str) -> str: return re.sub(pattern, replacement, text)
    @tool(description="Extract emails from text")
    def extract_emails(self, text: str) -> list: return re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', text)
    @tool(description="Extract URLs from text")
    def extract_urls(self, text: str) -> list: return re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
    @tool(description="Extract phone numbers from text")
    def extract_phones(self, text: str) -> list: return re.findall(r'[\+]?[1-9]\d{0,2}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,9}', text)
    @tool(description="Summarize text (simple extractive)")
    def summarize(self, text: str, sentences: int = 3) -> str:
        sents = re.split(r'(?<=[.!?])\s+', text)
        return " ".join(sents[:sentences]) if sents else text[:500]
    @tool(description="Convert case")
    def convert_case(self, text: str, case: str = "lower") -> str:
        cases = {"lower": text.lower(), "upper": text.upper(), "title": text.title(), "capitalize": text.capitalize(), "snake": re.sub(r'\s+', '_', text.lower()), "kebab": re.sub(r'\s+', '-', text.lower()), "camel": text[0].lower() + re.sub(r'[\s_-]+(\w)', lambda m: m.group(1).upper(), text)[1:]}
        return cases.get(case, text)
    @tool(description="Generate Lorem Ipsum")
    def lorem_ipsum(self, words: int = 50) -> str:
        base = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat".split()
        import random; return " ".join(random.choices(base, k=words)) + "."
    @tool(description="Diff two texts")
    def diff(self, text1: str, text2: str) -> str:
        import difflib; d = difflib.unified_diff(text1.splitlines(), text2.splitlines(), lineterm=""); return "\n".join(d)

class APITool(BaseTool):
    name = "api"; description = "REST API builder and testing"; category = "web"
    @tool(description="Generate API endpoint boilerplate (FastAPI)")
    def generate_endpoint(self, path: str, method: str = "GET", response_model: str = "dict") -> str:
        return f'@app.{method.lower()}("{path}")\nasync def handler():\n    return {{"message": "ok"}}'
    @tool(description="Generate OpenAPI spec")
    def generate_openapi(self, title: str, paths: dict) -> dict:
        return {"openapi": "3.0.0", "info": {"title": title, "version": "1.0.0"}, "paths": paths}
    @tool(description="Test an API endpoint")
    def test_endpoint(self, url: str, method: str = "GET", data: dict = None, headers: dict = None) -> dict:
        import httpx; resp = httpx.request(method, url, json=data, headers=headers, timeout=30)
        return {"status": resp.status_code, "body": resp.text[:5000], "headers": dict(resp.headers)}

class SystemTool(BaseTool):
    name = "system"; description = "System information and management"; category = "system"
    @tool(description="Get CPU info")
    def cpu_info(self) -> dict:
        try:
            import psutil; return {"cores": psutil.cpu_count(), "physical_cores": psutil.cpu_count(logical=False), "percent": psutil.cpu_percent(interval=1), "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}}
        except ImportError: return {"cores": os.cpu_count(), "platform": platform.processor()}
    @tool(description="Get uptime")
    def uptime(self) -> str:
        try: import psutil; boot = datetime.datetime.fromtimestamp(psutil.boot_time()); return str(datetime.datetime.now() - boot)
        except: return "unknown"
    @tool(description="Get environment variables")
    def env_vars(self) -> dict: return dict(os.environ)
    @tool(description="Get temp directory path")
    def temp_dir(self) -> str: import tempfile; return tempfile.gettempdir()
    @tool(description="Get current timestamp")
    def timestamp(self) -> dict: now = datetime.datetime.now(); return {"iso": now.isoformat(), "unix": time.time(), "human": now.strftime("%Y-%m-%d %H:%M:%S")}
    @tool(description="Sleep for N seconds")
    def sleep(self, seconds: float) -> str: time.sleep(seconds); return f"Slept for {seconds}s"

class CalendarTool(BaseTool):
    name = "calendar"; description = "Date and time operations"; category = "datetime"
    @tool(description="Get current date and time")
    def now(self) -> dict: n = datetime.datetime.now(); return {"date": n.strftime("%Y-%m-%d"), "time": n.strftime("%H:%M:%S"), "day": n.strftime("%A"), "iso": n.isoformat()}
    @tool(description="Add days to a date")
    def add_days(self, date: str, days: int) -> str: d = datetime.datetime.fromisoformat(date) + datetime.timedelta(days=days); return d.strftime("%Y-%m-%d")
    @tool(description="Difference between two dates")
    def date_diff(self, date1: str, date2: str) -> dict:
        d1, d2 = datetime.datetime.fromisoformat(date1), datetime.datetime.fromisoformat(date2); diff = abs(d2 - d1)
        return {"days": diff.days, "hours": diff.total_seconds() / 3600, "seconds": diff.total_seconds()}
    @tool(description="Parse date string")
    def parse_date(self, text: str, fmt: str = "%Y-%m-%d") -> dict:
        d = datetime.datetime.strptime(text, fmt); return {"iso": d.isoformat(), "day": d.strftime("%A"), "timestamp": d.timestamp()}
    @tool(description="Format timestamp")
    def format_timestamp(self, timestamp: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str: return datetime.datetime.fromtimestamp(timestamp).strftime(fmt)
    @tool(description="Is it a weekday?")
    def is_weekday(self, date: str = None) -> bool: d = datetime.datetime.fromisoformat(date) if date else datetime.datetime.now(); return d.weekday() < 5
    @tool(description="Get calendar month")
    def month_calendar(self, year: int = None, month: int = None) -> str:
        import calendar; now = datetime.datetime.now(); return calendar.month(year or now.year, month or now.month)
