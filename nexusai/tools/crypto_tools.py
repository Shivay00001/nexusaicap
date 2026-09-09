"""Crypto Tools — Hashing, encoding, encryption."""
from __future__ import annotations
import hashlib, hmac, base64, secrets, uuid, json
from nexusai.tools.base import BaseTool, tool

class CryptoTool(BaseTool):
    name = "crypto"; description = "Cryptography, hashing, encoding, and secrets generation"; category = "security"

    @tool(description="Generate hash (md5, sha256, sha512, etc.)")
    def hash(self, text: str, algorithm: str = "sha256") -> str: return hashlib.new(algorithm, text.encode()).hexdigest()
    @tool(description="Generate HMAC")
    def hmac_sign(self, message: str, key: str, algorithm: str = "sha256") -> str: return hmac.new(key.encode(), message.encode(), algorithm).hexdigest()
    @tool(description="Base64 encode")
    def b64_encode(self, text: str) -> str: return base64.b64encode(text.encode()).decode()
    @tool(description="Base64 decode")
    def b64_decode(self, text: str) -> str: return base64.b64decode(text).decode()
    @tool(description="Generate secure random token")
    def generate_token(self, length: int = 32) -> str: return secrets.token_hex(length)
    @tool(description="Generate UUID")
    def generate_uuid(self) -> str: return str(uuid.uuid4())
    @tool(description="Generate secure password")
    def generate_password(self, length: int = 16) -> str: return secrets.token_urlsafe(length)
    @tool(description="URL-safe base64 encode")
    def b64_url_encode(self, text: str) -> str: return base64.urlsafe_b64encode(text.encode()).decode()
    @tool(description="Hex encode")
    def hex_encode(self, text: str) -> str: return text.encode().hex()
    @tool(description="Hex decode")
    def hex_decode(self, hex_string: str) -> str: return bytes.fromhex(hex_string).decode()
    @tool(description="Generate JWT (unsigned, for testing)")
    def mock_jwt(self, payload: dict) -> str:
        header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).decode().rstrip("=")
        body = base64.urlsafe_b64encode(json.dumps(payload, default=str).encode()).decode().rstrip("=")
        return f"{header}.{body}."
    @tool(description="Decode JWT payload (without verification)")
    def decode_jwt(self, token: str) -> dict:
        parts = token.split(".")
        if len(parts) >= 2:
            padding = 4 - len(parts[1]) % 4
            payload = base64.urlsafe_b64decode(parts[1] + "=" * padding)
            return json.loads(payload)
        return {"error": "Invalid JWT"}
    @tool(description="Compare two strings in constant time")
    def safe_compare(self, a: str, b: str) -> bool: return hmac.compare_digest(a, b)
    @tool(description="ROT13 encode/decode")
    def rot13(self, text: str) -> str: import codecs; return codecs.encode(text, "rot_13")
