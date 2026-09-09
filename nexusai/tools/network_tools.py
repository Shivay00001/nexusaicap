"""Network Tools — DNS, ping, port scanning, IP info."""
from __future__ import annotations
import socket, subprocess, platform
from nexusai.tools.base import BaseTool, tool

class NetworkTool(BaseTool):
    name = "network"; description = "Network diagnostics — DNS, ping, ports, IP info"; category = "network"

    @tool(description="DNS lookup")
    def dns_lookup(self, hostname: str) -> dict:
        try: ips = socket.getaddrinfo(hostname, None); return {"hostname": hostname, "ips": list(set(ip[4][0] for ip in ips))}
        except Exception as e: return {"error": str(e)}
    @tool(description="Reverse DNS lookup")
    def reverse_dns(self, ip: str) -> dict:
        try: return {"ip": ip, "hostname": socket.gethostbyaddr(ip)[0]}
        except Exception as e: return {"ip": ip, "error": str(e)}
    @tool(description="Ping a host")
    def ping(self, host: str, count: int = 4) -> dict:
        param = "-n" if platform.system() == "Windows" else "-c"
        r = subprocess.run(["ping", param, str(count), host], capture_output=True, text=True, timeout=30)
        return {"success": r.returncode == 0, "output": r.stdout}
    @tool(description="Check if a port is open")
    def port_check(self, host: str, port: int, timeout: int = 5) -> dict:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(timeout)
        try: s.connect((host, port)); s.close(); return {"host": host, "port": port, "open": True}
        except: return {"host": host, "port": port, "open": False}
    @tool(description="Scan common ports on a host")
    def port_scan(self, host: str, ports: list = None) -> list:
        ports = ports or [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3306, 3389, 5432, 5900, 6379, 8080, 8443, 27017]
        results = []
        for p in ports:
            r = self.port_check(host, p, timeout=2)
            if r["open"]: results.append(r)
        return results
    @tool(description="Get local IP address")
    def local_ip(self) -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try: s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
        finally: s.close()
    @tool(description="Get public IP address")
    def public_ip(self) -> str:
        import httpx; return httpx.get("https://api.ipify.org", timeout=10).text
    @tool(description="Whois lookup")
    def whois(self, domain: str) -> str:
        r = subprocess.run(["whois" if platform.system() != "Windows" else "nslookup", domain], capture_output=True, text=True, timeout=15)
        return r.stdout[:3000]
    @tool(description="Traceroute")
    def traceroute(self, host: str) -> str:
        cmd = "tracert" if platform.system() == "Windows" else "traceroute"
        r = subprocess.run([cmd, host], capture_output=True, text=True, timeout=30)
        return r.stdout[:3000]
    @tool(description="Get hostname")
    def hostname(self) -> str: return socket.gethostname()
