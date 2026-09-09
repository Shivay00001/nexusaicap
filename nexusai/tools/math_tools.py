"""Math Tools — Mathematical computations."""
from __future__ import annotations
import math, statistics, random
from nexusai.tools.base import BaseTool, tool

class MathTool(BaseTool):
    name = "math"; description = "Mathematical operations, statistics, and number theory"; category = "math"

    @tool(description="Evaluate a math expression") 
    def evaluate(self, expression: str) -> float: return float(eval(expression, {"__builtins__": {}}, {"math": math, "pi": math.pi, "e": math.e, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10, "abs": abs, "pow": pow, "round": round}))
    @tool(description="Calculate statistics for a list of numbers")
    def stats(self, numbers: list) -> dict: nums = [float(n) for n in numbers]; return {"mean": statistics.mean(nums), "median": statistics.median(nums), "stdev": statistics.stdev(nums) if len(nums) > 1 else 0, "variance": statistics.variance(nums) if len(nums) > 1 else 0, "min": min(nums), "max": max(nums), "sum": sum(nums), "count": len(nums)}
    @tool(description="Generate random number")
    def random_number(self, min_val: float = 0, max_val: float = 1, integer: bool = False) -> float: return random.randint(int(min_val), int(max_val)) if integer else random.uniform(min_val, max_val)
    @tool(description="Calculate factorial")
    def factorial(self, n: int) -> int: return math.factorial(n)
    @tool(description="Calculate GCD")
    def gcd(self, a: int, b: int) -> int: return math.gcd(a, b)
    @tool(description="Calculate LCM")
    def lcm(self, a: int, b: int) -> int: return abs(a * b) // math.gcd(a, b)
    @tool(description="Check if number is prime")
    def is_prime(self, n: int) -> bool:
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True
    @tool(description="Convert between number bases")
    def convert_base(self, number: str, from_base: int, to_base: int) -> str: n = int(number, from_base); return bin(n)[2:] if to_base == 2 else oct(n)[2:] if to_base == 8 else hex(n)[2:] if to_base == 16 else str(n)
    @tool(description="Fibonacci sequence up to n terms")
    def fibonacci(self, n: int) -> list:
        seq = [0, 1]
        for _ in range(n - 2): seq.append(seq[-1] + seq[-2])
        return seq[:n]
    @tool(description="Matrix operations (multiply, transpose, determinant)")
    def matrix_multiply(self, a: list, b: list) -> list: return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
    @tool(description="Calculate percentage")
    def percentage(self, value: float, total: float) -> float: return round((value / total) * 100, 2) if total else 0
    @tool(description="Unit conversion")
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        conversions = {"km_mi": 0.621371, "mi_km": 1.60934, "kg_lb": 2.20462, "lb_kg": 0.453592, "c_f": lambda x: x * 9/5 + 32, "f_c": lambda x: (x - 32) * 5/9, "m_ft": 3.28084, "ft_m": 0.3048}
        key = f"{from_unit}_{to_unit}"
        factor = conversions.get(key)
        if callable(factor): return round(factor(value), 4)
        return round(value * factor, 4) if factor else value
