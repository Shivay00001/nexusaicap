"""Data Tools — CSV, JSON, Excel, SQL operations."""
from __future__ import annotations
import json, csv, io
from pathlib import Path
from nexusai.tools.base import BaseTool, tool

class DataTool(BaseTool):
    name = "data"; description = "Data processing — CSV, JSON, Excel, SQL, transformations"; category = "data"

    @tool(description="Read a CSV file")
    def read_csv(self, path: str, delimiter: str = ",", max_rows: int = 100) -> list:
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            return [row for i, row in enumerate(reader) if i < max_rows]

    @tool(description="Write data to CSV")
    def write_csv(self, path: str, data: list, headers: list = None) -> str:
        if not data: return "No data to write"
        headers = headers or list(data[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader(); w.writerows(data)
        return f"Written {len(data)} rows to {path}"

    @tool(description="Read a JSON file")
    def read_json(self, path: str) -> dict:
        with open(path) as f: return json.load(f)

    @tool(description="Write data to JSON")
    def write_json(self, path: str, data: dict, indent: int = 2) -> str:
        with open(path, "w") as f: json.dump(data, f, indent=indent, default=str)
        return f"Written JSON to {path}"

    @tool(description="Parse JSON string")
    def parse_json(self, text: str) -> dict: return json.loads(text)

    @tool(description="Convert CSV to JSON")
    def csv_to_json(self, csv_path: str, json_path: str = None) -> list:
        data = self.read_csv(csv_path, max_rows=10000)
        if json_path:
            self.write_json(json_path, data)
        return data

    @tool(description="Convert JSON to CSV")
    def json_to_csv(self, json_path: str, csv_path: str) -> str:
        data = self.read_json(json_path)
        if isinstance(data, dict): data = [data]
        return self.write_csv(csv_path, data)

    @tool(description="Query data using SQL-like syntax on a CSV/JSON")
    def query(self, data: list, where: dict = None, select: list = None, order_by: str = None, limit: int = 50) -> list:
        results = data
        if where:
            for key, value in where.items():
                results = [r for r in results if str(r.get(key, "")) == str(value)]
        if select:
            results = [{k: r.get(k) for k in select} for r in results]
        if order_by:
            reverse = order_by.startswith("-")
            key = order_by.lstrip("-")
            results.sort(key=lambda x: x.get(key, ""), reverse=reverse)
        return results[:limit]

    @tool(description="Aggregate data (count, sum, avg, min, max)")
    def aggregate(self, data: list, column: str, operation: str = "count") -> dict:
        values = [float(r[column]) for r in data if column in r and r[column] not in (None, "")]
        ops = {"count": len(values), "sum": sum(values), "avg": sum(values) / len(values) if values else 0, "min": min(values) if values else None, "max": max(values) if values else None}
        return {"column": column, "operation": operation, "result": ops.get(operation, f"Unknown: {operation}")}

    @tool(description="Get data summary/statistics")
    def describe(self, data: list) -> dict:
        if not data: return {"rows": 0}
        cols = list(data[0].keys())
        return {"rows": len(data), "columns": cols, "column_count": len(cols), "sample": data[:3]}

    @tool(description="Deduplicate data")
    def deduplicate(self, data: list, key: str = None) -> list:
        seen = set()
        result = []
        for item in data:
            k = item.get(key, json.dumps(item, sort_keys=True)) if key else json.dumps(item, sort_keys=True)
            if k not in seen:
                seen.add(k); result.append(item)
        return result

    @tool(description="Merge two datasets on a common key")
    def merge(self, left: list, right: list, key: str) -> list:
        right_map = {r[key]: r for r in right if key in r}
        return [{**l, **right_map.get(l.get(key), {})} for l in left]

    @tool(description="Pivot data")
    def pivot(self, data: list, index: str, column: str, value: str) -> dict:
        result = {}
        for row in data:
            idx = row.get(index, "")
            col = row.get(column, "")
            val = row.get(value, "")
            result.setdefault(idx, {})[col] = val
        return result

    @tool(description="Read an Excel file")
    def read_excel(self, path: str, sheet: str = None) -> list:
        try:
            import pandas as pd
            df = pd.read_excel(path, sheet_name=sheet)
            return df.head(100).to_dict("records")
        except ImportError:
            return [{"error": "pandas not installed. pip install nexusai[tools]"}]
