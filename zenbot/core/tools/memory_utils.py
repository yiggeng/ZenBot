import os
import json
import uuid
import threading
from datetime import datetime
from typing import List, Dict, Optional

from ..config import MEMORY_DIR

MEMORIES_DIR = os.path.join(MEMORY_DIR, "memories")
INDEX_PATH = os.path.join(MEMORIES_DIR, "index.json")

VALID_CATEGORIES = ["fact", "preference", "decision", "project", "technical", "general"]

_lock = threading.Lock()


def _ensure_dir():
    os.makedirs(MEMORIES_DIR, exist_ok=True)


def _load_index() -> List[Dict]:
    if not os.path.exists(INDEX_PATH):
        return []
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_index(index: List[Dict]):
    _ensure_dir()
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _write_memory_file(memory_id: str, timestamp: str, category: str, keywords: str, content: str) -> str:
    _ensure_dir()
    date_slug = timestamp.replace("-", "").replace(":", "").replace(" ", "_")
    filename = f"{date_slug}_{memory_id}.md"
    filepath = os.path.join(MEMORIES_DIR, filename)

    keywords_display = keywords if keywords else "无"
    file_content = (
        f"<!-- memory: {memory_id} -->\n"
        f"# {timestamp} | {category}\n"
        f"**keywords:** {keywords_display}\n\n"
        f"{content}\n"
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(file_content)
    return filename


def _read_memory_file(filename: str) -> Optional[str]:
    filepath = os.path.join(MEMORIES_DIR, filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def save_memory_to_disk(content: str, category: str, keywords: str) -> str:
    if category not in VALID_CATEGORIES:
        category = "general"

    memory_id = uuid.uuid4().hex[:4]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    filename = _write_memory_file(memory_id, timestamp, category, keywords, content)

    with _lock:
        index = _load_index()
        index.append({
            "id": memory_id,
            "file": filename,
            "timestamp": timestamp,
            "category": category,
            "keywords": [k.strip() for k in keywords.split(",") if k.strip()],
            "summary": content[:80],
        })
        _save_index(index)

    return f"记忆已保存。ID: {memory_id} | 分类: {category} | 关键词: {keywords}"


def search_memories_on_disk(query: str, max_results: int = 5) -> str:
    keywords = [k.strip().lower() for k in query.split() if k.strip()]
    if not keywords:
        return "请提供搜索关键词。"

    index = _load_index()
    if not index:
        return "记忆库为空，暂无任何记忆。"

    # 先用 index 中的 keywords + summary 做粗筛，避免全量读文件
    candidates = []
    for entry in index:
        index_text = (" ".join(entry.get("keywords", [])) + " " + entry.get("summary", "")).lower()
        matches = sum(1 for kw in keywords if kw in index_text)
        if matches > 0:
            candidates.append((matches, entry))

    # 对粗筛命中的条目再读全文精确打分
    scored = []
    for pre_score, entry in candidates:
        file_content = _read_memory_file(entry.get("file", ""))
        if not file_content:
            continue
        full_text = (file_content + " " + " ".join(entry.get("keywords", []))).lower()
        matches = sum(1 for kw in keywords if kw in full_text)
        if matches > 0:
            scored.append((matches, entry, file_content))

    if not scored:
        return f"未找到与 \"{query}\" 相关的记忆。"

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for i, (score, entry, content) in enumerate(scored[:max_results], 1):
        results.append(
            f"[{i}] ID: {entry['id']} | {entry['timestamp']} | {entry['category']}\n"
            f"    关键词: {', '.join(entry.get('keywords', []))}\n"
            f"    {content}"
        )
    return f"找到 {len(scored)} 条相关记忆（显示前 {min(len(scored), max_results)} 条）：\n\n" + "\n\n".join(results)


def list_memories_on_disk(category: str = "", limit: int = 20) -> str:
    index = _load_index()
    if not index:
        return "记忆库为空，暂无任何记忆。"

    filtered = index
    if category:
        filtered = [e for e in index if e.get("category") == category]
        if not filtered:
            return f"没有找到分类为 \"{category}\" 的记忆。"

    filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    filtered = filtered[:limit]

    results = []
    for entry in filtered:
        kw = ", ".join(entry.get("keywords", []))
        results.append(
            f"- [{entry['id']}] {entry['timestamp']} | {entry['category']} | {kw}\n  {entry.get('summary', '')}"
        )
    total = len(index)
    shown = len(filtered)
    return f"共 {total} 条记忆（显示 {shown} 条）：\n" + "\n".join(results)


def delete_memory_on_disk(memory_id: str) -> str:
    with _lock:
        index = _load_index()
        target = None
        for entry in index:
            if entry.get("id") == memory_id:
                target = entry
                break

        if not target:
            return f"未找到 ID 为 {memory_id} 的记忆。"

        filepath = os.path.join(MEMORIES_DIR, target.get("file", ""))
        if os.path.exists(filepath):
            os.remove(filepath)

        new_index = [e for e in index if e.get("id") != memory_id]
        _save_index(new_index)
    return f"记忆 [{memory_id}] 已删除。"


def load_recent_memories(limit: int = 10) -> str:
    index = _load_index()
    if not index:
        return ""

    sorted_index = sorted(index, key=lambda x: x.get("timestamp", ""), reverse=True)
    entries = sorted_index[:limit]

    lines = []
    for entry in entries:
        file_content = _read_memory_file(entry.get("file", ""))
        if file_content:
            body_lines = file_content.strip().split("\n")
            body = "\n".join(body_lines[3:]).strip() if len(body_lines) > 3 else entry.get("summary", "")
            lines.append(f"- [{entry['category']}] {body}")
        else:
            lines.append(f"- [{entry['category']}] {entry.get('summary', '')}")

    return "\n".join(lines)
