"""
Redis-backed cache for chat history and small lookups, with in-memory fallback.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional

try:
    import redis
except ImportError:
    redis = None


class BaseCache:
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ...

    def get(self, key: str) -> Optional[Any]:
        ...

    def add_message(self, session_id: str, role: str, content: str) -> None:
        ...

    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        ...

    def set_cached_response(self, user_message: str, response: Dict[str, Any], ttl: int = 900) -> None:
        ...

    def get_cached_response(self, user_message: str) -> Optional[Dict[str, Any]]:
        ...


class RedisCache(BaseCache):
    def __init__(self):
        if not redis:
            raise ImportError("redis package not available")
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client = redis.Redis.from_url(url, decode_responses=True)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        data = json.dumps(value)
        if ttl:
            self.client.setex(key, ttl, data)
        else:
            self.client.set(key, data)

    def get(self, key: str) -> Optional[Any]:
        data = self.client.get(key)
        if data is None:
            return None
        try:
            return json.loads(data)
        except Exception:
            return data

    def add_message(self, session_id: str, role: str, content: str) -> None:
        entry = json.dumps({"role": role, "content": content})
        list_key = f"chat:{session_id}"
        pipe = self.client.pipeline()
        pipe.rpush(list_key, entry)
        pipe.ltrim(list_key, -20, -1)  # keep last 20
        pipe.execute()

    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        list_key = f"chat:{session_id}"
        items = self.client.lrange(list_key, 0, -1)
        history = []
        for item in items:
            try:
                history.append(json.loads(item))
            except Exception:
                continue
        return history

    def set_cached_response(self, user_message: str, response: Dict[str, Any], ttl: int = 900) -> None:
        key = f"resp:{user_message}"
        self.set(key, response, ttl)

    def get_cached_response(self, user_message: str) -> Optional[Dict[str, Any]]:
        key = f"resp:{user_message}"
        return self.get(key)


class SimpleCache(BaseCache):
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._history: Dict[str, List[Dict[str, str]]] = {}

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = time.time() + ttl if ttl else None
        self._store[key] = {"value": value, "expires_at": expires_at}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if not entry:
            return None
        if entry["expires_at"] and entry["expires_at"] < time.time():
            self._store.pop(key, None)
            return None
        return entry["value"]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self._history.setdefault(session_id, []).append(
            {"role": role, "content": content}
        )
        if len(self._history[session_id]) > 20:
            self._history[session_id] = self._history[session_id][-20:]

    def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        return self._history.get(session_id, [])

    def set_cached_response(self, user_message: str, response: Dict[str, Any], ttl: int = 900) -> None:
        key = f"resp:{user_message}"
        self.set(key, response, ttl)

    def get_cached_response(self, user_message: str) -> Optional[Dict[str, Any]]:
        key = f"resp:{user_message}"
        return self.get(key)


def get_cache() -> BaseCache:
    if redis:
        try:
            return RedisCache()
        except Exception:
            pass
    return SimpleCache()


cache: BaseCache = get_cache()
