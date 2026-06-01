"""
API Key Manager
===============

Simplified API key management.

Usage:
    from src.auth import APIKeyManager

    manager = APIKeyManager()
    key = manager.create(user_id="user_123")
    user = manager.verify(key)
"""

from typing import Optional, Dict, Any
from datetime import datetime
import secrets


class APIKeyManager:
    """
    Simplified API key manager.

    Example:
        manager = APIKeyManager()

        # Create key
        key = manager.create(user_id="user_123")

        # Verify key
        user_id = manager.verify(key)

        # Revoke key
        manager.revoke(key)
    """

    def __init__(self):
        """Initialize API key manager."""
        self.keys: Dict[str, Dict[str, Any]] = {}

    def create(
        self,
        user_id: str,
        name: Optional[str] = None,
        permissions: Optional[list] = None
    ) -> str:
        """
        Create API key.

        Args:
            user_id: User ID
            name: Key name
            permissions: Key permissions

        Returns:
            API key string
        """
        key = f"rag_{secrets.token_urlsafe(32)}"

        self.keys[key] = {
            "user_id": user_id,
            "name": name,
            "permissions": permissions or ["read"],
            "created_at": datetime.now(),
            "is_active": True,
            "usage_count": 0,
        }

        return key

    def verify(self, key: str) -> Optional[str]:
        """
        Verify API key.

        Args:
            key: API key

        Returns:
            User ID or None
        """
        record = self.keys.get(key)

        if not record or not record["is_active"]:
            return None

        record["usage_count"] += 1
        return record["user_id"]

    def revoke(self, key: str) -> None:
        """
        Revoke API key.

        Args:
            key: API key
        """
        if key in self.keys:
            self.keys[key]["is_active"] = False

    def list_keys(self, user_id: str) -> list:
        """
        List keys for user.

        Args:
            user_id: User ID

        Returns:
            List of key records
        """
        return [
            {"key": k[:10] + "...", **v}
            for k, v in self.keys.items()
            if v["user_id"] == user_id and v["is_active"]
        ]
