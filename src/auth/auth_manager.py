"""
Auth Manager
============

Authentication and authorization manager.

Features:
- User management
- API key authentication
- JWT tokens
- Role-based access control

Usage:
    from src.auth import AuthManager

    auth = AuthManager()

    # Create user
    auth.create_user("user_123", roles=["user"])

    # Create API key
    api_key = auth.create_api_key("user_123")

    # Verify request
    user = auth.verify_api_key(api_key)
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import secrets
import json


@dataclass
class User:
    """User record."""
    id: str
    roles: List[str]
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


@dataclass
class APIKey:
    """API key record."""
    key: str
    user_id: str
    permissions: List[str]
    rate_limit: int  # requests per minute
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool = True
    usage_count: int = 0


class AuthManager:
    """
    Authentication and authorization manager.

    Manages users, API keys, and access control.

    Example:
        auth = AuthManager()

        # Create user
        auth.create_user("user_123", roles=["user"])

        # Create API key
        api_key = auth.create_api_key(
            user_id="user_123",
            permissions=["read", "write"],
            rate_limit=100,
            expires_in_days=30
        )

        # Verify request
        user = auth.verify_api_key(api_key)
        if user:
            # Process request
            pass
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize auth manager.

        Args:
            storage_path: Path for storing auth data
        """
        self.storage_path = storage_path
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}

        # Load existing data
        if storage_path:
            self._load_data()

    def create_user(
        self,
        user_id: str,
        roles: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> User:
        """
        Create a new user.

        Args:
            user_id: User ID
            roles: User roles
            metadata: Additional metadata

        Returns:
            User record
        """
        user = User(
            id=user_id,
            roles=roles or ["user"],
            created_at=datetime.now(),
            metadata=metadata or {},
        )

        self.users[user_id] = user
        self._save_data()

        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User record or None
        """
        return self.users.get(user_id)

    def update_user(
        self,
        user_id: str,
        roles: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update user.

        Args:
            user_id: User ID
            roles: New roles
            metadata: New metadata
        """
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")

        if roles is not None:
            user.roles = roles
        if metadata is not None:
            user.metadata.update(metadata)

        self._save_data()

    def delete_user(self, user_id: str) -> None:
        """
        Delete user.

        Args:
            user_id: User ID
        """
        if user_id not in self.users:
            raise ValueError(f"User not found: {user_id}")

        # Deactivate user
        self.users[user_id].is_active = False

        # Deactivate all API keys
        for api_key in self.api_keys.values():
            if api_key.user_id == user_id:
                api_key.is_active = False

        self._save_data()

    def create_api_key(
        self,
        user_id: str,
        permissions: Optional[List[str]] = None,
        rate_limit: int = 100,
        expires_in_days: Optional[int] = 30
    ) -> str:
        """
        Create API key for user.

        Args:
            user_id: User ID
            permissions: API key permissions
            rate_limit: Rate limit (requests per minute)
            expires_in_days: Days until expiration

        Returns:
            API key string
        """
        if user_id not in self.users:
            raise ValueError(f"User not found: {user_id}")

        # Generate API key
        key = f"rag_{secrets.token_urlsafe(32)}"

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        # Create record
        api_key = APIKey(
            key=key,
            user_id=user_id,
            permissions=permissions or ["read"],
            rate_limit=rate_limit,
            created_at=datetime.now(),
            expires_at=expires_at,
        )

        self.api_keys[key] = api_key
        self._save_data()

        return key

    def verify_api_key(self, key: str) -> Optional[User]:
        """
        Verify API key and return user.

        Args:
            key: API key

        Returns:
            User record or None
        """
        api_key = self.api_keys.get(key)

        if not api_key:
            return None

        if not api_key.is_active:
            return None

        # Check expiration
        if api_key.expires_at and datetime.now() > api_key.expires_at:
            api_key.is_active = False
            return None

        # Get user
        user = self.users.get(api_key.user_id)

        if not user or not user.is_active:
            return None

        # Update usage
        api_key.usage_count += 1

        return user

    def check_permission(self, key: str, permission: str) -> bool:
        """
        Check if API key has permission.

        Args:
            key: API key
            permission: Permission to check

        Returns:
            True if has permission
        """
        api_key = self.api_keys.get(key)

        if not api_key or not api_key.is_active:
            return False

        return permission in api_key.permissions

    def revoke_api_key(self, key: str) -> None:
        """
        Revoke API key.

        Args:
            key: API key
        """
        if key in self.api_keys:
            self.api_keys[key].is_active = False
            self._save_data()

    def list_api_keys(self, user_id: str) -> List[APIKey]:
        """
        List API keys for user.

        Args:
            user_id: User ID

        Returns:
            List of API key records
        """
        return [
            api_key for api_key in self.api_keys.values()
            if api_key.user_id == user_id and api_key.is_active
        ]

    def get_usage_stats(self, key: str) -> Dict[str, Any]:
        """
        Get usage statistics for API key.

        Args:
            key: API key

        Returns:
            Usage statistics
        """
        api_key = self.api_keys.get(key)

        if not api_key:
            raise ValueError(f"API key not found: {key}")

        return {
            "key": key[:10] + "...",
            "user_id": api_key.user_id,
            "usage_count": api_key.usage_count,
            "rate_limit": api_key.rate_limit,
            "created_at": api_key.created_at.isoformat(),
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            "is_active": api_key.is_active,
        }

    def _save_data(self) -> None:
        """Save auth data to storage."""
        if not self.storage_path:
            return

        data = {
            "users": {
                uid: {
                    "id": u.id,
                    "roles": u.roles,
                    "created_at": u.created_at.isoformat(),
                    "metadata": u.metadata,
                    "is_active": u.is_active,
                }
                for uid, u in self.users.items()
            },
            "api_keys": {
                key: {
                    "key": k.key,
                    "user_id": k.user_id,
                    "permissions": k.permissions,
                    "rate_limit": k.rate_limit,
                    "created_at": k.created_at.isoformat(),
                    "expires_at": k.expires_at.isoformat() if k.expires_at else None,
                    "is_active": k.is_active,
                    "usage_count": k.usage_count,
                }
                for key, k in self.api_keys.items()
            },
        }

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_data(self) -> None:
        """Load auth data from storage."""
        if not self.storage_path or not Path(self.storage_path).exists():
            return

        with open(self.storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load users
        for uid, u in data.get("users", {}).items():
            self.users[uid] = User(
                id=u["id"],
                roles=u["roles"],
                created_at=datetime.fromisoformat(u["created_at"]),
                metadata=u.get("metadata", {}),
                is_active=u.get("is_active", True),
            )

        # Load API keys
        for key, k in data.get("api_keys", {}).items():
            self.api_keys[key] = APIKey(
                key=k["key"],
                user_id=k["user_id"],
                permissions=k["permissions"],
                rate_limit=k["rate_limit"],
                created_at=datetime.fromisoformat(k["created_at"]),
                expires_at=datetime.fromisoformat(k["expires_at"]) if k.get("expires_at") else None,
                is_active=k.get("is_active", True),
                usage_count=k.get("usage_count", 0),
            )
