from typing import Optional, Dict, List, Union, Any, cast
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
import jwt  # Using PyJWT instead of python-jose
from datetime import datetime, timedelta, timezone
from loguru import logger
import secrets
from pydantic import BaseModel, Field, field_validator, ConfigDict
import time
import hashlib
# After
import os
from dotenv import load_dotenv

load_dotenv()
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Using in-memory cache only.")
from pathlib import Path
import json
import asyncio
from functools import wraps
from config.settings import settings

# Custom exceptions
class AuthenticationError(Exception):
    """Base class for authentication errors."""
    def __init__(self, message: str, error_code: str):
        super().__init__(message)
        self.error_code = error_code

class TokenError(AuthenticationError):
    """Token-related errors."""
    pass

class PermissionError(AuthenticationError):
    """Permission-related errors."""
    pass

# Security schemes with custom error handling
class CustomAPIKeyHeader(APIKeyHeader):
    async def __call__(self, request: Request) -> Optional[str]:
        try:
            return await super().__call__(request)
        except HTTPException as e:
            if e.status_code == 403:
                logger.warning(
                    f"Invalid API key attempt from {request.client.host}",
                    extra={
                        'ip': request.client.host,
                        'path': request.url.path,
                        'method': request.method
                    }
                )
            return None

# API Security schemes
api_key_header = CustomAPIKeyHeader(name="X-API-Key", auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


import os
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "").encode('utf-8')
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY must be set in environment variables")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_DAYS = settings.REFRESH_TOKEN_EXPIRE_DAYS

# Redis configuration
redis_client: Optional[redis.Redis] = None
if REDIS_AVAILABLE and settings.USE_REDIS:
    try:
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_timeout=5,
            retry_on_timeout=True
        )
        # Test connection
        redis_client.ping()
        logger.info("Redis connection established successfully")
    except Exception as e:
        logger.error(f"Redis connection failed: {str(e)}")
        redis_client = None

# Pydantic models with V2 style validation
class Token(BaseModel):
    """Token response model."""
    model_config = ConfigDict(extra='forbid')
    
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int

    @field_validator('token_type')
    @classmethod
    def validate_token_type(cls, v: str) -> str:
        if v.lower() != 'bearer':
            raise ValueError('Token type must be "bearer"')
        return v.lower()

    @field_validator('expires_in')
    @classmethod
    def validate_expires_in(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('expires_in must be positive')
        return v

class TokenData(BaseModel):
    """Token data model."""
    model_config = ConfigDict(extra='forbid')
    
    username: Optional[str] = None
    exp: Optional[int] = None
    permissions: List[str] = Field(default_factory=list)
    
    @field_validator('exp')
    @classmethod
    def validate_expiration(cls, v: Optional[int]) -> Optional[int]:
        if v and v < time.time():
            raise ValueError('Token has expired')
        return v

class UserPermission(BaseModel):
    """User permission model."""
    model_config = ConfigDict(extra='forbid')
    
    name: str
    description: str
    scope: str

    @field_validator('scope')
    @classmethod
    def validate_scope(cls, v: str) -> str:
        valid_scopes = ['global', 'limited', 'custom']
        if v not in valid_scopes:
            raise ValueError(f'Invalid scope. Must be one of: {valid_scopes}')
        return v

class AuthData(BaseModel):
    """Authentication data model."""
    model_config = ConfigDict(extra='forbid')
    
    user_id: str
    username: str
    permissions: List[str] = Field(default_factory=list)
    rate_limit: Dict[str, int]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('permissions')
    @classmethod
    def validate_permissions(cls, v: List[str]) -> List[str]:
        valid_permissions = ['read', 'write', 'admin']
        for perm in v:
            if perm not in valid_permissions:
                raise ValueError(f'Invalid permission: {perm}')
        return v

    @field_validator('rate_limit')
    @classmethod
    def validate_rate_limit(cls, v: Dict[str, int]) -> Dict[str, int]:
        if 'requests_per_minute' not in v:
            raise ValueError('rate_limit must include requests_per_minute')
        if v['requests_per_minute'] < 1:
            raise ValueError('requests_per_minute must be positive')
        return v

class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    model_config = ConfigDict(extra='forbid')
    
    window_size: int = Field(60, ge=1)  # seconds
    max_requests: int = Field(100, ge=1)
    burst_size: int = Field(10, ge=1)

    @field_validator('burst_size')
    @classmethod
    def validate_burst(cls, v: int, info: Dict[str, Any]) -> int:
        if 'max_requests' in info.data and v > info.data['max_requests']:
            raise ValueError('burst_size cannot exceed max_requests')
        return v

class AuthException(HTTPException):
    """Enhanced authentication exception with error codes."""
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        if headers is None:
            headers = {"WWW-Authenticate": "Bearer"}
        super().__init__(
            status_code=status_code,
            detail=detail,
            headers=headers
        )
        self.error_code = error_code
class CacheManager:
    """Manage authentication data caching with Redis fallback."""
    def __init__(self):
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get item from cache with Redis priority."""
        if redis_client:
            try:
                data = redis_client.get(key)
                if data:
                    return json.loads(data)
            except redis.RedisError as e:
                logger.warning(f"Redis get failed: {str(e)}")

        async with self._lock:
            cache_item = self._memory_cache.get(key)
            if not cache_item:
                return None
                
            if cache_item['expires'] > time.time():
                return cache_item['data']
                
            del self._memory_cache[key]
            return None

    async def set(
        self,
        key: str,
        data: Dict[str, Any],
        expire: int
    ) -> None:
        """Set item in cache with Redis priority."""
        if redis_client:
            try:
                redis_client.setex(
                    key,
                    expire,
                    json.dumps(data)
                )
                return
            except redis.RedisError as e:
                logger.warning(f"Redis set failed: {str(e)}")

        async with self._lock:
            self._memory_cache[key] = {
                'data': data,
                'expires': time.time() + expire
            }

    async def delete(self, key: str) -> None:
        """Delete item from cache."""
        if redis_client:
            try:
                redis_client.delete(key)
            except redis.RedisError as e:
                logger.warning(f"Redis delete failed: {str(e)}")

        async with self._lock:
            self._memory_cache.pop(key, None)

    async def clear_expired(self) -> None:
        """Clear expired cache items."""
        current_time = time.time()
        
        async with self._lock:
            expired_keys = [
                key for key, item in self._memory_cache.items()
                if item['expires'] <= current_time
            ]
            for key in expired_keys:
                del self._memory_cache[key]

class AuthManager:
    """Manage authentication and authorization."""
    def __init__(self):
        self.cache = CacheManager()
        self.rate_limiter = RateLimiter()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Periodic cleanup of expired cache items."""
        while True:
            try:
                await self.cache.clear_expired()
                await asyncio.sleep(300)  # Clean every 5 minutes
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
                await asyncio.sleep(60)

    async def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token with enhanced security."""
        try:
            to_encode = data.copy()
            expire = datetime.now(timezone.utc) + (
                expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            )
            
            to_encode.update({
                "exp": int(expire.timestamp()),
                "iat": int(datetime.now(timezone.utc).timestamp()),
                "jti": secrets.token_urlsafe(32),
                "type": "access"
            })
            
            encoded_jwt = jwt.encode(
                to_encode,
                JWT_SECRET_KEY,
                algorithm=JWT_ALGORITHM
            )
            
            # Store token metadata
            await self.cache.set(
                f"token:{to_encode['jti']}",
                {
                    "user_id": data.get("sub"),
                    "expires": to_encode["exp"],
                    "type": "access"
                },
                int(expire.timestamp() - time.time())
            )
            
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Token creation failed: {str(e)}")
            raise AuthException(
                status_code=500,
                detail="Token creation failed",
                error_code="TOKEN_CREATION_FAILED"
            )

    async def create_refresh_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create refresh token with enhanced security."""
        try:
            token = secrets.token_urlsafe(64)
            expire = datetime.now(timezone.utc) + (
                expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
            )
            
            await self.cache.set(
                f"refresh_token:{user_id}",
                {
                    "token": token,
                    "user_id": user_id,
                    "expires": int(expire.timestamp())
                },
                int(expire.timestamp() - time.time())
            )
            
            return token
            
        except Exception as e:
            logger.error(f"Refresh token creation failed: {str(e)}")
            raise AuthException(
                status_code=500,
                detail="Refresh token creation failed",
                error_code="REFRESH_TOKEN_CREATION_FAILED"
            )

    async def validate_token(
        self,
        token: str,
        token_type: str = "access"
    ) -> Optional[AuthData]:
        """Validate token with comprehensive checks."""
        try:
            # Check token blacklist
            if await self._is_token_blacklisted(token):
                raise AuthException(
                    401,
                    "Token has been revoked",
                    "TOKEN_REVOKED"
                )

            try:
                payload = jwt.decode(
                    token,
                    JWT_SECRET_KEY,
                    algorithms=[JWT_ALGORITHM]
                )
            except jwt.ExpiredSignatureError:
                raise AuthException(
                    401,
                    "Token has expired",
                    "TOKEN_EXPIRED"
                )
            except jwt.InvalidTokenError:
                raise AuthException(
                    401,
                    "Invalid token",
                    "INVALID_TOKEN"
                )

            # Validate token claims
            required_claims = ["sub", "exp", "jti", "type"]
            if not all(claim in payload for claim in required_claims):
                raise AuthException(
                    401,
                    "Invalid token claims",
                    "INVALID_CLAIMS"
                )

            # Validate token type
            if payload["type"] != token_type:
                raise AuthException(
                    401,
                    f"Invalid token type. Expected {token_type}",
                    "INVALID_TOKEN_TYPE"
                )

            # Check token in cache
            token_data = await self.cache.get(f"token:{payload['jti']}")
            if not token_data:
                raise AuthException(
                    401,
                    "Token not found",
                    "TOKEN_NOT_FOUND"
                )

            return await self._get_auth_data(payload["sub"])

        except AuthException:
            raise
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            raise AuthException(
                401,
                "Token validation failed",
                "VALIDATION_ERROR"
            )

    async def validate_api_key(self, api_key: str) -> Optional[AuthData]:
        """Validate API key with enhanced security."""
        if not api_key:
            return None

        try:
            # Check cache
            cache_key = f"api_key:{hashlib.sha256(api_key.encode()).hexdigest()}"
            cached_data = await self.cache.get(cache_key)
            
            if cached_data:
                return AuthData(**cached_data)


            API_KEY = os.getenv("API_KEY", "")
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            stored_hash = hashlib.sha256(API_KEY.encode()).hexdigest() if API_KEY else ""
            if key_hash == stored_hash:
                auth_data = AuthData(
                    user_id="api_user",
                    username="api_user",
                    permissions=["read", "write"],
                    rate_limit={"requests_per_minute": settings.API_RATE_LIMIT},
                    metadata={"api_key": True}
                )
                
                # Cache result
                await self.cache.set(
                    cache_key,
                    auth_data.dict(),
                    300  # 5 minutes
                )
                
                return auth_data
            
            return None

        except Exception as e:
            logger.error(f"API key validation error: {str(e)}")
            return None

    async def _get_auth_data(self, user_id: str) -> AuthData:
        """Get user authentication data with caching."""
        cache_key = f"user_auth:{user_id}"
        
        # Check cache
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            return AuthData(**cached_data)

        # In production, get from database
        # For demo, return mock data
        auth_data = AuthData(
            user_id=user_id,
            username=user_id,
            permissions=["read", "write"],
            rate_limit={"requests_per_minute": settings.DEFAULT_RATE_LIMIT},
            metadata={}
        )
        
        # Cache result
        await self.cache.set(
            cache_key,
            auth_data.dict(),
            300  # 5 minutes
        )
        
        return auth_data

    async def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted."""
        return bool(await self.cache.get(f"blacklist:{token}"))

    async def revoke_token(self, token: str, token_type: str = "access"):
        """Revoke token with type awareness."""
        try:
            payload = jwt.decode(
                token,
                JWT_SECRET_KEY,
                algorithms=[JWT_ALGORITHM]
            )
            
            await self.cache.set(
                f"blacklist:{token}",
                {
                    "revoked_at": int(time.time()),
                    "type": token_type
                },
                int(payload["exp"] - time.time())
            )
            
            if "jti" in payload:
                await self.cache.delete(f"token:{payload['jti']}")
                
        except Exception as e:
            logger.error(f"Token revocation failed: {str(e)}")
            raise AuthException(
                500,
                "Token revocation failed",
                "REVOCATION_FAILED"
            )
class RateLimiter:
    """Rate limiting with sliding window and burst handling."""
    def __init__(self):
        self.cache = CacheManager()
        self._lock = asyncio.Lock()

    async def check_rate_limit(
        self,
        user_id: str,
        limit: int,
        window: int = 60,
        burst_size: int = 10
    ) -> bool:
        """
        Check if request is within rate limits.
        
        Args:
            user_id: Unique identifier for the user
            limit: Maximum requests per window
            window: Time window in seconds
            burst_size: Maximum concurrent requests allowed
        
        Returns:
            bool: True if request is allowed, False if limit exceeded
        """
        try:
            cache_key = f"rate_limit:{user_id}"
            current_time = time.time()
            
            async with self._lock:
                # Get current request data
                cached_data = await self.cache.get(cache_key) or {
                    'requests': [],
                    'burst_count': 0
                }
                
                # Clean old requests
                requests = [
                    req_time for req_time in cached_data['requests']
                    if current_time - req_time < window
                ]
                
                # Check burst limit
                if len(requests) - len(cached_data['requests']) + cached_data['burst_count'] > burst_size:
                    return False
                
                # Check rate limit
                if len(requests) >= limit:
                    return False
                
                # Add new request
                requests.append(current_time)
                burst_count = cached_data['burst_count'] + 1
                
                # Update cache
                await self.cache.set(
                    cache_key,
                    {
                        'requests': requests,
                        'burst_count': burst_count,
                        'last_request': current_time
                    },
                    window
                )
                
                # Schedule burst count reset
                asyncio.create_task(self._reset_burst_count(cache_key))
                
                return True
                
        except Exception as e:
            logger.error(f"Rate limit check failed: {str(e)}")
            return True  # Fail open for better user experience

    async def _reset_burst_count(self, cache_key: str) -> None:
        """Reset burst count after a delay."""
        try:
            await asyncio.sleep(1)  # 1 second delay
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                cached_data['burst_count'] = max(0, cached_data['burst_count'] - 1)
                await self.cache.set(cache_key, cached_data, 60)
        except Exception as e:
            logger.error(f"Burst count reset failed: {str(e)}")

    async def get_remaining_requests(
        self,
        user_id: str,
        limit: int,
        window: int = 60
    ) -> int:
        """Get remaining allowed requests in current window."""
        try:
            cache_key = f"rate_limit:{user_id}"
            cached_data = await self.cache.get(cache_key)
            
            if not cached_data:
                return limit
            
            current_time = time.time()
            active_requests = len([
                req_time for req_time in cached_data['requests']
                if current_time - req_time < window
            ])
            
            return max(0, limit - active_requests)
            
        except Exception as e:
            logger.error(f"Failed to get remaining requests: {str(e)}")
            return 0  # Fail closed for safety

    async def get_rate_limit_info(
        self,
        user_id: str,
        limit: int,
        window: int = 60
    ) -> Dict[str, Any]:
        """Get detailed rate limit information."""
        try:
            remaining = await self.get_remaining_requests(user_id, limit, window)
            cache_key = f"rate_limit:{user_id}"
            cached_data = await self.cache.get(cache_key) or {}
            
            return {
                'limit': limit,
                'remaining': remaining,
                'reset': int(time.time() + window),
                'window': window,
                'burst_count': cached_data.get('burst_count', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get rate limit info: {str(e)}")
            return {
                'limit': limit,
                'remaining': 0,
                'reset': int(time.time() + window),
                'window': window,
                'burst_count': 0
            }

# Initialize singleton instances
auth_manager = AuthManager()
rate_limiter = RateLimiter()

# Dependency for getting current user
async def get_current_user(
    request: Request,
    api_key: str = Security(api_key_header),
    token: str = Security(oauth2_scheme)
) -> AuthData:
    """Get current user with comprehensive validation."""
    auth_data = None
    
    try:
        if api_key:
            auth_data = await auth_manager.validate_api_key(api_key)
        elif token:
            auth_data = await auth_manager.validate_token(token)

        if not auth_data:
            raise AuthException(
                401,
                "Invalid authentication credentials",
                "INVALID_AUTH"
            )

        # Check rate limit
        if not await rate_limiter.check_rate_limit(
            auth_data.user_id,
            auth_data.rate_limit["requests_per_minute"]
        ):
            raise AuthException(
                429,
                "Rate limit exceeded",
                "RATE_LIMIT_EXCEEDED"
            )

        # Add rate limit headers to response
        rate_limit_info = await rate_limiter.get_rate_limit_info(
            auth_data.user_id,
            auth_data.rate_limit["requests_per_minute"]
        )
        
        # Store rate limit info in request state
        request.state.rate_limit_info = rate_limit_info
        
        return auth_data
        
    except AuthException:
        raise
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise AuthException(
            500,
            "Authentication failed",
            "AUTH_FAILED"
        )

def requires_permissions(required_permissions: List[str]):
    """Decorator for requiring specific permissions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            auth_data = kwargs.get('auth')
            if not auth_data:
                raise AuthException(
                    401,
                    "Authentication required",
                    "AUTH_REQUIRED"
                )
                
            if not all(
                perm in auth_data.permissions
                for perm in required_permissions
            ):
                raise AuthException(
                    403,
                    "Insufficient permissions",
                    "INSUFFICIENT_PERMISSIONS"
                )
                
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Cleanup handler
import atexit

@atexit.register
def cleanup():
    """Cleanup resources on shutdown."""
    try:
        if redis_client:
            redis_client.close()
        logger.info("Auth cleanup completed")
    except Exception as e:
        logger.error(f"Auth cleanup failed: {str(e)}")
