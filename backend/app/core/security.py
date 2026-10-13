"""Authentication has been removed: every request runs as a single local user."""

DEFAULT_USER_ID = "local-user"


async def get_current_user_id() -> str:
    return DEFAULT_USER_ID
