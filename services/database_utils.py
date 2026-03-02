from services.sqlite_store import SQLiteStore


store = SQLiteStore()


def initialize_logs_table():
    return None


def log_message(conversation_id, user_id, user_msg, bot_msg):
    store.log_message(conversation_id, user_id, user_msg, bot_msg)


def fetch_conversation(conversation_id):
    return store.fetch_conversation(conversation_id)


def create_user_location_table():
    return None


def insert_or_update_user_location(user_id, location):
    store.insert_or_update_user_location(user_id, location)


def fetch_user_location(user_id):
    return store.fetch_user_location(user_id)


def create_memory_consent_table():
    return None


def set_memory_consent(user_id, consent: bool):
    store.set_memory_consent(user_id, consent)


def has_opted_in_memory(user_id):
    return store.has_opted_in_memory(user_id)


def index_user_message(*args, **kwargs):
    return None


def init_message_expansions():
    return None


def save_message_expansion(message_id: int, full_text: str, expanded: bool = False):
    store.save_message_expansion(message_id, full_text, expanded)


def get_message_expansion(message_id: int):
    return store.get_message_expansion(message_id)


def set_message_expanded(message_id: int, expanded: bool):
    store.set_message_expanded(message_id, expanded)


def init_user_instructions():
    return None


def set_user_instruction(user_id: str, instruction: str):
    store.set_user_instruction(user_id, instruction)


def get_user_instruction(user_id: str) -> str | None:
    return store.get_user_instruction(user_id)


def init_sora_usage():
    return None


def log_sora_usage(user_id: str, video_id: str = None):
    store.log_sora_usage(user_id, video_id=video_id)


def get_last_sora_video_id(user_id: str) -> str | None:
    return store.get_last_sora_video_id(user_id)


def check_sora_limit(user_id: str, limit: int = 2, window_seconds: int = 3600) -> bool:
    return store.check_sora_limit(user_id, limit=limit, window_seconds=window_seconds)
