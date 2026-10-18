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


def get_conversation_persona_enabled(scope_key: str) -> bool:
    return store.get_conversation_persona_enabled(scope_key)


def set_conversation_persona_enabled(scope_key: str, enabled: bool) -> None:
    store.set_conversation_persona_enabled(scope_key, enabled)


def propose_behavior_change(user_id: str, instruction: str, *, created_by: str | None = None) -> int:
    return store.propose_behavior_change(user_id, instruction, created_by=created_by)


def activate_behavior_change(user_id: str, change_id: int) -> dict:
    return store.activate_behavior_change(user_id, change_id)


def rollback_behavior_change(user_id: str) -> dict | None:
    return store.rollback_behavior_change(user_id)


def get_behavior_change(user_id: str, change_id: int) -> dict | None:
    return store.get_behavior_change(user_id, change_id)


def list_behavior_changes(user_id: str, limit: int = 10) -> list[dict]:
    return store.list_behavior_changes(user_id, limit)


def create_code_proposal(owner_id: str, request: str, baseline_sha: str) -> int:
    return store.create_code_proposal(owner_id, request, baseline_sha)


def set_code_proposal_patch(owner_id: str, proposal_id: int, patch: str) -> dict:
    return store.set_code_proposal_patch(owner_id, proposal_id, patch)


def set_code_proposal_validation(owner_id: str, proposal_id: int, report: dict) -> dict:
    return store.set_code_proposal_validation(owner_id, proposal_id, report)


def review_code_proposal(owner_id: str, proposal_id: int, decision: str, *, reviewer_id: str) -> dict:
    return store.review_code_proposal(
        owner_id, proposal_id, decision, reviewer_id=reviewer_id
    )


def review_any_code_proposal(proposal_id: int, decision: str, *, reviewer_id: str) -> dict:
    return store.review_any_code_proposal(
        proposal_id, decision, reviewer_id=reviewer_id
    )


def set_code_proposal_approval_message(
    proposal_id: int, channel_id: str, message_id: str
) -> None:
    store.set_code_proposal_approval_message(proposal_id, channel_id, message_id)


def get_code_proposal(owner_id: str, proposal_id: int) -> dict | None:
    return store.get_code_proposal(owner_id, proposal_id)


def get_any_code_proposal(proposal_id: int) -> dict | None:
    return store.get_any_code_proposal(proposal_id)


def list_code_proposals(owner_id: str, limit: int = 10) -> list[dict]:
    return store.list_code_proposals(owner_id, limit)


def list_all_code_proposals(limit: int = 20) -> list[dict]:
    return store.list_all_code_proposals(limit)


def get_code_deployment(owner_id: str, proposal_id: int) -> dict | None:
    return store.get_code_deployment(owner_id, proposal_id)


def request_code_rollback(owner_id: str, proposal_id: int) -> int:
    return store.request_code_rollback(owner_id, proposal_id)


def request_any_code_rollback(reviewer_id: str, proposal_id: int) -> int:
    return store.request_any_code_rollback(reviewer_id, proposal_id)


def add_user_fact(user_id: str, fact: str, category: str | None = None) -> int:
    return store.add_user_fact(user_id, fact, category)


def list_user_facts(user_id: str, limit: int = 50) -> list[dict]:
    return store.list_user_facts(user_id, limit)


def delete_user_fact(user_id: str, fact_id: int) -> bool:
    return store.delete_user_fact(user_id, fact_id)


def delete_user_facts_matching(user_id: str, query: str) -> int:
    return store.delete_user_facts_matching(user_id, query)


def get_user_profile(user_id: str) -> dict | None:
    return store.get_user_profile(user_id)


def set_user_profile(user_id: str, profile: str) -> None:
    store.set_user_profile(user_id, profile)


def get_user_seen(user_id: str) -> dict | None:
    return store.get_user_seen(user_id)


def set_user_seen(user_id: str, *, intent: str | None = None, prompt: str | None = None) -> None:
    store.set_user_seen(user_id, intent=intent, prompt=prompt)


def veo_limit_status(user_id: str, limit: int = 2, window_seconds: int = 3600) -> dict:
    return store.veo_limit_status(user_id, limit=limit, window_seconds=window_seconds)


def get_cached_transcript_summary(video_id: str) -> str | None:
    return store.get_cached_transcript_summary(video_id)


def set_cached_transcript_summary(video_id: str, summary: str) -> None:
    store.set_cached_transcript_summary(video_id, summary)


def get_channel_last_seen(key: str) -> str | None:
    return store.get_channel_last_seen(key)


def set_channel_last_seen(key: str, last_seen_id: str) -> None:
    store.set_channel_last_seen(key, last_seen_id)


def init_veo_usage():
    return None


def log_veo_usage(user_id: str, video_id: str = None):
    store.log_veo_usage(user_id, video_id=video_id)


def check_veo_limit(user_id: str, limit: int = 2, window_seconds: int = 3600) -> bool:
    return store.check_veo_limit(user_id, limit=limit, window_seconds=window_seconds)
