# bootstrap_index.py
# One-time index creation (if you want to ensure the index exists on startup).

import logging
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests.auth import HTTPBasicAuth
from config import ELASTIC_URL, ELASTIC_USERNAME, ELASTIC_PASSWORD

def ensure_discord_index():
    try:
        auth = HTTPBasicAuth(ELASTIC_USERNAME, ELASTIC_PASSWORD) if (ELASTIC_USERNAME or ELASTIC_PASSWORD) else None
        cli = OpenSearch(
            hosts=[ELASTIC_URL],
            http_auth=auth,
            verify_certs=False,
            connection_class=RequestsHttpConnection,
            timeout=10,
        )
        if not cli.indices.exists("discord_chat_memory"):
            body = {
                "settings": {
                    "index": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "refresh_interval": "1s"
                    }
                },
                "mappings": {
                    "dynamic": False,
                    "properties": {
                        "user_id":    {"type": "keyword"},
                        "guild_id":   {"type": "keyword"},
                        "channel_id": {"type": "keyword"},
                        "role":       {"type": "keyword"},
                        "reply_to":   {"type": "keyword"},
                        "timestamp":  {"type": "date"},
                        "content": {
                            "type": "text",
                            "fields": {"exact": {"type": "keyword", "ignore_above": 2048}}
                        },
                    }
                }
            }
            cli.indices.create("discord_chat_memory", body=body)
            logging.info("Created index discord_chat_memory")
    except Exception as e:
        logging.warning(f"Index bootstrap failed: {e}")
