"""Generation token limits aligned with configs/gemma4_4b/sft.yaml."""

from __future__ import annotations

# configs/gemma4_4b/sft.yaml model.max_seq_length
SFT_MAX_SEQ_LENGTH = 8192
# Headroom for system prompt + long compose/edit/repair user text.
PROMPT_TOKEN_RESERVE = 1024

DEFAULT_MAX_NEW_TOKENS = SFT_MAX_SEQ_LENGTH - PROMPT_TOKEN_RESERVE
MAX_NEW_TOKENS_CAP = SFT_MAX_SEQ_LENGTH
