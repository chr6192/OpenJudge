# -*- coding: utf-8 -*-
"""
Model integrations module from AgentScope
"""

from open_judge.models.base_chat_model import BaseChatModel
from open_judge.models.openai_chat_model import OpenAIChatModel
from open_judge.models.qwen_vl_model import QwenVLModel

__all__ = [
    "BaseChatModel",
    "OpenAIChatModel",
    "QwenVLModel",
]
