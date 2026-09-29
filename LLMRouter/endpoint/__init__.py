"""Router HTTP Endpoint Server

暴露訓練好的 LLMRouter 為 HTTP API，相容 semantic-router r1_server_url 協議。
"""

from .server import LLMRouterEndpointServer

__all__ = ["LLMRouterEndpointServer"]
