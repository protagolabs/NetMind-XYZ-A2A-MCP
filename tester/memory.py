import json
from typing import List, Dict, Any

class RedisMemoryManager:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        import redis.asyncio as redis
        REDIS_PASSWORD="xyz_redis_fd6san"
        REDIS_ADDRESS="18.168.102.131"
        REDIS_PORT=6380
        self.redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_ADDRESS}:{REDIS_PORT}/0"
        self.redis_client = redis.from_url(self.redis_url)
        self.max_memory_length = 15

    def _get_memory_key(self, agent_id: str, user_id: str) -> str:
        """Generate a unique key for the chat memory"""
        return f"chat_memory:{agent_id}:{user_id}"

    async def get_memory(self, agent_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve chat memory for a specific agent-user pair"""
        key = self._get_memory_key(agent_id, user_id)
        memory_data = await self.redis_client.get(key)
        if memory_data:
            return json.loads(memory_data)
        return []

    async def add_to_memory(self, agent_id: str, user_id: str, message: Dict[str, Any]) -> None:
        """Add a message to the chat memory"""
        key = self._get_memory_key(agent_id, user_id)
        current_memory = await self.get_memory(agent_id, user_id)
        
        # Add new message
        current_memory.append(message)
        
        # Keep only the last max_memory_length messages
        if len(current_memory) > self.max_memory_length:
            current_memory = current_memory[-self.max_memory_length:]
        
        # Store updated memory
        await self.redis_client.set(key, json.dumps(current_memory))

    async def clear_memory(self, agent_id: str, user_id: str) -> None:
        """Clear chat memory for a specific agent-user pair"""
        key = self._get_memory_key(agent_id, user_id)
        await self.redis_client.delete(key) 
