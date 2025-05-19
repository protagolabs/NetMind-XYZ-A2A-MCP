from typing import Optional
import pydantic

class MakeResponseModel(pydantic.BaseModel):
    messages_list: list
    user_id: str
    mcp_info_list: Optional[list]
    other_data: Optional[dict]
