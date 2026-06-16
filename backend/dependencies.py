from fastapi import Request, HTTPException

from core.agent_flow import AgenticRAG


def get_agent(request: Request) -> AgenticRAG:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 服务尚未初始化")
    return agent
