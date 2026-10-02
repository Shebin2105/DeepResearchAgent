import sys
import os
import asyncio
from dotenv import load_dotenv

project_root = os.path.abspath(".")
sys.path.append(project_root)
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from deepresearch.utils import get_today_str
from deepresearch.prompt import final_report_generation_prompt
from deepresearch.state import AgentState, AgentInputState
from deepresearch.agents.scoping_agent import clarify_with_user, write_research_brief
from deepresearch.agents.supervisor_agent import supervisor_agent

_DEFAULT_FINAL_REPORT_MODEL = "groq:qwen/qwen3-32b"

async def final_report_generation(state: AgentState, config: RunnableConfig):
    cfg = config.get("configurable", {})
    model_name = cfg.get("final_report_model", _DEFAULT_FINAL_REPORT_MODEL)
    writer_model = init_chat_model(model_name, temperature=0.3)

    notes = state.get("notes", [])
    findings = "\n\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str()
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])
    return {
        "final_report": final_report.content,
        "messages": ["Here is the final report: " + final_report.content],
    }

async def run_research():
    deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)
    deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
    deep_researcher_builder.add_node("write_research_brief", write_research_brief)
    deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
    deep_researcher_builder.add_node("final_report_generation", final_report_generation)

    deep_researcher_builder.add_edge(START, "clarify_with_user")
    deep_researcher_builder.add_edge("write_research_brief", "supervisor_subgraph")
    deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
    deep_researcher_builder.add_edge("final_report_generation", END)

    agent = deep_researcher_builder.compile()
    print("Graph compiled.")

    config = {
        "configurable": {
            "thread_id": "test_run_1",
            "allow_clarification": False,
            "max_react_tool_calls": 3,
            "min_search_iterations": 2,
            "max_supervisor_history": 4
        }
    }
    
    query = "Compare the number of IPL titles won by Chennai Super Kings versus Mumbai Indians."
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    print(f"Running agent for query: {query}")
    result = await agent.ainvoke(initial_state, config=config)
    print("\nReport Generated Successfully!\n")
    print(result.get("final_report", "No report generated."))

if __name__ == "__main__":
    asyncio.run(run_research())
