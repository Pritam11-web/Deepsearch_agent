
import asyncio
from main import graph

async def run_report_generation(topic: str):
    """Runs the report generation graph with the given topic."""
    initial_state = {"topic": topic}
    finalResult = await graph.ainvoke(initial_state)
    return finalResult

if __name__ == "__main__":
    # Example usage:
    topic = "Crtically analyse how AI Agents are brining a new revolution in the AI era"
    final_report = asyncio.run(run_report_generation(topic))
    print(final_report['final_report'])
