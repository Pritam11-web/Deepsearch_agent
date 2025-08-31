
from typing_extensions import TypedDict
from typing import Annotated, List, Optional, Literal
from pydantic import BaseModel, Field
from langsmith import traceable
import operator
import asyncio
from googleapiclient.discovery import build

from langgraph.types import Send
from langgraph.graph import START, END, StateGraph
from langsmith import traceable
from langchain.chat_models import init_chat_model
from google.colab import userdata #Import the .env in case of non colab enviornment
import os

# Assuming configuration.py and prompts.py are in the same directory or importable
from configuration import Configuration
from prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions,
    section_writer_instructions,
    final_section_writer_instructions
)

# Seting Up enviornmental variables for search engine(for normal enviorment need to put inside .env)
google_cse_api_key = userdata.get('GOOGLE_CSE_API_KEY') # Assuming the API key is stored in GOOGLE_API_KEY
google_cse_id = userdata.get('GOOGLE_CSE_ID') # Assuming the CSE ID is stored in user data

# Seting Up enviornmental variables for Gemini and other apis (for normal enviorment need to put inside .env)
os.environ["GOOGLE_API_KEY"]= userdata.get('GOOGLE_API_KEY')

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# Classes for Core Graph State

class Section(BaseModel):
    name: str = Field(
        description= "Name for this section of the report",
    )
    description: str = Field(
        description= "brief overview of the main topics and concpets covered in this section",
    )
    research: bool = Field(
        description= "Whether to perform web search for this section of the report"
    )
    content: str = Field(
        description= "The content of the section"
    )
class Sections(BaseModel):
    sections: List[Section] = Field(
        description= "List of sections in the report.",
    )
class SearchQuery(BaseModel):
    search_query: str = Field(
        None, description= "Query of the web search."
    )
class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description= "List of search queries.",
    )
class ReportState(TypedDict):
    topic: str #Report Topic
    tavily_topic: Literal["general", "news"] #Tavily Search topic
    tavily_days: Optional[int] #only applicable for the news topic
    report_structure: str #Report structure
    numbers_of_queries: int #Number of web search queries to performe per section
    sections: list[Section] #List of report sections
    completed_sections: Annotated[list, operator.add] #send() Api key
    report_sections_from_research: str #string of any completed sections from research to write final sectins
    final_report: str #Final report

#Section Classes


#Classes for Section Graph state

class SectionState(TypedDict):
    section: Section # Report section
    search_queries: list[SearchQuery] # List of search queries
    source_str: str # String of formatted source content from web search
    report_sections_from_research: str # String of any completed sections from research to write final sections
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API
    query_writer_instructions: str # Instructions for generating search queries
    section_writer_instructions: str # Instructions for writing a section

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

# Graph Input and Output States
class ReportStateInput(TypedDict):
    topic: str # Report topic

class ReportStateOutput(TypedDict):
    final_report: str # Final report

#Function for web search
@traceable
async def google_cse_search_async(search_queries: List[SearchQuery]):
    """
    Performs concurrent web searches using the Google Custom Search Engine API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process.
        tavily_days (Optional[int]): This parameter is not used in Google CSE and is included for compatibility.

    Returns:
        List[dict]: List of search results from Google CSE API, one per query.
                    Each item in the list is a dictionary representing the search results for a single query.
                    The structure will differ from Tavily, but we will attempt to make it compatible.
    """

    google_cse_client = build("customsearch", "v1", developerKey=google_cse_api_key)
    semaphore = asyncio.Semaphore(10)

    async def fetch_with_semaphore(query):
        """A helper function to run a single search query within the semaphore."""
        async with semaphore:
            # Build the search request inside the semaphore context
            search_request = google_cse_client.cse().list(
                q=query.search_query,
                cx=google_cse_id,
                num=5
            )
            # Use asyncio.to_thread for async execution
            return await asyncio.to_thread(search_request.execute)

    # Create a list of tasks, each running with the semaphore
    search_tasks = [fetch_with_semaphore(query) for query in search_queries]

    # Execute all searches concurrently, limited by the semaphore
    search_results = await asyncio.gather(*search_tasks)

    return search_results


#Function for Formating and removing duplicate after web search

def deduplicate_and_format_sources(
    cse_responses: list[dict],
    max_tokens_per_source: int = 800
) -> str:
    """
    Processes a list of Google CSE API responses to extract and format unique sources.

    This function is specifically designed for the structure of the Google Custom
    Search Engine API JSON response.

    Args:
        cse_responses: A list of dictionary objects, where each object is a
                       raw response from the Google CSE API.
        max_tokens_per_source: The approximate token limit for the content snippet.

    Returns:
        A formatted string containing the title, URL, and content for each unique source.
    """
    unique_sources = {}

    # 1. Iterate through each API response and then through the items in it
    for response in cse_responses:
        for item in response.get('items', []):
            url = item.get('link')
            # 2. Use the URL as a key to automatically handle duplicates
            if url and url not in unique_sources:
                unique_sources[url] = {
                    'title': item.get('title', 'No Title Provided'),
                    'content': item.get('snippet', 'No snippet available.')
                }

    # 3. Format the collected unique sources into the desired string output
    formatted_parts = []
    for url, source_data in unique_sources.items():
        title = source_data['title']
        content = source_data['content']

        # Simple token truncation using a 4 characters/token approximation
        char_limit = max_tokens_per_source * 4
        if len(content) > char_limit:
            content = content[:char_limit] + "... [truncated]"

        # Construct the output string for each source
        # Note: Both "Most relevant" and "Full source" use the same 'snippet'
        # from the API, as that's the only content provided.
        source_string = (
            f"Source: {title}\n===\n"
            f"URL: {url}\n===\n"
            f"Most relevant content from source: {source_data['content']}\n===\n"
            f"Full source content limited to {max_tokens_per_source} tokens: {content}"
        )
        formatted_parts.append(source_string)

    return "\n\n".join(formatted_parts)

# For Report Planning

async def generate_report_plan(state: ReportState, config: RunnableConfig):

    # Inputs
    topic = state["topic"]
    #Input
    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    tavily_topic = configurable.tavily_topic
    tavily_days = configurable.tavily_days
    #tavily_days = state.get("tavily_days", None)

    #convert JSON object to string if neccessory
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    #Generate Search Query
    structured_llm = llm.with_structured_output(Queries)


    # Format  system instructions
    system_instruction_query = report_planner_query_writer_instructions.format(
        topic=topic,
        report_organization=report_structure,
        number_of_queries=number_of_queries

    )

    #Generate queries
    result = structured_llm.invoke([SystemMessage(content=system_instruction_query)]+[HumanMessage(content="Generate search queries that will help in planning the report outline")])
    #print(result)

    #Web search
    query_list = result.queries # Use the list of SearchQuery objects directly
    # Replace tavily_search_async with google_cse_search_async
    search_docs = await google_cse_search_async(query_list) # Pass tavily_days for compatibility, though not used by google_cse_search_async
    #print(f"Query list {query_list}")
    #print(f" searched doc output {search_docs}")

    #Duplicate and format sources
    source_str = deduplicate_and_format_sources(search_docs, max_tokens_per_source = 800)
    #print(f"source str {source_str}")

    #Format system Instruction
    system_instructions_sections = report_planner_instructions.format(
        topic=state["topic"],
        report_organization=report_structure,
        context=source_str
    )

    #Generate Sections
    structured_llm = llm.with_structured_output(Sections)
    report_sections = structured_llm.invoke([SystemMessage(content=system_instructions_sections)]+[HumanMessage(content="Generate the sections of the report")])

    return {"sections": report_sections.sections}


# Generting Query for section writting

def generate_queries(state: SectionState, config: RunnableConfig):
    """ Generate search queries for a report section """

    # Get state
    section = state["section"]
    query_writer_instructions = state["query_writer_instructions"] # Access from state

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries
    structured_llm = llm.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(section_topic=section.description, number_of_queries=number_of_queries)

    # Generate queries
    queries = structured_llm.invoke([SystemMessage(content=system_instructions)]+[HumanMessage(content="Generate search queries on the provided topic.")])

    return {"search_queries": queries.queries}


# Implementing web search

async def search_web(state: SectionState, config: RunnableConfig):
    """ Search the web for each query, then return a list of raw sources and a formatted string of sources."""

    # Get state
    search_queries = state["search_queries"]
    print(f"search_query1: {search_queries}")
    print(type(search_queries))

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    tavily_topic = configurable.tavily_topic
    tavily_days = configurable.tavily_days


    # Web search
    query_list = [query.search_query for query in search_queries]
    print(f"Querylist2: {query_list}")
    print(type(query_list))
    search_docs = await google_cse_search_async(search_queries) # Pass the list of SearchQuery objects directly

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(search_docs, max_tokens_per_source=5000)

    return {"source_str": source_str}

# writing sections which need web search

def write_section(state: SectionState):
    """ Write a section of the report """

    # Get state
    section = state["section"]
    source_str = state["source_str"]
    section_writer_instructions = state["section_writer_instructions"] # Access from state

    # Format system instructions
    system_instructions = section_writer_instructions.format(section_topic=section.description, context=source_str)

    # Generate section
    section_content = llm.invoke([SystemMessage(content=system_instructions)]+[HumanMessage(content="Generate a report section based on the provided sources.")])

    # Write content to the section object
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

# Add nodes and edges(for the subgraph)
section_builder = StateGraph(SectionState, output_schema=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")
section_builder.add_edge("write_section", END)


# This is the "map" step when we kick off web research for some sections of the report
def initiate_section_writing(state: ReportState):

    # Kick off section writing in parallel via Send() API for any sections that require research
    return [
        Send("build_section_with_web_research", {
            "section": s,
            "query_writer_instructions": query_writer_instructions, # Pass instructions to the subgraph
            "section_writer_instructions": section_writer_instructions # Pass instructions to the subgraph
        })
        for s in state["sections"]
        if s.research
    ]


def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research:
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str

def write_final_sections(state: SectionState):
    """ Write final sections of the report, which do not require web search and use the completed sections as context """

    # Get state
    section = state["section"]

    # Safely access report_sections_from_research, provide empty string if not found
    completed_report_sections = state.get("report_sections_from_research", "")

    # Format system instructions
    system_instructions = final_section_writer_instructions.format(section_topic=section.name, context=completed_report_sections)


    # Generate section
    section_content = llm.invoke([SystemMessage(content=system_instructions)]+[HumanMessage(content="Generate a report section based on the provided sources.")])

    # Write content to section
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

def gather_completed_sections(state: ReportState):
    """ Gather completed sections from research and format them as context for writing the final sections """

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    print(f"Gathered completed sections. Formatted context length: {len(completed_report_sections)}") # Debug print
    print(f"State after gather_completed_sections: {state.keys()}") # Debug print

    return {"report_sections_from_research": completed_report_sections}

# Modified function to be a regular node
def initiate_final_sections_node(state: ReportState):
    """ Node to initiate writing any final sections using the Send API to parallelize the process """
    print(f"Initiate final sections node state keys: {state.keys()}") # Debug print

    # Safely access report_sections_from_research, provide empty string if not found
    report_sections_from_research = state.get("report_sections_from_research", "")

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections", {"section": s, "report_sections_from_research": report_sections_from_research})
        for s in state["sections"]
        if not s.research
    ]

def compile_final_report(state: ReportState):
    """ Compile the final report """

    # Get sections
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections[section.name]

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections])

    return {"final_report": all_sections}

# Building the main Graph

# Add nodes and edges
builder = StateGraph(ReportState, input_schema=ReportStateInput, output_schema=ReportStateOutput, context_schema= Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Define the edges
builder.add_edge(START, "generate_report_plan")
# Initiate research sections in parallel based on the plan
builder.add_conditional_edges("generate_report_plan", initiate_section_writing, ["build_section_with_web_research"])
# After research sections are built, gather them
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
# initiate uncompleted section writting in async. by sending completed sections
builder.add_conditional_edges("gather_completed_sections", initiate_final_sections_node, ["write_final_sections"])
# After final sections are written (collected via Send), compile the report
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)


graph = builder.compile()
