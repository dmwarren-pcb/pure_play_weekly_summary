import os
import json
from typing import TypedDict, List, Annotated
import operator
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import StateGraph, END

# Import the tools we created
from brand_analysis_tools import (
    get_category_health, 
    get_performance_and_contribution_summary, 
    get_brand_and_competitor_diagnostics
)

load_dotenv()

# --- 1. Define Agent State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    category_health_summary: str
    supervisor_plan: str
    analyst_findings: str

# --- 2. Helper Functions ---
def save_markdown(filename: str, content: str):
    """Saves the given content to a markdown file."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"--- Saved output to {filename} ---")

def create_agent(llm, tools, system_prompt: str):
    """Factory function to create a new agent."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 3. Initialize LLM and Agents ---
llm = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2023-07-01-preview"),
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)

category_agent = create_agent(llm, [get_category_health], 
    "You are a market intelligence analyst. Your job is to call the `get_category_health` tool and summarize the findings in a brief, introductory paragraph.")

# UPDATED SUPERVISOR PROMPT
supervisor_agent = create_agent(llm, [get_performance_and_contribution_summary],
    """You are a Supervisor. Your role is to plan the weekly analysis.
    1. Use the `get_performance_and_contribution_summary` tool to get the data.
    2. From the `focus_brand_summary`, identify the most important brands to analyze.
    3. From the `competitor_summary`, identify the most significant competitor movements.
    4. Create a concise, numbered list of investigation tasks for the BrandAnalyst. Explicitly include tasks for both your own brands AND the key competitors you identified.
    Your output should be ONLY the numbered list of tasks.""")

# The Brand Analyst agent now uses the more advanced prompt below
brand_analyst_agent = create_agent(llm, [get_brand_and_competitor_diagnostics],
    """You are a specialist Brand Analyst agent. Your role is to interpret a rich set of pre-calculated data to explain business performance. You must not perform any mathematical calculations yourself. Your entire analysis must be based on synthesizing the metrics provided by your tools.

For each numbered task you receive, you must perform a detailed investigation.

**If the task is about one of OUR BRANDS:**
Follow this 5-step framework:
1.  **What Happened?** State the performance facts for the brand from the supervisor's task list. List the dollars for the recent period and the % chg that is associated with it to the previous period and the year 
    ago period. When applicable, look at the trend of the l4, l13, l26 and l52 periods, and explain if the trend is flat, going up, down, or reversing (ie, l4 is positive, and l13 is negative, etc).
2.  **What Drove the Change? (Price vs. Volume):** Use `get_brand_and_competitor_diagnostics` to get data. Your first step is to analyze the Price-Volume decomposition. Look at the `L4_Price_Effect` and `L4_Volume_Effect` metrics. State which of these two effects was larger and therefore the primary driver of the sales change.
3.  **Why Did That Happen? (Internal Factors):** Find the metrics that explain the primary driver. If Volume was the main driver, look for changes in `Total Traffic`, `Buy Box - Rate`. If Price was the main driver, look at the change in `Retail Price_L4` vs. `Retail Price_P4`.
4.  **What Was the Market Context?** Look at the `competitor_details` section from the tool's output. Briefly summarize their performance. Again, always look at the absolute in the recent period, and the % chg. If contribution measures are given to the category, you can summarize those as well.
5.  **Conclusion:** Provide a concise summary of your findings for that specific task. This can be in narrative form.

**If the task is about a COMPETITOR:**
Your goal is to understand what drove their performance.
1.  **State the Task:** "Investigating competitor: [Competitor Name] in [Category]."
2.  **Get Competitor Data:** Use the `get_brand_and_competitor_diagnostics` tool, passing in the *competitor's* brand and category.
3.  **Analyze Their Price vs. Volume:** Look at the `L4_Price_Effect` and `L4_Volume_Effect` for the competitor. State which factor was the primary driver of their sales change.
4.  **Conclusion:** Summarize why the competitor's sales changed. For example: "The competitor's growth of $500k was driven primarily by a strong positive Volume Effect, suggesting they gained new shoppers, rather than just taking price up."

Address each task separately and clearly.
""")

final_report_agent = create_agent(llm, [], 
    """
    You are a senior business strategist analyzing the most recent sales data from Amazon. You work in the pet food industry for Post Consumer Brands, and your brands are Rachael Ray Nutrish, Nature's Recipe, 9Lives, and Kibbles 'n Bits. Your task is to synthesize multiple inputs — a category health summary and detailed brand-level analyses — into a single, cohesive, and forward-looking business review in Markdown.

    Your primary goal is not just to report the data, but to tell a clear story about our position in the market and to provide actionable intelligence.

    **Before you write, first understand the overarching narrative by considering:**
    - What is the main trend in the category (growth, decline, stagnation)?
    - How does our focus brand's performance story fit into that larger category trend?
    - What are the key tensions or contrasts in the data? (e.g., Our brand is declining but a key competitor is soaring; the category is down but our brand is growing).

    **Structure your final report as follows:**

    **1. Executive Summary & Key Insights**
    - Begin with critical, high-level bullet points that summarize the entire situation. A leader should be able to read this section alone and understand the core challenges and opportunities. This section must be your own synthesis of all the provided information.

    **2. Category Health & Market Dynamics**
    - **Do not simply copy the provided category summary.**
    - Synthesize the key findings to set the stage. Describe the overall environment in which we are operating. Is the market expanding or contracting? What major forces are at play? Use the category summary as your source of truth to build this narrative.

    **3. Deep Dive: Our Brands vs. The Competition**
    - This section integrates the detailed findings from the Brand Analyst agents.
    - Start with an analysis of our **Brand**. Use the provided summary to explain the drivers of its performance (the "why" behind its numbers).
    - Then, for each **Key Competitor**, use their summary to describe their performance and strategy.
    - **Crucially, create a comparative analysis.** Directly contrast our brand's performance drivers against the competitors. For example: "While our volume dropped due to a 25% cut in ad spend, Competitor X saw 11% growth fueled by a significant marketing push, allowing them to capture the traffic we lost." Use sub-headers for each brand to maintain clarity.

    **4. Actionable Strategic Recommendations**
    - Based on the evidence and narrative established in the previous sections, provide 2-4 distinct, actionable recommendations.
    - **For each recommendation, you must:**
        a. **State the recommendation clearly** (e.g., "Launch a targeted campaign to reclaim the Buy Box for key ASINs.").
        b. **Provide the "Why."** Justify it by explicitly referencing a key finding from your analysis (e.g., "...to reverse the critical 100% Buy Box loss, which was the primary driver of our L4 sales decline.").
        c. **Define the intended outcome** (e.g., "The goal is to restore our primary sales channel and defend our market share against growing competitors like Sheba and Tiki Pets.").
        
    """)

# --- 4. Define Graph Nodes ---
def category_health_node(state: AgentState):
    print("\n---PHASE 1: CATEGORY HEALTH CHECK---")
    result = category_agent.invoke(state)
    state['category_health_summary'] = result['output']
    state['messages'].append(HumanMessage(content=result['output'], name="CategoryAgent"))
    return state

def supervisor_planning_node(state: AgentState):
    print("\n---PHASE 2: SUPERVISOR PLANNING---")
    result = supervisor_agent.invoke(state)
    state['supervisor_plan'] = result['output']
    state['messages'].append(HumanMessage(content=result['output'], name="Supervisor"))
    save_markdown("01_supervisor_plan.md", f"# Supervisor Investigation Plan\n\n{result['output']}")
    return state

def brand_analyst_node(state: AgentState):
    print("\n---PHASE 3: BRAND ANALYST INVESTIGATION---")
    # The analyst now receives the full context including the category health and the plan
    result = brand_analyst_agent.invoke(state)
    state['analyst_findings'] = result['output']
    state['messages'].append(HumanMessage(content=result['output'], name="BrandAnalyst"))
    save_markdown("02_analyst_findings.md", f"# Brand Analyst Findings\n\n{result['output']}")
    return state

def supervisor_reporting_node(state: AgentState):
    print("\n---PHASE 4: FINAL REPORTING---")
    final_prompt = f"""Here is all the information gathered for the weekly report:
    
    ## Category Health Summary:
    {state['category_health_summary']}
    
    ## Detailed Analyst Findings:
    {state['analyst_findings']}
    
    Please compile the final report based on these inputs."""
    
    result = final_report_agent.invoke({"messages": [HumanMessage(content=final_prompt)]})
    state['messages'].append(HumanMessage(content=result['output'], name="FinalReportAgent"))
    return state

# --- 5. Build and Run the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("CategoryHealth", category_health_node)
workflow.add_node("SupervisorPlanning", supervisor_planning_node)
workflow.add_node("BrandAnalyst", brand_analyst_node)
workflow.add_node("FinalReporting", supervisor_reporting_node)

workflow.set_entry_point("CategoryHealth")
workflow.add_edge("CategoryHealth", "SupervisorPlanning")
workflow.add_edge("SupervisorPlanning", "BrandAnalyst")
workflow.add_edge("BrandAnalyst", "FinalReporting")
workflow.add_edge("FinalReporting", END)

app = workflow.compile()

if __name__ == '__main__':
    initial_message = {"messages": [HumanMessage(content="Begin the weekly performance analysis.")]}
    final_state = app.invoke(initial_message)
    
    final_report = final_state['messages'][-1].content
    date_str = datetime.now().strftime("%Y-%m-%d")
    final_filename = f"03_final_report_{date_str}.md"
    save_markdown(final_filename, final_report)

    print("\n" + "="*60)
    print("Agentic workflow complete.")
    print(f"Final report saved to: {final_filename}")
    print("="*60 + "\n")
