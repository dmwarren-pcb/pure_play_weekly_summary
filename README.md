# Agentic Sales & Share Analysis System

This project implements a multi-agent system using LangGraph to perform a sophisticated, automated weekly business review of retail sales data, specifically tailored for Amazon performance analysis. The system mimics an organizational hierarchy, with a Supervisor agent planning the analysis and a specialist Analyst agent performing deep dives to uncover the root causes of performance changes.

The final output is a comprehensive, human-readable report in Markdown, detailing performance trends, causal factors, market share shifts, and competitor activity.

---

## Key Features

This system goes beyond simple sales reporting by incorporating advanced analytical concepts to generate strategic insights:

-   **Multi-Period Analysis:** Simultaneously analyzes performance across Latest Week (LW), Latest 4 Weeks (L4), Latest 13 Weeks (L13), and Latest 26 Weeks (L26) against both previous periods and year-ago periods.
-   **Price-Volume Decomposition:** For any change in sales, the system automatically calculates the dollar impact of selling more/fewer units (Volume Effect) versus changes in average selling price (Price Effect).
-   **Root Cause Analysis:** Automatically summarizes changes in key causal metrics (Total Traffic, Paid Ad Spend, Buy Box Rate, In-Stock Rate) to explain *why* Price or Volume effects occurred.
-   **Market Share & Contribution:** Calculates dollar market share and share point changes. It also identifies which brands and competitors are the most significant contributors to overall category growth or decline.
-   **Contextual "Lapping" Analysis:** The system is aware of historical performance and will flag when current trends are influenced by lapping a prior period of unusually high or low sales (e.g., a promotion).
-   **Automated Markdown Reporting:** The entire workflow culminates in a set of clear, well-formatted Markdown files that document the agent's plan, findings, and the final executive-ready report.

---

## Architecture

The system is built on LangGraph and uses a supervisor-analyst agent hierarchy to manage the workflow. All mathematical calculations and data processing are handled robustly in Python using `pandas`, ensuring the LLM is only used for reasoning, planning, and summarization.

1.  **`brand_analysis_tools.py`**: This is the data analysis engine of the project. It contains several `@tool`-decorated functions that prepare a clean, aggregated dataset and provide pre-calculated heuristics (like Price-Volume effects) to the agents. This ensures the LLM never has to perform raw mathematical calculations.

2.  **`langraph_agentic_app.py`**: This file orchestrates the agentic workflow.
    -   **`CategoryHealth_Agent`**: The first agent in the chain, providing a high-level overview of the market.
    -   **`Supervisor_Agent`**: Receives the market overview and uses a tool to get a summary of all brand and competitor performance. It then formulates a detailed, step-by-step investigation plan.
    -   **`BrandAnalyst_Agent`**: Receives the plan and executes it task by task. It uses diagnostic tools to perform deep dives on focus brands and key competitors, synthesizing the results into a detailed narrative.
    -   **`FinalReport_Agent`**: Gathers all the information from the previous steps and compiles the final, structured report.

---

## How to Use

### 1. Prerequisites
- Python 3.9+
- An Azure OpenAI account with a deployed model (e.g., GPT-4).

### 2. Setup
- **Clone the repository:**
  ```bash
  git clone <your-repo-url>
  cd <your-repo-name>
Install dependencies:

Bash

pip install -r requirements.txt
(You will need to create a requirements.txt file containing pandas, numpy, langchain, langgraph, langchain-openai, and python-dotenv)

Configure Environment Variables:
Create a file named .env in the root of the project and add your Azure OpenAI credentials:

AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_API_VERSION="2023-07-01-preview"
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="YOUR_DEPLOYMENT_NAME"
Prepare Data:
Place your sales data file in the root directory and ensure it is named stackline_sales.csv.

Configure Analysis Scope:
Open brand_analysis_tools.py and modify the constants at the top of the file to match your Brand Owner and Supercategory focus.

3. Execution
Run the main application from your terminal:

Bash

python langraph_agentic_app.py
The script will execute the entire agentic workflow, printing the status of each phase to the console.

Output
Upon successful completion, the script will generate three Markdown files in your project directory:

01_supervisor_plan.md: A list of the specific investigation tasks the Supervisor delegated to the Analyst.
02_analyst_findings.md: The detailed, step-by-step findings from the Analyst's investigation.
03_final_report_YYYY-MM-DD.md: The final, executive-ready report combining all analyses into a cohesive narrative with strategic recommendations.