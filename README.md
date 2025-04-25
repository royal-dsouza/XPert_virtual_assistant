ASL Open Project – XPert AI Agent
Team: Royal, Amit, Mohit, Pushkar

<img width="1053" alt="image" src="https://github.com/user-attachments/assets/451e9f10-c4c4-4cf9-a0ec-e95c794794e0" />


Project Overview
Customer support agents often face inefficiencies and delays due to the need to access multiple systems to answer customer queries. The ASL Open Project addresses this by integrating an AI-powered chat platform, aiming to streamline information retrieval and enhance agent productivity.

Problem Statement & Business Impact
Problem:
Agents spend significant time switching between systems, causing delays and increased workload.

Business Impact:

1. Reduces time spent switching between systems
2. Enables faster resolution times
3. Increases customer satisfaction
4. Lowers operational costs
5. Enhances agent productivity
6. Supports scalability as the business grows

Learning Goals:
Understand how GenAI can improve end-user efficiency by simplifying information retrieval

Dataset and Tasks
Data Used:
Collections, corrections, disputes, claims, and shipment information from Google BigQuery

Tasks Performed:
1. Data is organized at the shipment/PRO level
2. Provided detailed table and column definitions for model context
3. Experimented with various prompts for model flexibility
4. Evaluated different GenAI models (Gemini 1.5 Flash, Gemini 1.5 Pro, Gemini 2.0 Flash) to determine the best fit

Solution Overview
1. Users ask questions in natural language via a chat interface (built with Streamlit)
2. Questions are processed by an LLM (Large Language Model)
3. The LLM references BigQuery table schemas and definitions
4. LLM constructs and executes SQL queries against BigQuery
5. Results are translated back into natural language responses
6. The service is hosted on Google Cloud Run

Model Comparison


Production Roadmap

Short-Term Goals:
1. Provide model with a list of APIs for more accurate, real-time information
2. Include application links in responses for quick user verification
3. Integrate service into CRM or Edge platforms

Long-Term Goals:
1. Develop multiple domain-specific agents, coordinated by a master agent
2. Enable the model to perform actions (e.g., create pickup requests)
3. Deploy the platform directly for customer use


For more information, refer to the presentation: https://docs.google.com/presentation/d/1QkIE_AnnF6bJOU7_pB7n_sCMWX8qnHyh/edit?usp=sharing&ouid=105545888623314265270&rtpof=true&sd=true
