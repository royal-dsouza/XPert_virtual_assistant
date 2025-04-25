# XPert AI Agent - Streamlining Customer Support with GenAI

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-blue)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-Hosting-blue)
![BigQuery](https://img.shields.io/badge/BigQuery-Data_Source-blue)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/royal-dsouza/XPert_virtual_assistant/graphs/commit-activity)

## Overview

The **XPert AI Agent** is an innovative solution designed to enhance customer support efficiency by leveraging the power of Generative AI. This project addresses the common problem of customer support agents needing to access multiple systems to answer customer queries, leading to delays and increased workload. By integrating an AI-powered chat platform, XPert AI Agent streamlines information retrieval, reduces resolution times, and ultimately boosts customer satisfaction.

<img width="1128" alt="image" src="https://github.com/user-attachments/assets/76d96b01-6839-496b-8505-366a67eaf62e" />


## Key Features

-   **Natural Language Chat Interface:** Users can interact with the agent using natural language, making it easy to ask questions and get quick answers.
-   **GenAI-Powered Information Retrieval:** Employs advanced GenAI models (Gemini 1.5 Flash, Gemini 1.5 Pro, Gemini 2.0 Flash) to understand queries and retrieve relevant information from Google BigQuery.
-   **SQL Query Generation:** Automatically translates natural language questions into SQL queries, executes them against BigQuery, and presents the results in an easy-to-understand format.
-   **Integration with Google Cloud:** Hosted on Google Cloud Run for scalability and reliability.
-   **Real-time Data Access:** Connects to Google BigQuery to provide up-to-date information on collections, corrections, disputes, claims, and shipments.

## Business Impact

-   Reduces time spent switching between systems
-   Enables faster resolution times
-   Increases customer satisfaction
-   Lowers operational costs
-   Enhances agent productivity
-   Supports scalability as the business grows

## Solution Architecture

The XPert AI Agent follows this architecture:

1.  Users interact with the system via a chat interface built with Streamlit.
2.  User questions are processed by a Large Language Model (LLM).
3.  The LLM references BigQuery table schemas and definitions to understand the data structure.
4.  The LLM constructs and executes SQL queries against BigQuery.
5.  Results are translated back into natural language responses and displayed to the user.

![System Flow](https://cdn.mathpix.com/cropped/2025_04_25_ea9b4771e273da2eb4c9g-4.jpg?height=562&width=2112&top_left_y=395&top_left_x=127)

## XPert AI Agent Interface

Here's a look at the XPert AI Agent interface:

![XPert AI Agent Interface](https://pplx-res.cloudinary.com/image/private/user_uploads/HUGUnIYwxBvyaVt/image.jpg)

## Model Comparison

| Model                    | Answer Quality                                                                 | % Correct Answers | Avg. Response Time | Summary                                                                                     |
| :----------------------- | :----------------------------------------------------------------------------- | :------------------ | :------------------- | :------------------------------------------------------------------------------------------ |
| `gemini-1.5-flash`       | To the point, fast, sometimes struggles with SQL, occasional 400 errors       | 90%               | 3.8 secs            | Great speed, good accuracy, frequent 400 errors                                             |
| `gemini-1.5-pro`         | Additional explanations, great accuracy, fewer 400 errors                     | 95%               | 8.6 secs            | Good speed, great accuracy, fewer errors                                                    |
| `gemini-2.0-flash-exp`   | Better SQL, best answers, but often quota exceeded (429 errors)               | 98%               | 3.25 secs           | Excellent speed/accuracy, frequent 429 errors, latest model with high usage                 |

## Getting Started

### Prerequisites

-   Google Cloud Account with access to BigQuery and Cloud Run
-   Python 3.6+
-   Streamlit
-   Access to the required datasets in Google BigQuery

### Installation

1.  Clone the repository:

    ```
    git clone https://github.com/royal-dsouza/XPert_virtual_assistant.git
    cd XPert_virtual_assistant
    ```

2.  Install the required Python packages:

    ```
    pip install -r requirements.txt
    ```

3.  Configure your Google Cloud credentials.

4.  Set up your BigQuery connection.

5.  Deploy the Streamlit app to Google Cloud Run.

### Usage

1.  Run the Streamlit application:

    ```
    streamlit run your_app_name.py
    ```

2.  Access the application through your web browser.

3.  Start asking questions related to collections, corrections, disputes, claims, and shipments.

## Production Roadmap

### Short-Term Goals

-   Provide the model with a list of APIs for more accurate, real-time information.
-   Include application links in responses for quick user verification.
-   Integrate the service into CRM or Edge platforms.

### Long-Term Goals

-   Develop multiple domain-specific agents, coordinated by a master agent.
-   Enable the model to perform actions (e.g., create pickup requests).
-   Deploy the platform directly for customer use.

## Team

-   Royal
-   Amit
-   Mohit
-   Pushkar

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

-   This project was inspired by the need to improve customer support efficiency and reduce agent workload.
-   We thank Google Cloud for providing the tools and resources to build and deploy this solution.

