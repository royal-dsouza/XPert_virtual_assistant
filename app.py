import time
import os
from google.cloud import bigquery
import streamlit as st
from typing import Any, Callable, Optional, Tuple, Union
from vertexai.generative_models import FunctionDeclaration, GenerativeModel, Part, Tool, GenerationConfig, GenerationResponse, ChatSession, Content

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/royaldsouza/Downloads/qwiklabs-asl-02-a9f444ba3980-2a66e775b84d.json"  # Update this path

PROJECT = 'qwiklabs-asl-02-a9f444ba3980'

BIGQUERY_DATASET_ID = "SANDBOX"

class ChatAgent:
    def __init__(
        self,
        model: GenerativeModel,
        tool_handler_fn: Callable[[str, dict], Any],
        max_iterative_calls: int = 5,
    ):
        self.tool_handler_fn = tool_handler_fn
        self.chat_session = model.start_chat()
        self.max_iterative_calls = 5

    def send_message(self, message: str) -> GenerationResponse:
        response = self.chat_session.send_message(message)

        # This is None if a function call was not triggered
        fn_call = response.candidates[0].content.parts[0].function_call

        num_calls = 0
        # Reasoning loop. If fn_call is None then we never enter this
        # and simply return the response
        while fn_call:
            if num_calls > self.max_iterative_calls:
                break

            # Handle the function call
            fn_call_response = self.tool_handler_fn(fn_call.name, dict(fn_call.args))
            num_calls += 1

            # Send the function call result back to the model
            response = self.chat_session.send_message(
                Part.from_function_response(
                    name=fn_call.name,
                    response={
                        "content": fn_call_response,
                    },
                ),
            )

            # If the response is another function call then we want to
            # stay in the reasoning loop and keep calling functions.
            fn_call = response.candidates[0].content.parts[0].function_call

        return response

list_available_tables_func = FunctionDeclaration(
    name="list_available_tables",
    description="Get the list all available BigQuery tables with fully qualified IDs.",
    parameters={"type": "object", "properties": {}},
)

get_table_info_func = FunctionDeclaration(
    name="get_table_info",
    description="Get information about a BigQuery table and it's schema so you can better answer user questions.",
    parameters={
        "type": "object",
        "properties": {
            "table_id": {
                "type": "string",
                "description": "Fully qualified ID of BigQuery table",
            }
        },
    },
)

sql_query_func = FunctionDeclaration(
    name="sql_query",
    description="Get information from data in BigQuery using SQL queries",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": f"SQL query on a single line (no \\n characters) that will help give answers to users questions when run on BigQuery. Only query tables in project: {PROJECT}",
            }
        },
        "required": [
            "query",
        ],
    },
)

sql_query_tool = Tool(
    function_declarations=[
        list_available_tables_func,
        get_table_info_func,
        sql_query_func
    ],
)

model = GenerativeModel(
    "gemini-2.0-flash-exp",
    generation_config=GenerationConfig(temperature=0.0),
    tools=[sql_query_tool],
)

def list_available_tables():
    return [f"{PROJECT}.SANDBOX.CUSTOMER_AR_DATA",f"{PROJECT}.SANDBOX.CLAIMS_DATA"]


def get_table_info(table_id: str) -> dict:
    """Returns dict from BigQuery API with table information"""
    bq_client = bigquery.Client()
    return bq_client.get_table(table_id).to_api_repr()


def sql_query(query_str: str):
    bq_client = bigquery.Client()
    try:
        # clean up query string a bit
        query_str = (
            query_str.replace("\\n", "").replace("\n", "").replace("\\", "")
        )
        # print(query_str)
        query_job = bq_client.query(query_str)
        result = query_job.result()
        result = str([dict(x) for x in result])
        return result
    except Exception as e:
        return f"Error from BigQuery Query API: {str(e)}"
    
def handle_query_fn_call(fn_name: str, fn_args: dict):
    """Handles query tool function calls."""

    print(f"Function calling: {fn_name} with args: {str(fn_args)}\n")
    if fn_name == "list_available_tables":
        result = list_available_tables()

    elif fn_name == "get_table_info":
        result = get_table_info(fn_args["table_id"])

    elif fn_name == "sql_query":
        result = sql_query(fn_args["query"])

    else:
        raise ValueError(f"Unknown function call: {fn_name}")

    return result

st.set_page_config(
    page_title="XPert Virtal Agent",
    page_icon="XPO_logo.svg",
    layout="wide",
)

col1, col2 = st.columns([8, 1])
with col1:
    st.title("XPert Virtal Agent")
with col2:
    st.image("XPO_logo.svg")

with st.expander("Sample prompts", expanded=True):
    st.write(
        """
        - What kind of information is in this database?
        - What percentage of orders are returned?
        - How is inventory distributed across our regional distribution centers?
        - Do customers typically place more than one order?
        - Which product categories have the highest profit margins?
    """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", r"\$"))  # noqa: W605

if prompt := st.chat_input("Ask me about information on claims, disputes, correction, invoice, collection and shipment tracking..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        chat = ChatAgent(model=model, tool_handler_fn=handle_query_fn_call)

        init_prompt = """
            Please give a concise and easy to understand answer to any questions.
            Only use information that you learn by querying the BigQuery table.
            Do not make up information. Be sure to look at which tables are available
            and get the info of any relevant tables before trying to write a query.
            
            Question:
            """

        try:
            response = chat.send_message(init_prompt + prompt)

            print(response.text)

            full_response = response.text
            with message_placeholder.container():
                st.markdown(full_response.replace("$", r"\$"))  # noqa: W605

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response
                }
            )
        except Exception as e:
            print(e)
            error_message = f"""
                Something went wrong! We encountered an unexpected error while
                trying to process your request. Please try rephrasing your
                question. Details:

                {str(e)}"""
            st.error(error_message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )
