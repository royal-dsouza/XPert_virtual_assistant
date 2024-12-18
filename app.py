import time
import os
from google.cloud import bigquery
from google.oauth2 import service_account
import streamlit as st
from typing import Any, Callable, Optional, Tuple, Union
from vertexai.generative_models import FunctionDeclaration, GenerativeModel, Part, Tool, GenerationConfig, GenerationResponse, ChatSession, Content, HarmCategory, HarmBlockThreshold
import google.auth
import json
import re

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\Pushkar\Python\Qwiklab_Credentials.json"  # Update this path

PROJECT = 'qwiklabs-asl-02-a9f444ba3980'

BIGQUERY_DATASET_ID = "SANDBOX"

# credentials = service_account.Credentials.from_service_account_file('Qwiklab_Credentials.json')
credentials, project = google.auth.default(
    scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/bigquery"
        ]
    )


class ChatAgent:
    def __init__(
        self,
        model: GenerativeModel,
        tool_handler_fn: Callable[[str, dict], Any],
        max_iterative_calls: int = 5,
        chat_history: list[Content] = []
    ):
        self.tool_handler_fn = tool_handler_fn
        self.chat_session = model.start_chat(history = chat_history)
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

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

model = GenerativeModel(
    "gemini-1.5-pro",
    generation_config=GenerationConfig(temperature=0.0),
    tools=[sql_query_tool],
    safety_settings=safety_settings
)

def list_available_tables():
    return [f"{PROJECT}.SANDBOX.CUSTOMER_COLLECTIONS_DATA",
        f"{PROJECT}.SANDBOX.CLAIMS_DATA",
        f"{PROJECT}.SANDBOX.CUSTOMER_CORRECTIONS_DATA",
        f"{PROJECT}.SANDBOX.CUSTOMER_DISPUTES_DATA",
        f"{PROJECT}.SANDBOX.SHIPMENT_DATA"]


def get_table_info(table_id: str) -> dict:
    """Returns dict from BigQuery API with table information"""
    bq_client = bigquery.Client(project=project, credentials=credentials)
    return bq_client.get_table(table_id).to_api_repr()


def sql_query(query_str: str):
    bq_client = bigquery.Client(project=project, credentials=credentials)
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

def validate_pro_number(s):
    # Define the regular expression pattern
    pattern = r'^(0\d{3}0\d{6}|(\d{3}-\d{6})|\d{9})$'
    
    # Match the pattern
    if re.match(pattern, s):
        return True
    else:
        return False
    
def validate_madcode(s):
    # Check if the string starts with a letter, ends with a number, and contains no special characters
    return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9 ]*\d+$', s))

def format_pro_number(s):
    # Clean the input string by hyphens
    cleaned_number = s.replace("-","")
    print(cleaned_number)
    
    # Check the length and format of the cleaned number
    if len(cleaned_number) == 9:
        # Add leading zero to convert to 0XXX0XXXXXX format
        formatted_number = '0' + cleaned_number[:3] + '0' + cleaned_number[3:]
        return f"{formatted_number} or PRO NUMBER = {cleaned_number}"
    
    elif len(cleaned_number) == 11 and s[0] == '0':
        # remove the zero to convert to xxxxxxxxx
        formatted_number = cleaned_number[1:4] + cleaned_number[5:]
        return f"{cleaned_number} or PRO NUMBER = {formatted_number}"

st.set_page_config(
    page_title="XPert AI Agent",
    page_icon="XPO_logo.svg",
    layout="wide",
)

col1, col2 = st.columns([8, 1])
with col1:
    st.title("XPert AI Agent")
with col2:
    st.image("XPO_logo.svg")

st.markdown("""
        <style>
        .input-label {
            font-weight: bold;
            color: #FF6347;  /* Red label */
            font-size: 18px;
        }
        </style>
    """, unsafe_allow_html=True)

# Add three small text input boxes on the same line
col1, col2 = st.columns([6, 6])
with col1:
    st.markdown('<label class="input-label">Shipment Tracking Number / PRO Number</label>', unsafe_allow_html=True)
    shipment_tracking_number = st.text_input(label="shipment_tracking_number",key="shipment_tracking_number", placeholder="Enter Shipment Tracking Number / PRO Number", help="Enter the tracking number / PRO Number of the shipment.", label_visibility="collapsed")
        
with col2:
    st.markdown('<label class="input-label">Customer Reference Number / Customer MADCODE</label>', unsafe_allow_html=True)
    customer_reference_number = st.text_input(label="customer_reference_number",key="customer_reference_number", placeholder="Enter Customer Ref. No. / Customer MADCODE", help="Enter the customer reference number / Customer MADCODE.", label_visibility="collapsed")

if not shipment_tracking_number and not customer_reference_number:
    st.error("Please enter at least one of the following: Shipment Tracking Number / PRO Number or Customer Reference Number / Customer MADCODE to answer you better.")

with st.expander("Sample prompts", expanded=True):
    st.write(
        """
        - Where is the shipment?
        - Are there any open invoices on this shipment?
        - Can you provide claims information on this pro?
        - What is the status of the claim?
        - When this shipment will be delivered?
    """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", r"\$"))  # noqa: W605

if prompt := st.chat_input("Ask me about information on claims, disputes, correction, invoice, collection and shipment tracking..."):
    if not shipment_tracking_number and not customer_reference_number:
        st.error("Please enter at least one of the following: Shipment Tracking Number / PRO Number or Customer Reference Number / Customer MADCODE to answer you better.")
    if shipment_tracking_number and not validate_pro_number(shipment_tracking_number):
        st.error("Entered Shipment tracking no. / PRO Number is not in the valid format")
    if customer_reference_number and not validate_madcode(customer_reference_number):
        st.error("Entered Customer Reference Number / Customer MADCODE is not in the valid format")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # print(st.session_state.messages)
            chat = ChatAgent(model=model, tool_handler_fn=handle_query_fn_call, chat_history = st.session_state.history)

            init_prompt = f"""
                Please give a concise and easy to understand answer to any questions.
                Only use information that you learn by querying the BigQuery table.
                Do not make up information. Be sure to look at which tables are available
                and get the info of any relevant tables before trying to write a query.
                
                When providing dollar amounts or anything related to money please format it in terms of USD.
                When the user mentions the word PRO search for PRO_NUMBER or PRO_NBR_TXT.

                PRO NUMBER = {format_pro_number(shipment_tracking_number)}
                MADCODE = {(customer_reference_number).upper()}
                
                If PRO NUMBER is provided then always use the PRO NUMBER on the where condition of the sql to filter by pro number.
                If MADCODE is provided then always use the MADCODE on the where condition of the sql to filter by customer madcode or debtor madcode

                Question:
                """

            try:
                response = chat.send_message(init_prompt + prompt)
                st.session_state.history += chat.chat_session.history

                # print("History: ", st.session_state.history)

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
