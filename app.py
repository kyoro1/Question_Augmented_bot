from src import common
import streamlit as st
import pandas as pd

# 0. Several configuration
## Initialization of the tools
tools = common.AOAI_TOOLS(config_file='./config.yml')

## Set the client
tools.setClient()
conversation_history = []

## Load the data
df_all = pd.read_pickle(tools.AUGMENTED_QA)

## Load the prompts
tools.load_prompts(prompt_name='QA_prompt_template', 
                    prompt_path=tools.OPERATIONAL_PROMPTS)
prompt_template = tools.promptBank['QA_prompt_template']['PROMPTS']


#0. 

with st.sidebar:
    st.title('Mode')

    mode = st.radio(
    "Select the modes",
    ["Just run", "debug"],
    captions = ["Just run", 'Show the probability, used prompt'])

# 1. Manage conversation
## Set the title
st.title("Microsoft FAQ bot (unofficial)")

## Set the description
if "messages" not in st.session_state:
    ## Initialize the chat history
    st.session_state["messages"] = []

## Display the chat history
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

## Chat input
if input := st.chat_input("Type your requests."):
    query_result = tools.judge_if_KB(INPUT=input,
                            df_all=df_all)

    ## Display the user input
    with st.chat_message("user"):
        st.markdown(input)

    ## Select the prompt to be used
    prompt = tools.set_prompt(prompt_template=prompt_template,
                            query_result=query_result,
                            threshold=tools.CONFIDENCE_COSINE_SIMILARITY)
    
    print(prompt)

    ## Generate the response
    conversation_history, returned_message = tools.manualConversation(INPUT=input,
                                            prompt=prompt,
                                            conversation_history=st.session_state.messages,)

    print('Chat History: ', conversation_history)

    ## Display the response
    with st.chat_message("assistant"):
        st.markdown(returned_message)
        if mode == 'debug':
            st.write(f"Cosine Similarity: {query_result['similarity']}")
            st.write("Used prompt:")
            st.code(prompt)