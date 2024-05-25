import streamlit as st
from dotenv import load_dotenv
from texttosql import create_llm, call_openai_model
import os
import asyncio

load_dotenv()

def main():
    st.title("SQL Chatbot")
    
    #Initialize necessary components
    client = asyncio.run(create_llm())
    prompt = open(file="system.txt", encoding="utf8").read().strip()
    
    
    # Get configuration settings 
    azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
    azure_oai_key = os.getenv("AZURE_OAI_KEY")
    azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
    
    
    # Get or create message history from session state
    if 'message_history' not in st.session_state:
        st.session_state.message_history = []

    # Unique keys for widgets
    input_key = "input_query"
    submit_button_key = "submit_button"
    exit_button_key = "exit_button"

    # Input field for user query
    query = st.text_input("Enter your query:", key=input_key)
    
    # Display submit button and exit button side by side
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Submit", key=submit_button_key):
            if query:
                print("\nSending request to Azure OpenAI model...\n")

                # Process the query and store message/response in history
                response = asyncio.run(call_openai_model(prompt, query, azure_oai_deployment, client))
                if response:
                    print("\got response from Azure OpenAI model...\n")
                    #print(response.choices())
                
                    print("The summarized text is:")
                    response_message = response.choices[0].message.content
                    print(response_message)
                    my_code = response_message.strip()
                    st.code(my_code)

                #print("Response:\n" + response.choices[0].message.content + "\n")
                st.session_state.message_history.append((query, my_code))
                #st.write(f"{response}")
    with col2:
        if st.button("Exit", key=exit_button_key):
            st.stop()
            
    # Display message history
    st.subheader("Message History")
    for i, (msg, resp) in enumerate(reversed(st.session_state.message_history[-5:]), 1):
        st.write(f"{i}. **User:** {msg}")
        st.write(f"   **Bot:** {resp}")
           
            
if __name__ == "__main__":
    main()