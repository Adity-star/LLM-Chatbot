import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
import logging
import re

logging.basicConfig(level=logging.DEBUG)

# Function to format the output in a more user-friendly way
def format_output(text):
    """Convert Markdown bold syntax to HTML strong tags."""
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

# Function to initialize the chatbot pipeline
def ask_llama2():
    try:
        # Define the prompt template for the chat
        create_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', 'Hello! How can I help you today?'),
                ('user', 'question: {question}')
            ]
        )
        
        # Initialize the Ollama model using the updated class
        llama_model = OllamaLLM(model='llama2')

        # Langchain allows composing these steps, so use the proper pipeline structure
        chatbot_pipeline = create_prompt | llama_model

        return chatbot_pipeline

    except Exception as e:
        logging.error(f"Failed to initialize chatbot: {e}")
        raise

# Initialize chatbot pipeline
chatbot_pipeline = ask_llama2()

# Streamlit app logic
def main():

    # Streamlit user interface
    st.set_page_config(page_title="Chat with Llama2", page_icon=":robot_face:", layout="centered")
    

    # Add the logo 
    st.markdown("""
        <div style="text-align:center;">
            <img src = "https://th.bing.com/th/id/OIP.zEno2jdHhS8EfqUxqGrPKAHaHa?w=157&h=180&c=7&pcl=1b1a19&r=0&o=5&dpr=1.3&pid=1.7" />
        </div>
    """, unsafe_allow_html=True)

    # Add  header
    st.markdown("""
        <h1 style="text-align:center; color: #4CAF50;">Welcome to Llama2 Chatbot</h1>
        <p style="text-align:center; font-size:18px;">Ask me anything and I'll do my best to help!</p>
    """, unsafe_allow_html=True)

    # Create a chat-like input/output section
    query_input = st.text_input("Ask a question:", placeholder="Enter your question here...", key="input_field")


    # Check if the button is clicked 
    if st.button("Send"):
        if query_input:
            try:

                # Provide the input to the model and get the response
                response = chatbot_pipeline.invoke({'question': query_input})
                
                # Format the output and display it in a more attractive way
                #formatsted_output = format_output(response)
                #st.markdown(f"<div style='padding: 10px; background-color: #f1f1f1; border-radius: 5px;'>{formatted_output}</div>", unsafe_allow_html=True)

                # Display the conversation
                st.write(f"**You asked:** {query_input}")
                st.write(f"**Llama2 says:** {response}")
                
            except Exception as e:
                logging.error(f"Error during chatbot invocation: {e}")
                st.error("Sorry, an error occurred while processing your request.")
        else:
            st.warning("Please enter a question before clicking 'Send'.")

    # Additional user-friendly touch: A footer for closing
    st.markdown("""
        <hr>
        <footer style="text-align:center;">
            <p style="color: #888; font-size: 14px;">Powered by Llama2 | Aditya Ak </p>
        </footer>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
