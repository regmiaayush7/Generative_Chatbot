# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_core.output_parsers import StrOutputParser

# from dotenv import load_dotenv
# import streamlit as st

# import os
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Build prompt
# template = """You are a helpful assistant. Please respond to the user queries.
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# # Streamlit framework
# st.title('LangChain Demo with Google Gemini API')
# input_text = st.text_input("Ask me anything!")

# # Google Gemini LLM
# llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
# output_parser = StrOutputParser()

# # Create a simple LLM chain without retrieval
# qa_chain = LLMChain(
#     llm=llm,
#     prompt=QA_CHAIN_PROMPT,
#     output_parser=output_parser,
# )

# # Handle input
# # Handle input
# if input_text:
#     response = qa_chain.invoke({'question': input_text})
    
#     # Formatting the output to be more presentable
#     formatted_response = f"**Question:** {response.get('question')}\n\n**Answer:**\n\n{response.get('text')}"
    
#     # Displaying the response
#     st.markdown(formatted_response)




############################################################
#Start of pdf 
# Import necessary libraries
# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import ConversationalRetrievalChain, LLMChain
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.memory import ConversationBufferMemory
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_core.output_parsers import StrOutputParser

# from dotenv import load_dotenv
# import streamlit as st
# import os
# import tempfile

# # Load environment variables and configure Google Gemini API
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Set up Streamlit app
# st.title('LangChain Demo with Google Gemini API')
# input_text = st.text_input("Ask me anything!")

# # PDF file uploader
# uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# # Define the output parser globally so it can be used in both conditions
# output_parser = StrOutputParser()

# # Check if a PDF is uploaded
# if uploaded_file:
#     # Create a temporary file to save the uploaded PDF
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         temp_file.write(uploaded_file.read())
#         temp_file_path = temp_file.name

#     # Load and split the PDF document into smaller chunks
#     loader = PyPDFLoader(temp_file_path)
#     documents = loader.load()

#     # Split documents using a text splitter
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1500,
#         chunk_overlap=150
#     )
#     splits = text_splitter.split_documents(documents)

#     # Correct instantiation of GoogleGenerativeAIEmbeddings with a specified model
#     embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     persist_directory = 'docs/chroma'
#     vectordb = Chroma(persist_directory=persist_directory, 
#                       embedding_function=embedding)

#     question = input_text
#     # Similarity search according to embeddings
#     docs = vectordb.similarity_search(question, k=6)

#     # Define a prompt template for general questions
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """
#     QA_CHAIN_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#     # Google Gemini LLM and memory chain
#     llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     memory = ConversationBufferMemory(
#         memory_key="chat_history", output_key="answer", return_messages=True
#     )
    
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm,
#         vectordb.as_retriever(search_kwargs={"k": 6}),
#         return_source_documents=True,
#         memory=memory,
#         verbose=False,
#         combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
#     )

#     result = qa_chain.invoke({"question": question})
#     LLM_result = result["answer"]
#     st.write(LLM_result)
    
# else:
#     # Build prompt
#     template = """You are a helpful assistant. Please respond to the user queries.
#     Question: {question}
#     Helpful Answer:"""
#     QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

#     # Google Gemini LLM
#     llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

#     # Create a simple LLM chain without retrieval
#     qa_chain = LLMChain(
#         llm=llm,
#         prompt=QA_CHAIN_PROMPT,
#         output_parser=output_parser,
#     )

# # Handle input
# if input_text and not uploaded_file:  # Ensuring this block doesn't run if a PDF has been uploaded
#     response = qa_chain.invoke({'question': input_text})
    
#     # Formatting the output to be more presentable
#     formatted_response = f"**Question:** {response.get('question')}\n\n**Answer:**\n\n{response.get('text')}"
    
#     # Displaying the response
#     st.markdown(formatted_response)



# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import ConversationalRetrievalChain, LLMChain
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.memory import ConversationBufferMemory
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_core.output_parsers import StrOutputParser

# from dotenv import load_dotenv
# import streamlit as st
# import os
# import tempfile

# # Load environment variables and configure Google Gemini API
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Set up Streamlit app
# st.title('LangChain Demo with Google Gemini API')

# # PDF file uploader
# uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# # Define the output parser globally so it can be used in both conditions
# output_parser = StrOutputParser()

# # Check if a PDF is uploaded
# if uploaded_file:
#     input_text = st.text_input("Ask me anything!")
#     # Create a temporary file to save the uploaded PDF
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         temp_file.write(uploaded_file.read())
#         temp_file_path = temp_file.name

#     # Load and split the PDF document into smaller chunks
#     loader = PyPDFLoader(temp_file_path)
#     documents = loader.load()

#     # Split documents using a text splitter
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1500,
#         chunk_overlap=150
#     )
#     splits = text_splitter.split_documents(documents)

#     # Debug: Display the first few splits to check context
#     st.write("Extracted Document Chunks:", splits[:3])

#     # Correct instantiation of GoogleGenerativeAIEmbeddings with a specified model
#     embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     persist_directory = 'docs/chroma'
#     vectordb = Chroma(persist_directory=persist_directory, 
#                       embedding_function=embedding)

#     question = input_text

#     # Perform similarity search according to embeddings
#     docs = vectordb.similarity_search(question, k=6)

#     # Debug: Display retrieved documents to verify correct context
#     st.write("Retrieved Context:", docs[:3])

#     # Combine all retrieved docs into a single context string
#     context = "\n\n".join([doc.page_content for doc in docs])

#     # Define a prompt template for general questions with improved instructions
#     prompt_template = """
#     Answer the question as detailed as possible based on the provided context.\n
#     Context:\n{context}\n
#     Question:\n{question}\n
#     If the context does not provide enough information, respond with 'The context does not provide the information needed to answer the question.'\n
#     Answer:
#     """
#     QA_CHAIN_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#     # Google Gemini LLM and memory chain
#     llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     memory = ConversationBufferMemory(
#         memory_key="chat_history", output_key="answer", return_messages=True
#     )

#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm,
#         vectordb.as_retriever(search_kwargs={"k": 6}),
#         return_source_documents=True,
#         memory=memory,
#         verbose=False,
#         combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
#     )

#     # Check if the context is empty or insufficient
#     if not context.strip():
#         st.write("The retrieved context is empty or irrelevant. Please try asking a different question.")
#     else:
#         result = qa_chain.invoke({"context": context, "question": question})
#         LLM_result = result["answer"]
#         st.write(LLM_result)
    
# else:
#     # Build prompt for questions without PDF context
#     input_text = st.text_input("Ask me anything!")
#     template = """You are a helpful assistant. Please respond to the user queries.
#     Question: {question}
#     Helpful Answer:"""
#     QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

#     # Google Gemini LLM
#     llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

#     # Create a simple LLM chain without retrieval
#     qa_chain = LLMChain(
#         llm=llm,
#         prompt=QA_CHAIN_PROMPT,
#         output_parser=output_parser,
#     )

# # Handle input when no PDF is uploaded
# if input_text and not uploaded_file:  
#     response = qa_chain.invoke({'question': input_text})
    
#     # Formatting the output to be more presentable
#     formatted_response = f"**Question:** {response.get('question')}\n\n**Answer:**\n\n{response.get('text')}"
    
#     # Displaying the response
#     st.markdown(formatted_response)


# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import ConversationalRetrievalChain, LLMChain
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.memory import ConversationBufferMemory
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_core.output_parsers import StrOutputParser

# from dotenv import load_dotenv
# import streamlit as st
# import os
# import tempfile

# # Load environment variables and configure Google Gemini API
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Set up Streamlit app
# st.title('LangChain Demo with Google Gemini API')
# input_text = st.text_input("Ask me anything!")

# # PDF file uploader
# uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# # Define the output parser globally so it can be used in both conditions
# output_parser = StrOutputParser()

# # Check if a PDF is uploaded
# if uploaded_file:
#     # Create a temporary file to save the uploaded PDF
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         temp_file.write(uploaded_file.read())
#         temp_file_path = temp_file.name

#     # Load and split the PDF document into smaller chunks
#     loader = PyPDFLoader(temp_file_path)
#     documents = loader.load()

#     # Split documents using a text splitter
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1500,
#         chunk_overlap=150
#     )
#     splits = text_splitter.split_documents(documents)

#     # Debug: Display the first few splits to check context
#     st.write("Extracted Document Chunks:", splits[:3])

#     # Correct instantiation of GoogleGenerativeAIEmbeddings with a specified model
#     embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     persist_directory = 'docs/chroma'
#     vectordb = Chroma(persist_directory=persist_directory, 
#                       embedding_function=embedding)

#     question = input_text

#     # Perform similarity search according to embeddings
#     docs = vectordb.similarity_search(question, k=6)

#     # Debug: Display retrieved documents to verify correct context
#     st.write("Retrieved Context Documents:", docs[:3])

#     # Combine all retrieved docs into a single context string
#     context = "\n\n".join([doc.page_content for doc in docs])

#     # Check if context is retrieved correctly
#     if not context.strip():
#         st.write("The retrieved context is empty or irrelevant. Please try asking a different question.")
#     else:
#         # Define a prompt template for general questions with improved instructions
#         prompt_template = """
#         Answer the question as detailed as possible based on the provided context. 
#         If the context is not sufficient to answer the question, indicate this clearly.
#         \n
#         Context:\n{context}\n
#         Question:\n{question}\n
#         Answer:
#         """
#         QA_CHAIN_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#         # Google Gemini LLM and memory chain
#         llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#         memory = ConversationBufferMemory(
#             memory_key="chat_history", output_key="answer", return_messages=True
#         )

#         qa_chain = ConversationalRetrievalChain.from_llm(
#             llm,
#             vectordb.as_retriever(search_kwargs={"k": 6}),
#             return_source_documents=True,
#             memory=memory,
#             verbose=False,
#             combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
#         )

#         # Debug: Display the combined context that is sent to LLM
#         st.write("Combined Context Sent to LLM:", context)

#         result = qa_chain.invoke({"context": context, "question": question})
#         LLM_result = result["answer"]
#         st.write(LLM_result)

# else:
#     # Build prompt for questions without PDF context
#     template = """You are a helpful assistant. Please respond to the user queries.
#     Question: {question}
#     Helpful Answer:"""
#     QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

#     # Google Gemini LLM
#     llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

#     # Create a simple LLM chain without retrieval
#     qa_chain = LLMChain(
#         llm=llm,
#         prompt=QA_CHAIN_PROMPT,
#         output_parser=output_parser,
#     )

# # Handle input when no PDF is uploaded
# if input_text and not uploaded_file:  
#     response = qa_chain.invoke({'question': input_text})
    
#     # Formatting the output to be more presentable
#     formatted_response = f"**Question:** {response.get('question')}\n\n**Answer:**\n\n{response.get('text')}"
    
#     # Displaying the response
#     st.markdown(formatted_response)

# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import ConversationalRetrievalChain, LLMChain
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.memory import ConversationBufferMemory
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_core.output_parsers import StrOutputParser

# from dotenv import load_dotenv
# import streamlit as st
# import os
# import tempfile

# # Load environment variables and configure Google Gemini API
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Set up Streamlit app
# st.title('LangChain Demo with Google Gemini API')
# input_text = st.text_input("Ask me anything!")

# # PDF file uploader
# uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# # Define the output parser globally so it can be used in both conditions
# output_parser = StrOutputParser()

# # Initialize variables to hold document chunks and context
# context = ""
# docs = []

# # Check if a PDF is uploaded
# if uploaded_file:
#     # Create a temporary file to save the uploaded PDF
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         temp_file.write(uploaded_file.read())
#         temp_file_path = temp_file.name

#     # Load and split the PDF document into smaller chunks
#     loader = PyPDFLoader(temp_file_path)
#     documents = loader.load()

#     # Split documents using a text splitter
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1500,
#         chunk_overlap=150
#     )
#     splits = text_splitter.split_documents(documents)

#     # Debug: Display the first few splits to check context
#     st.write("Extracted Document Chunks:", splits[:3])

#     # Correct instantiation of GoogleGenerativeAIEmbeddings with a specified model
#     embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     persist_directory = 'docs/chroma'
#     vectordb = Chroma(persist_directory=persist_directory, 
#                       embedding_function=embedding)

#     # Delay the execution of similarity search until a question is asked
#     if input_text:
#         # Perform similarity search based on the input question
#         docs = vectordb.similarity_search(input_text, k=6)

#         # Combine all retrieved docs into a single context string
#         context = "\n\n".join([doc.page_content for doc in docs])

#         # Debug: Display retrieved documents to verify correct context
#         st.write("Retrieved Context Documents:", docs[:3])

# # Define a single prompt template for both normal questions and PDF-based questions
# prompt_template = """
# You are a helpful assistant. Please respond to the user queries using the provided context if available.
# If context is insufficient or irrelevant, answer based on your knowledge.
# \n
# Context: {context}
# \n
# Question: {question}
# \n
# Answer:
# """
# QA_CHAIN_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# # Google Gemini LLM and memory chain
# llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
# memory = ConversationBufferMemory(
#     memory_key="chat_history", output_key="answer", return_messages=True
# )

# # Create a conversational chain that handles context if provided
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm,
#     vectordb.as_retriever(search_kwargs={"k": 6}) if uploaded_file else None,
#     return_source_documents=True,
#     memory=memory,
#     verbose=False,
#     combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
# )

# # Only process the input question when it is provided
# if input_text:
#     result = qa_chain.invoke({"context": context, "question": input_text})
#     LLM_result = result["answer"]
#     st.write(LLM_result)

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import streamlit as st
import os
import tempfile

# Load environment variables and configure Google Gemini API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up Streamlit app
st.title('LangChain Demo with Google Gemini API')
input_text = st.text_input("Ask me anything!")

# PDF file uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Define the output parser globally so it can be used in both conditions
output_parser = StrOutputParser()

# Initialize variables to hold document chunks and context
context = ""
docs = []
retriever = None

# Check if a PDF is uploaded
if uploaded_file:
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load and split the PDF document into smaller chunks
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # Split documents using a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(documents)

    # Debug: Display the first few splits to check context
    # st.write("Extracted Document Chunks:", splits[:3])

    # Correct instantiation of GoogleGenerativeAIEmbeddings with a specified model
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    persist_directory = 'docs/chroma'
    vectordb = Chroma(persist_directory=persist_directory, 
                      embedding_function=embedding)

    # Set the retriever to be used in ConversationalRetrievalChain
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})

# Define a single prompt template for both normal questions and PDF-based questions
prompt_template = """
You are a helpful assistant. Please respond to the user queries using the provided context if available.
If context is insufficient or irrelevant, answer based on your knowledge.
\n
Context: {context}
\n
Question: {question}
\n
Answer:
"""
QA_CHAIN_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Google Gemini LLM and memory chain
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
memory = ConversationBufferMemory(
    memory_key="chat_history", output_key="answer", return_messages=True
)

# Create the appropriate chain based on whether a retriever is available
if retriever and input_text:
    # Use ConversationalRetrievalChain when PDF is uploaded and question is asked
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        memory=memory,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    # Perform similarity search based on the input question
    result = qa_chain.invoke({"question": input_text})  # Pass only the question here
    LLM_result = result["answer"]
    st.write(LLM_result)

elif input_text:
    # Use a simple LLMChain for normal questions without PDF
    qa_chain = LLMChain(
        llm=llm,
        prompt=QA_CHAIN_PROMPT,
        output_parser=output_parser,
    )
    # Handle input and provide the response
    response = qa_chain.invoke({'context': context, 'question': input_text})
    formatted_response = f"**Question:** {input_text}\n\n**Answer:**\n\n{response.get('text')}"
    st.markdown(formatted_response)
