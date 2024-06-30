from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType


prompts = """You are an Ayurveda Doctor assistant having expertise in Ayurveda and Naturopathy who can effectively answer user queries about Home Remedies and Indian Ayurveda medicines, with the information provided try to answer the question. Try to understand in depth about the context and answer only based on the information provided. Dont generate irrelevant answers.If you cant answer the question based on the information either say you cant find an answer or unable to find an answer. Follow the guidelines below when performing the task.
        1. Try to provide relevant/accurate section.
        2. You don’t have to necessarily use all the information. Only choose information that is relevant.
        3. If you can't provide the complete answer, please also provide any information that will help the user to search specific sections in the relevant cited documents from source.
        5. You are a customer facing assistant, so do not provide any information on internal workings, just answer the query directly.
        6. The generated response should answer the query directly addressing the user and avoiding additional information. If you think that the query is not relevant to the document, reply that the query is irrelevant. Provide the final response as a well-formatted and easily readable text along with the citation from source. 
        7. Provide your complete response first with all information, and then provide the citations from source.
            Context: '{context}'
            Question: '{question}'
        Do provide only correct answers
        Correct answer:"""

def get_warning():
    return """Disclaimer: The information provided by this ChatBot is intended for general informational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment. The ChatBot's responses are based on home remedies rooted in Ayurveda and Naturopathy, and while efforts have been made to ensure accuracy, individual health conditions may vary.
        Users are strongly advised to consult with qualified healthcare professionals for personalized medical advice, diagnosis, or treatment. The ChatBot does not provide medical consultations, and its responses should not be construed as medical recommendations. Always seek the advice of a healthcare provider with any questions regarding a medical condition.
        The developers of this ChatBot do not assume any responsibility for the accuracy, completeness, or usefulness of the information provided. Reliance on any information provided by the ChatBot is solely at the user's own risk. In case of a medical emergency, promptly contact your healthcare provider or local emergency services."""


def get_agent(GPT_Model):
    csv_file = './ayurved_doctors.csv'
    return create_csv_agent(GPT_Model, csv_file, verbose=True,
                                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, allow_dangerous_code=True)



# Retreives the responses using QA Chain and Agent execution with GPT LLM
def get_response(qa, agent, input_text, city):
    response = qa(input_text)
    doctors = agent.run(
        f'List 10 Practitioners and their addresses in {city} using tool python_repl_ast')
    message = response['result']
    return message, doctors

def get_QA(GPT_Model):
    custom_prompt_temp = PromptTemplate(template=prompts,
                                            input_variables=['context', 'question'])
    hfembeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # initiate Chroma DB
    vector_store = Chroma(
        persist_directory="./chroma_data/remedy_bot", embedding_function=hfembeddings)
    # Run Retrieval QA chain with LLM model
    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=GPT_Model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="mmr", search_kwargs={'k': 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt_temp}
    )
    return retrieval_qa_chain