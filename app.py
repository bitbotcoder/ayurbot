import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from st_social_media_links import SocialMediaIcons
from util import get_warning, get_agent, get_QA,get_response


st.set_page_config(
    page_title="Nature's Rx ChatBot - powered by ChatGPT, Langchain and Chroma", page_icon="🤖")
st.header("👨‍⚕️ Nature's Rx ChatBot ",divider="rainbow")
st.write("🍀Your Ayurveda and Naturopathy Companion for Natural Healing 🍀🌿💊")
st.info("""This app does not store or share API keys externally. Configure Your OpenAI API Key and Hit Enter""", icon="ℹ️")
api_key = st.text_input("OpenAI Key", label_visibility="collapsed",placeholder='OpenAI API Key',type="password")

#api_key = open("openai_api_key.txt", "r").read().strip()
# Instantiate LLM model
if api_key:
    GPT_Model2 = ChatOpenAI(api_key=api_key, temperature=0.1,
                    model="gpt-3.5-turbo-0125")
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    agent = get_agent(GPT_Model2)
    qa = get_QA(GPT_Model2)

    # Streamlit form components, using sessions state to maintain the chat history, chat_message and text components.
    
    with st.form("my_form"):
        col3, col4 = st.columns([1, 1])
        prompt = col3.text_input("Enter text:", "How to cure Common Flu?")
        city = col4.text_input(label="Your City", value="Amritsar")
        submitted = st.form_submit_button("Submit")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    st.chat_message("user").write(prompt)
    
    if submitted:
        st.session_state.messages.append({"role": "user", "content": prompt})
        response, doctors = get_response(qa, agent, prompt, city)
        if response:
            st.chat_message("assistant").write(response)
            st.chat_message("assistant").write(
                f"List of Ayurveda Practioners in {city} and their addresses :")
            st.chat_message("assistant").write(doctors)
            st.session_state["messages"].append(
                {"role": "assistant", "content": response})
            st.session_state["messages"].append(
                {"role": "assistant", "content": f"List of Ayurveda Practioners in {city} and their addresses :"})
            st.session_state["messages"].append(
                {"role": "assistant", "content": doctors})
else:
    st.warning("OpenAI API Key is not configured. Enter your OpenAI Key to access GPT models.", icon="❎")


st.divider()    
st.write("📢 Share with wider community")
social_media_links = [
        "https://x.com/intent/tweet?hashtags=streamlit%2Cpython&text=Check%20out%20this%20awesome%20Streamlit%20app%20I%20built%0A&url=https%3A%2F%2Fautoml-wiz.streamlit.app",
        "https://www.linkedin.com/sharing/share-offsite/?summary=https%3A%2F%2Fautoml-wiz.streamlit.app%20%23streamlit%20%23python&title=Check%20out%20this%20awesome%20Streamlit%20app%20I%20built%0A&url=https%3A%2F%2Fautoml-wiz.streamlit.app",
        "https://www.facebook.com/sharer/sharer.php?kid_directed_site=0&u=https%3A%2F%2Fautoml-wiz.streamlit.app",
        "https://github.com/bitbotcoder/ayurbot"
    ]
social_media_icons = SocialMediaIcons(social_media_links, colors=["white"] * len(social_media_links))

social_media_icons.render()
st.divider()    

st.warning(get_warning(), icon='⚠️')
