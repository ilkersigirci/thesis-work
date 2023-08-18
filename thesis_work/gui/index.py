import streamlit as st

st.set_page_config(
    page_title="Thesis-Work",
    page_icon="👋",
)
st.write("# :blue[Thesis-Work] on the web !")
st.info(
    "For more information please visit the [github page](https://github.com/ilkersigirci/thesis-work)"
)
st.sidebar.success("Select a page in the listed order.")


st.image("thesis_work/gui/resources/dna.jpg")

# video_link = "TODO"
# st.video(video_link)
# video_html = f"""
# <video controls width="700" autoplay="true" muted="true" loop="true">
# <source
# src={video_link}
# type="video/mp4" />
# </video>
# """
# st.markdown(video_html, unsafe_allow_html=True)
