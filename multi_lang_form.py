import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# Initialize translation pipeline
text_translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0)

# Language options for translation
language_dict = {
    "Greek": "ell_Grek", 
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Italian": "ita_Latn",
    "Japanese": "jpn_Jpan",
    "Kannada": "kan_Knda",
    "Luxembourgish": "ltz_Latn",
    "Urdu": "urd_Arab"
}

# Function to translate text
def translate(text, src_lang="eng_Latn", tgt_lang="eng_Latn"):
    try:
        translated_text = text_translator(text, src_lang=src_lang, tgt_lang=tgt_lang)
        return translated_text[0]['translation_text']
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return text

# Streamlit frontend
def main():
    st.title("Multi-Language Form")

    # Select language
    selected_language = st.selectbox("Select Language", list(language_dict.keys()))
    src_lang = language_dict["English"]  # Default source language is English
    tgt_lang = language_dict[selected_language]  # Target language is based on user selection

    # Labels in English
    labels = {
        "name": "Name",
        "address": "Address",
        "income": "Income",
        "email": "Email",
        "submit": "Submit"
    }

    # Translate labels based on selected language
    translated_labels = {key: translate(value, src_lang=src_lang, tgt_lang=tgt_lang) for key, value in labels.items()}

    # Form fields
    with st.form(key='multi_lang_form'):
        name = st.text_input(translated_labels["name"])
        address = st.text_area(translated_labels["address"])
        income = st.number_input(translated_labels["income"], min_value=0, step=1000)
        email = st.text_input(translated_labels["email"])

            # Submit button with customized label for English
        if selected_language == "English":
            submit_button = st.form_submit_button("Submit")  # Static label for English
        else:
            submit_button = st.form_submit_button(translated_labels["submit"])
        
        if submit_button:
            st.write("Form Submitted!")
            st.write(f"{translated_labels['name']}: {name}")
            st.write(f"{translated_labels['address']}: {address}")
            st.write(f"{translated_labels['income']}: {income}")
            st.write(f"{translated_labels['email']}: {email}")

if __name__ == '__main__':
    main()        

# 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
