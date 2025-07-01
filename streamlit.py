# Import dan Setup
import streamlit as st
import pandas as pd
from sentence-transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# ----------------- Backend RAG & LLM -----------------

## Inisiasi Model Sentence Transformer
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

## FAISS & Cosine
def build_faiss_index_cosine(teks):
    embeddings = model.encode(teks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    return index, embeddings

## Retrieval
def retrieve(query, index, df, top_k = None):
    return df

## LLM - Generate Answer
def generate_answer(query, context, api_key) :
    #untuk memasukkan api key dari generative model yang digunakan
    openai.api_key = api_key
    #untuk memberitahu secara spesifik apa yang perlu dilakukan oleh model generative AI
    system_message = "Kamu adalah assisten cerdas yang menjawab petanyaan berdasarkan data yang diberikan"
    #user untuk memberikan inputan pertanyaan ataupun data  yang ingin dipelajari
    user_message = f"""
    Pertanyaan{query}

    Data yang relevan:
    {context}
    """
    response = openai.ChatCompletion.create(
        model='gpt-4.1-mini', #model yang digunakan
        #system message untuk mengolah inputan data ataupun user
        messages = [
            {"role" : "system", "content" : system_message},
            {"role" : "user", "content" : user_message}
        ],
        #untuk mengatur tingkat pemilihan prediksi kata berikutnya
        temperature = 0.7,
        #untuk mengatur jumlah token yang bisa di proses
        max_tokens=1000
    )
    return response.choices[0].message["content"]

# ----------------- UI -----------------

## Title Main Page
st.title("Nama Judul Main Page!!")

## Sidebar
### Input Sidebar
st.sidebar.header("Nama Judul Sidebar")

uploaded_file = st.sidebar.file_uploader("Upload File", type='csv')
input_api_key = st.sidebar.text_input("Masukkan API Key", type='password')
button_api = st.sidebar.button('Aktifkan API Key')

## Pengaturan Backend Sidebar
#penjelasan :
# 1. API KEY akan di anggap kosong
# 2. API KEY sudah terisi, akan di ingat oleh fungsi session state
#  
if 'api_key' not in st.session_state: #api_key adalah semacam variabel session, diberi nama api_key untuk nampung input_api_key
    st.session_state.api_key = None
if input_api_key and button_api:
    st.session_state.api_key = input_api_key
    st.sidebar.success("API Key Aktif")


## Pengaturan Output File Setelah di Upload 

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    selected_columns = st.multiselect(  #pemilihan kolum di input ke variabel SELECTED_COLUMNS
        label = 'Silahkan Pilih Kolom Data',
        options=df.columns.to_list(),
        default=df.columns.to_list()
    )

    if not selected_columns :
        st.warning("Silahkan Pilih 1 Kolom")
        st.stop()
    ### Tampilan Preview Kolom Yang Dipilih
    st.dataframe(df[selected_columns])

    ### Fungsi Menggabungkan Kolom
    def penggabungan_kolom(df, selected_columns):
        df['teks'] = df[selected_columns].astype('str').agg(' | '.join, axis = 1)
        return df

    ### Input Pertanyaan Hanya Muncul Jika Kolom Telah Dipilih
    query = st.text_input("Masukkan Pertanyaan")
    run_query = st.button("Jawab Pertanyaan")

    ### Menjalankan Semua Proses
    if run_query and st.session_state.api_key:
        try:
            df = penggabungan_kolom(df, selected_columns)
            index,_= build_faiss_index_cosine(df['teks'].to_list())

            with st.spinner("Mencari Data Yang Relevan") :
                result = retrieve(query, index, df)
                context = "\n".join(result['teks'].to_list())
            
            with st.spinner("Memberikan Jawaban") :
                answer = generate_answer(query, context, st.session_state.api_key)
            
            st.subheader("Jawaban:")
            st.success(answer)

        except Exception as e:
            st.error(f"Terjadi sebuah error: {str(e)}")
    elif run_query and not st.session_state.api_key:
        st.warning("Masukkan API Key")
    else :
        st.warning("Masukkan data atau coba jalankan query terlebih dahulu")

