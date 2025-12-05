import streamlit as st
import re
import torch 
import transformers 
from transformers import RobertaForSequenceClassification, AutoTokenizer, pipeline
from underthesea import word_tokenize
import sqlite3
import pandas as pd
#@title Khởi tạo pipeline 
@st.cache_resource
def load_pipeline():
  model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
  tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

  try:
    sentiment_pipeline = pipeline(
      "sentiment-analysis",
      model=model,
      tokenizer=tokenizer,
    )
  except Exception as e:
    print(f"Lỗi tải pipeline: {e}")
  return sentiment_pipeline
MIN_WORDS = 5
MAX_WORDS = 50
def main():

  #@title Xử lý cơ bản
  def basic_clean(text):
    # Nếu không phải string
    if not isinstance(text, str): return ""
    # Chuyển chữ thường
    text = text.lower()
    # Chuẩn hóa sau dấu câu có whitespace
    text = re.sub(r'([.,!?;:()])', r'\1 ', text)
    # Loại bỏ số
    #text = re.sub(r'\d+', '', text)
    # Loại dấu câu, kí tự đặc biệt
    text = re.sub(r'[^\w\s]', '', text)
    # Loại bỏ whitespace thừa
    text  = re.sub(r'\s+', ' ', text)
    # Loại chữ cái giống nhau liên tiếp
    text = re.sub(r"(.)\1+", r"\1", text)
    return text

  #@title Chuẩn hóa viết tắt, không dấu, giới hạn ký tự 
  VI_NORM_DICT = {
      "bth": "bình thường", "hnay": "hôm nay",
      "rat": "rất", "toi": "tôi",
      "wa": "quá", "nhieu": "nhiều",
      "dc": "được", "troi": "trời",
      "ko": "không", "k": "không",
      "khong": "không", "t": "tôi",
      "bh": "bây giờ", "bùn": "buồn",
      "lun": "luôn", "tr": "trời",
      "mún": "muốn", "qua": "quá",
      "thui": "thôi", "hom": "hôm",
      "do": "dở",
      "dui": "vui",
  }
  def VI_normalize(text:str, max_len:int):
    words = text.split()
    processed = []
    for word in words:
      processed.append(VI_NORM_DICT.get(word, word))
    normalized = " ".join(processed)
    if len(normalized) > max_len:
      normalized = normalized[:max_len]
    return normalized

  #@title Text Preprocessing Utility 
  def Text_Preprocess_Util(text):
    cleaned = basic_clean(text)
    normalized = VI_normalize(cleaned, 50)
    return normalized

  # DATABASE
  # Khởi tạo database
  
  DB_NAME = 'sentiment_history.db'

  def init_db():
      conn = sqlite3.connect(DB_NAME)
      c = conn.cursor()
      # Bảng history (id, timestamp tự động)
      c.execute('''
          CREATE TABLE IF NOT EXISTS history (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              text TEXT NOT NULL,
              sentiment TEXT NOT NULL,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
          )
      ''')
      conn.commit()
      conn.close()
  # Lưu kq vào database
  def save_to_db(text, sentiment):
      conn = sqlite3.connect(DB_NAME)
      c = conn.cursor()
      c.execute('''
          INSERT INTO history (text, sentiment)
          VALUES (?, ?)
      ''', (text, sentiment))
      conn.commit()
      conn.close()

  # Map label cho database
  def map_label_for_db(raw_label):
          if 'POS' in raw_label.upper():
              return 'POSITIVE'
          elif 'NEG' in raw_label.upper():
              return 'NEGATIVE'
          else:
              return 'NEUTRAL'
  # Đọc lịch sử
  def load_history():
      conn = sqlite3.connect(DB_NAME)
      # Sử dụng read_sql_query để đọc trực tiếp thành DataFrame
      df_history = pd.read_sql_query("SELECT * FROM history ORDER BY timestamp DESC LIMIT 50", conn)
      conn.close()
      return df_history

  # Load pipeline
  pipeline = load_pipeline()
  st.title("Trợ lý phân loại cảm xúc VN")
  st.markdown("Sử dụng pre-trained model PhoBERT")
  init_db()
  usr_input = st.text_area("Nhập câu tiếng việt:", height=150)
  # Xử lý nhấn nút
  if st.button("Phân tích"):
    # Nếu chuỗi trống
    if not usr_input or usr_input.strip() == "":
      st.error("**Lỗi:** Vui lòng nhập câu!")
      st.stop() 
    if len(usr_input) <= MIN_WORDS:
      st.error(f"**Lỗi:** Vui lòng nhập ít nhất {MIN_WORDS} ký tự!")
      st.stop()
    elif len(usr_input) > MAX_WORDS:
      st.warning(f"Câu tối đa {MAX_WORDS} ký tự, sẽ được rút gọn")
    # Nếu input không lỗi
    with st.spinner('Đang tiến hành phân tích...'):
        # Chuẩn hóa input
        cleaned = Text_Preprocess_Util(usr_input)
        result = pipeline(cleaned)
        # Lưu vào database
        raw_label = result[0]['label']
        label = map_label_for_db(raw_label)
        score = result[0]['score']
        save_to_db(cleaned, label)
        # Hiển thị UI
        st.subheader("Kết quả phân tích:")
        st.write(f"Input đã chuẩn hóa: {cleaned}")
        if 'POS' in label.upper():
          st.success(f"**Cảm xúc: POSITIVE** (Confidence: {score:.4f})")
        elif 'NEG' in label.upper():
          st.error(f"**Cảm xúc: NEGATIVE** (Confidence: {score:.4f})")
        else:
          st.info(f"**Cảm xúc: NEUTRAL** (Confidence: {score:.4f})")
    # Hiển thị lịch sử
    st.markdown("--------------------")
    st.subheader("Lịch sử phân tích")
    with st.expander("Hiển thị 50 kết quả gần nhất"):
      df_history = load_history()
      if not df_history.empty:
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_display = df_history[['text', 'sentiment']]
        df_display.columns = ['Văn bản', 'Nhãn']

        st.dataframe(df_display, use_container_width=True)
      else:
        st.info("Không có lịch sử phân tích nào.")

if __name__ == "__main__":
    main()