"""
AI 文本智能助手 - Streamlit 應用
整合多個 NLP 功能的互動式 Web 介面
"""

import streamlit as st
import sys
import os

# 添加 src 目錄到路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sentiment_analysis import SentimentAnalyzer
from text_summarization import TextSummarizer
from ner import NamedEntityRecognizer
from question_answering import QuestionAnsweringSystem
from text_generation import TextGenerator

# 頁面配置
st.set_page_config(
    page_title="AI 文本智能助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .entity-per {
        background-color: #ffcccc;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .entity-org {
        background-color: #ccffcc;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .entity-loc {
        background-color: #ccccff;
        padding: 2px 6px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# 初始化模型 (使用 session_state 避免重複加載)
@st.cache_resource
def load_models():
    """載入所有模型"""
    models = {
        'sentiment': SentimentAnalyzer(),
        'summarizer': TextSummarizer(),
        'ner': NamedEntityRecognizer(),
        'qa': QuestionAnsweringSystem(),
        'generator': TextGenerator()
    }
    return models

# 主標題
st.markdown('<p class="main-header">🤖 AI 文本智能助手</p>', unsafe_allow_html=True)
st.markdown("---")

# 側邊欄
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/artificial-intelligence.png", width=150)
    st.title("功能選單")
    
    app_mode = st.selectbox(
        "選擇功能",
        ["🏠 首頁", "😊 情感分析", "📝 文本摘要", "🏷️ 命名實體識別", 
         "❓ 問答系統", "✍️ 文本生成"]
    )
    
    st.markdown("---")
    st.markdown("### 關於專題")
    st.info("""
    **NLP 深度學習專題**
    
    整合五大 NLP 功能:
    - 情感分析
    - 文本摘要  
    - 命名實體識別
    - 問答系統
    - 文本生成
    
    使用 BERT、GPT-2 等
    先進的預訓練模型
    """)

# 載入模型
try:
    with st.spinner("正在載入模型..."):
        models = load_models()
    st.sidebar.success("✅ 模型載入完成")
except Exception as e:
    st.sidebar.error(f"❌ 模型載入失敗: {str(e)}")
    models = None

# 首頁
if app_mode == "🏠 首頁":
    st.header("歡迎使用 AI 文本智能助手!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 😊 情感分析
        - 分析文本情感傾向
        - 正面/負面/中立分類
        - 信心度評分
        """)
        
        st.markdown("""
        ### 🏷️ 命名實體識別
        - 識別人名、地名、組織
        - 自動標註實體
        - 統計分析
        """)
    
    with col2:
        st.markdown("""
        ### 📝 文本摘要
        - 自動生成摘要
        - 支援長文本
        - 壓縮率控制
        """)
        
        st.markdown("""
        ### ✍️ 文本生成
        - AI 創意寫作
        - 故事續寫
        - 多樣化生成
        """)
    
    with col3:
        st.markdown("""
        ### ❓ 問答系統
        - 基於上下文問答
        - 智能答案提取
        - 多文檔支援
        """)
    
    st.markdown("---")
    st.info("👈 請從左側選單選擇功能開始使用")

# 情感分析
elif app_mode == "😊 情感分析":
    st.header("😊 情感分析")
    st.write("分析文本的情感傾向,判斷是正面、負面還是中立")
    
    text_input = st.text_area(
        "輸入要分析的文本:",
        height=150,
        placeholder="例如: This product is amazing! I love it so much!"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        language = st.selectbox("語言", ["English", "中文"])
        lang_code = 'zh' if language == "中文" else 'en'
    
    if st.button("🔍 分析情感", type="primary"):
        if text_input.strip():
            with st.spinner("正在分析..."):
                result = models['sentiment'].analyze(text_input, lang_code)
            
            if 'error' not in result:
                st.markdown("### 分析結果")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("情感", result['sentiment'])
                with col2:
                    st.metric("信心度", f"{result['confidence']}%")
                
                # 視覺化
                sentiment_color = {
                    '正面': '🟢',
                    '負面': '🔴',
                    '中立': '🟡'
                }
                st.markdown(f"## {sentiment_color.get(result['sentiment'], '⚪')} {result['sentiment']}")
                
                # 進度條
                st.progress(result['confidence'] / 100)
            else:
                st.error(f"錯誤: {result['error']}")
        else:
            st.warning("請輸入文本")

# 文本摘要
elif app_mode == "📝 文本摘要":
    st.header("📝 文本摘要")
    st.write("自動生成文本摘要,快速掌握長文重點")
    
    text_input = st.text_area(
        "輸入要摘要的文本:",
        height=200,
        placeholder="輸入較長的文本..."
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_length = st.slider("最大長度", 50, 300, 150)
    with col2:
        min_length = st.slider("最小長度", 20, 100, 50)
    with col3:
        method = st.selectbox("方法", ["生成式", "抽取式"])
    
    if st.button("📄 生成摘要", type="primary"):
        if text_input.strip():
            with st.spinner("正在生成摘要..."):
                if method == "生成式":
                    result = models['summarizer'].summarize(
                        text_input, 
                        max_length=max_length,
                        min_length=min_length
                    )
                else:
                    num_sentences = max_length // 20  # 估算句子數
                    result = models['summarizer'].extractive_summary(
                        text_input,
                        num_sentences=max_sentences
                    )
            
            if 'error' not in result:
                st.markdown("### 摘要結果")
                st.markdown(f'<div class="result-box">{result["summary"]}</div>', 
                          unsafe_allow_html=True)
                
                # 統計信息
                if 'compression_ratio' in result:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("原文長度", f"{result.get('original_length', 0)} 詞")
                    with col2:
                        st.metric("摘要長度", f"{result.get('summary_length', 0)} 詞")
                    with col3:
                        st.metric("壓縮率", f"{result.get('compression_ratio', 0)}")
            else:
                st.error(f"錯誤: {result.get('error', 'Unknown error')}")
        else:
            st.warning("請輸入文本")

# 命名實體識別
elif app_mode == "🏷️ 命名實體識別":
    st.header("🏷️ 命名實體識別 (NER)")
    st.write("自動識別文本中的人名、地名、組織名等實體")
    
    text_input = st.text_area(
        "輸入文本:",
        height=150,
        placeholder="例如: Apple Inc. CEO Tim Cook announced new products in California."
    )
    
    if st.button("🔍 識別實體", type="primary"):
        if text_input.strip():
            with st.spinner("正在識別..."):
                result = models['ner'].recognize(text_input)
            
            if result['total_entities'] > 0:
                st.markdown(f"### 識別到 {result['total_entities']} 個實體")
                
                # 顯示高亮文本
                highlighted = models['ner'].highlight_entities(text_input)
                st.markdown("#### 標註文本:")
                st.markdown(f'<div class="result-box">{highlighted}</div>', 
                          unsafe_allow_html=True)
                
                # 實體列表
                st.markdown("#### 實體詳情:")
                for entity in result['entities']:
                    entity_type = entity['type']
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{entity['text']}**")
                    with col2:
                        st.write(f"`{entity_type}`")
                    with col3:
                        st.write(f"{entity['confidence']}%")
                
                # 統計
                st.markdown("#### 實體統計:")
                entity_types = result['entity_types']
                for etype, entities in entity_types.items():
                    st.write(f"**{etype}**: {', '.join(set(entities))}")
            else:
                st.info("未識別到實體")
        else:
            st.warning("請輸入文本")

# 問答系統
elif app_mode == "❓ 問答系統":
    st.header("❓ 智能問答系統")
    st.write("基於上下文的智能問答,提供精確答案")
    
    context = st.text_area(
        "上下文 (Context):",
        height=200,
        placeholder="輸入背景資料或文章內容..."
    )
    
    question = st.text_input(
        "問題 (Question):",
        placeholder="根據上下文提出問題..."
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        top_k = st.slider("答案數量", 1, 5, 1)
    
    if st.button("💡 獲取答案", type="primary"):
        if context.strip() and question.strip():
            with st.spinner("正在思考..."):
                result = models['qa'].answer(question, context, top_k=top_k)
            
            if 'error' not in result:
                st.markdown("### 答案")
                
                if top_k == 1:
                    st.markdown(f'<div class="result-box"><h3>{result["answer"]}</h3></div>', 
                              unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("信心度", f"{result['confidence']}%")
                    with col2:
                        if result['confidence'] > 80:
                            st.success("高信心度 ✅")
                        elif result['confidence'] > 50:
                            st.info("中等信心度 ℹ️")
                        else:
                            st.warning("低信心度 ⚠️")
                    
                    # 上下文片段
                    st.markdown("#### 相關上下文:")
                    st.markdown(f'<div class="result-box">{result.get("context_snippet", "")}</div>',
                              unsafe_allow_html=True)
                else:
                    st.markdown("#### 多個答案候選:")
                    for i, ans in enumerate(result['answers'], 1):
                        st.markdown(f"**{i}. {ans['answer']}** (信心度: {ans['confidence']}%)")
            else:
                st.error(f"錯誤: {result['error']}")
        else:
            st.warning("請輸入上下文和問題")

# 文本生成
elif app_mode == "✍️ 文本生成":
    st.header("✍️ AI 文本生成")
    st.write("使用 AI 進行創意寫作和文本續寫")
    
    prompt = st.text_area(
        "輸入提示 (Prompt):",
        height=100,
        placeholder="例如: Once upon a time in a magical forest..."
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_length = st.slider("最大長度", 50, 300, 100)
    with col2:
        temperature = st.slider("創意度", 0.1, 2.0, 0.7, 0.1)
    with col3:
        num_sequences = st.slider("生成數量", 1, 5, 1)
    
    style = st.selectbox("寫作風格", ["創意 (Creative)", "正式 (Formal)", "隨意 (Casual)"])
    
    if st.button("✨ 生成文本", type="primary"):
        if prompt.strip():
            with st.spinner("AI 正在創作..."):
                result = models['generator'].generate(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=num_sequences,
                    temperature=temperature
                )
            
            if result['generated_texts']:
                st.markdown("### 生成結果")
                
                for i, text in enumerate(result['generated_texts'], 1):
                    if num_sequences > 1:
                        st.markdown(f"#### 版本 {i}:")
                    st.markdown(f'<div class="result-box">{text}</div>', 
                              unsafe_allow_html=True)
                    st.markdown("---")
                
                # 顯示參數
                with st.expander("生成參數"):
                    st.json(result['parameters'])
            else:
                st.error("生成失敗")
        else:
            st.warning("請輸入提示文本")

# 頁腳
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 AI 文本智能助手 | NLP 深度學習專題 | Powered by Transformers & Streamlit</p>
</div>
""", unsafe_allow_html=True)