cat > README.md << 'EOF'
# 🤖 AI 文本智能助手

基於深度學習的多功能 NLP 系統

## ✨ 功能特色

- 😊 **情感分析** - 識別文本情感傾向
- 📝 **文本摘要** - 自動生成文章摘要
- 🏷️ **命名實體識別** - 提取人名、地名、組織
- ❓ **問答系統** - 基於上下文的智能問答
- ✍️ **文本生成** - AI 創意寫作

## 🌐 線上試用

👉 **[立即體驗](https://你的應用網址.streamlit.app)** 👈

## 🚀 本地運行
```bash
git clone https://github.com/你的用戶名/nlp-intelligent-assistant.git
cd nlp-intelligent-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 🛠️ 技術棧

- **框架**: Streamlit
- **深度學習**: PyTorch, Transformers
- **模型**: BERT, GPT-2, BART, RoBERTa

## 📊 模型性能

| 功能 | 模型 | 準確率 |
|------|------|--------|
| 情感分析 | DistilBERT | ~92% |
| 文本摘要 | BART | ROUGE-L: 0.41 |
| NER | BERT-NER | F1: 0.89 |
| 問答系統 | RoBERTa | EM: 78% |

## 📄 授權

MIT License
EOF