"""
文本摘要模組
使用 BART/T5 模型進行自動摘要生成
"""

import torch
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
from typing import List, Dict

# 下載必要的 NLTK 資源
try:
    nltk.download('punkt', quiet=True)
except:
    pass

class TextSummarizer:
    def __init__(self, model_type='bart'):
        """
        初始化文本摘要器
        
        Args:
            model_type: 模型類型 ('bart' 或 't5')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {self.device}")
        
        self.model_type = model_type
        
        # 使用 pipeline 簡化
        if model_type == 'bart':
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
        else:  # t5
            self.summarizer = pipeline(
                "summarization",
                model="t5-small",
                device=0 if torch.cuda.is_available() else -1
            )
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50, 
                  do_sample: bool = False) -> Dict:
        """
        生成文本摘要
        
        Args:
            text: 輸入文本
            max_length: 最大長度
            min_length: 最小長度
            do_sample: 是否使用採樣
            
        Returns:
            dict: 包含摘要的字典
        """
        try:
            # 檢查文本長度
            if len(text.split()) < 50:
                return {
                    'summary': text,
                    'compression_ratio': 1.0,
                    'message': '文本太短,無需摘要'
                }
            
            # 生成摘要
            result = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                truncation=True
            )[0]
            
            summary = result['summary_text']
            
            # 計算壓縮率
            original_length = len(text.split())
            summary_length = len(summary.split())
            compression_ratio = round(summary_length / original_length, 2)
            
            return {
                'summary': summary,
                'original_length': original_length,
                'summary_length': summary_length,
                'compression_ratio': compression_ratio,
                'message': '摘要生成成功'
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'summary': '',
                'message': '摘要生成失敗'
            }
    
    def extractive_summary(self, text: str, num_sentences: int = 3) -> Dict:
        """
        抽取式摘要 (簡單實現)
        
        Args:
            text: 輸入文本
            num_sentences: 提取句子數量
            
        Returns:
            dict: 包含摘要的字典
        """
        try:
            # 分句
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) <= num_sentences:
                return {
                    'summary': text,
                    'method': 'extractive',
                    'sentences_selected': len(sentences)
                }
            
            # 簡單方法:選擇前N句
            # 實際應用中可以使用 TF-IDF 或其他方法排序
            selected_sentences = sentences[:num_sentences]
            summary = ' '.join(selected_sentences)
            
            return {
                'summary': summary,
                'method': 'extractive',
                'sentences_selected': num_sentences,
                'total_sentences': len(sentences)
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'summary': '',
                'method': 'extractive'
            }
    
    def batch_summarize(self, texts: List[str], **kwargs) -> List[Dict]:
        """
        批次摘要
        
        Args:
            texts: 文本列表
            **kwargs: 傳遞給 summarize 的參數
            
        Returns:
            list: 摘要結果列表
        """
        results = []
        for text in texts:
            results.append(self.summarize(text, **kwargs))
        return results

# 便捷函數
def quick_summary(text: str, max_length: int = 150) -> str:
    """快速生成摘要"""
    summarizer = TextSummarizer()
    result = summarizer.summarize(text, max_length=max_length)
    return result.get('summary', '')

if __name__ == "__main__":
    # 測試代碼
    summarizer = TextSummarizer(model_type='bart')
    
    test_article = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. Colloquially, the term 
    "artificial intelligence" is often used to describe machines (or computers) 
    that mimic "cognitive" functions that humans associate with the human mind, 
    such as "learning" and "problem solving". As machines become increasingly 
    capable, tasks considered to require "intelligence" are often removed from 
    the definition of AI, a phenomenon known as the AI effect. A quip in 
    Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, 
    optical character recognition is frequently excluded from things considered 
    to be AI, having become a routine technology.
    """
    
    print("=== 原始文本 ===")
    print(test_article[:200] + "...")
    
    print("\n=== 生成式摘要 ===")
    result = summarizer.summarize(test_article, max_length=100, min_length=30)
    print(f"摘要: {result['summary']}")
    print(f"壓縮率: {result['compression_ratio']}")
    
    print("\n=== 抽取式摘要 ===")
    result2 = summarizer.extractive_summary(test_article, num_sentences=2)
    print(f"摘要: {result2['summary']}")