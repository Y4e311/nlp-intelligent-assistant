"""
情感分析模組
使用 BERT 模型進行文本情感分類
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_name='bert-base-uncased'):
        """
        初始化情感分析器
        
        Args:
            model_name: 預訓練模型名稱
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {self.device}")
        
        # 使用 pipeline 簡化流程
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # 中文情感分析
        self.chinese_pipeline = None
        try:
            self.chinese_pipeline = pipeline(
                "sentiment-analysis",
                model="bert-base-chinese",
                device=0 if torch.cuda.is_available() else -1
            )
        except:
            print("中文模型加載失敗,僅支援英文")
    
    def analyze(self, text, language='en'):
        """
        分析文本情感
        
        Args:
            text: 輸入文本
            language: 語言 ('en' 或 'zh')
            
        Returns:
            dict: 包含標籤和分數的字典
        """
        try:
            if language == 'zh' and self.chinese_pipeline:
                result = self.chinese_pipeline(text)[0]
            else:
                result = self.sentiment_pipeline(text)[0]
            
            # 格式化結果
            label = result['label']
            score = result['score']
            
            # 轉換標籤
            sentiment_map = {
                'POSITIVE': '正面',
                'NEGATIVE': '負面',
                'NEUTRAL': '中立',
                'LABEL_0': '負面',
                'LABEL_1': '正面'
            }
            
            return {
                'sentiment': sentiment_map.get(label, label),
                'confidence': round(score * 100, 2),
                'raw_label': label,
                'raw_score': score
            }
        except Exception as e:
            return {
                'error': str(e),
                'sentiment': '未知',
                'confidence': 0.0
            }
    
    def batch_analyze(self, texts, language='en'):
        """
        批次分析多個文本
        
        Args:
            texts: 文本列表
            language: 語言
            
        Returns:
            list: 結果列表
        """
        results = []
        for text in texts:
            results.append(self.analyze(text, language))
        return results
    
    def get_sentiment_distribution(self, texts, language='en'):
        """
        獲取文本集合的情感分布
        
        Args:
            texts: 文本列表
            language: 語言
            
        Returns:
            dict: 情感分布統計
        """
        results = self.batch_analyze(texts, language)
        
        sentiment_counts = {
            '正面': 0,
            '負面': 0,
            '中立': 0
        }
        
        total_confidence = 0
        
        for result in results:
            sentiment = result.get('sentiment', '未知')
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            total_confidence += result.get('confidence', 0)
        
        return {
            'distribution': sentiment_counts,
            'total': len(texts),
            'average_confidence': round(total_confidence / len(texts), 2) if texts else 0
        }

# 便捷函數
def quick_sentiment(text, language='en'):
    """快速情感分析"""
    analyzer = SentimentAnalyzer()
    return analyzer.analyze(text, language)

if __name__ == "__main__":
    # 測試代碼
    analyzer = SentimentAnalyzer()
    
    # 測試英文
    test_texts_en = [
        "I love this product! It's amazing!",
        "This is terrible. I hate it.",
        "It's okay, nothing special."
    ]
    
    print("=== 英文情感分析測試 ===")
    for text in test_texts_en:
        result = analyzer.analyze(text, 'en')
        print(f"文本: {text}")
        print(f"情感: {result['sentiment']}, 信心度: {result['confidence']}%\n")
    
    # 測試批次分析
    print("=== 情感分布統計 ===")
    distribution = analyzer.get_sentiment_distribution(test_texts_en, 'en')
    print(distribution)