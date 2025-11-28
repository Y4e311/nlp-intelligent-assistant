"""
命名實體識別 (NER) 模組
使用 BERT-NER 模型識別人名、地名、組織等實體
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Tuple

class NamedEntityRecognizer:
    def __init__(self, model_name='dslim/bert-base-NER'):
        """
        初始化NER模型
        
        Args:
            model_name: 預訓練模型名稱
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {self.device}")
        
        # 使用 pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def recognize(self, text: str) -> Dict:
        """
        識別文本中的命名實體
        
        Args:
            text: 輸入文本
            
        Returns:
            dict: 包含實體列表和統計信息
        """
        try:
            # 執行NER
            entities = self.ner_pipeline(text)
            
            # 格式化結果
            formatted_entities = []
            entity_types = {}
            
            for entity in entities:
                entity_type = entity['entity_group']
                entity_text = entity['word']
                score = entity['score']
                
                formatted_entities.append({
                    'text': entity_text,
                    'type': entity_type,
                    'confidence': round(score * 100, 2),
                    'start': entity['start'],
                    'end': entity['end']
                })
                
                # 統計實體類型
                if entity_type not in entity_types:
                    entity_types[entity_type] = []
                entity_types[entity_type].append(entity_text)
            
            return {
                'entities': formatted_entities,
                'entity_types': entity_types,
                'total_entities': len(formatted_entities),
                'text': text
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'entities': [],
                'entity_types': {},
                'total_entities': 0
            }
    
    def get_entities_by_type(self, text: str, entity_type: str) -> List[str]:
        """
        獲取特定類型的實體
        
        Args:
            text: 輸入文本
            entity_type: 實體類型 (PER, ORG, LOC, MISC)
            
        Returns:
            list: 實體列表
        """
        result = self.recognize(text)
        entity_types = result.get('entity_types', {})
        return entity_types.get(entity_type, [])
    
    def highlight_entities(self, text: str) -> str:
        """
        在文本中高亮實體
        
        Args:
            text: 輸入文本
            
        Returns:
            str: 帶有標記的文本
        """
        result = self.recognize(text)
        entities = result.get('entities', [])
        
        if not entities:
            return text
        
        # 按位置排序(從後往前,避免位置偏移)
        entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        highlighted_text = text
        for entity in entities_sorted:
            start = entity['start']
            end = entity['end']
            entity_text = entity['text']
            entity_type = entity['type']
            
            # 插入標記
            marked_text = f"[{entity_text}]({entity_type})"
            highlighted_text = highlighted_text[:start] + marked_text + highlighted_text[end:]
        
        return highlighted_text
    
    def get_entity_statistics(self, texts: List[str]) -> Dict:
        """
        獲取多個文本的實體統計
        
        Args:
            texts: 文本列表
            
        Returns:
            dict: 統計信息
        """
        all_entity_types = {
            'PER': [],  # Person
            'ORG': [],  # Organization
            'LOC': [],  # Location
            'MISC': []  # Miscellaneous
        }
        
        total_entities = 0
        
        for text in texts:
            result = self.recognize(text)
            entity_types = result.get('entity_types', {})
            total_entities += result.get('total_entities', 0)
            
            for etype, entities in entity_types.items():
                if etype in all_entity_types:
                    all_entity_types[etype].extend(entities)
        
        # 去重並計數
        for etype in all_entity_types:
            all_entity_types[etype] = list(set(all_entity_types[etype]))
        
        return {
            'entity_counts': {k: len(v) for k, v in all_entity_types.items()},
            'unique_entities': all_entity_types,
            'total_entities': total_entities
        }

# 便捷函數
def quick_ner(text: str) -> List[Dict]:
    """快速NER識別"""
    recognizer = NamedEntityRecognizer()
    result = recognizer.recognize(text)
    return result.get('entities', [])

if __name__ == "__main__":
    # 測試代碼
    recognizer = NamedEntityRecognizer()
    
    test_texts = [
        "Apple Inc. is planning to open a new store in New York. Tim Cook announced this yesterday.",
        "Microsoft CEO Satya Nadella visited the Seattle headquarters last month.",
        "Google and Facebook are the largest tech companies in Silicon Valley."
    ]
    
    print("=== NER 測試 ===")
    for text in test_texts:
        print(f"\n原文: {text}")
        result = recognizer.recognize(text)
        print(f"識別到 {result['total_entities']} 個實體:")
        
        for entity in result['entities']:
            print(f"  - {entity['text']} ({entity['type']}) - 信心度: {entity['confidence']}%")
        
        print(f"高亮文本: {recognizer.highlight_entities(text)}")
    
    # 統計測試
    print("\n=== 實體統計 ===")
    stats = recognizer.get_entity_statistics(test_texts)
    print(f"總實體數: {stats['total_entities']}")
    print(f"各類型實體數: {stats['entity_counts']}")