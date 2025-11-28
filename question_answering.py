"""
問答系統模組
基於上下文的問答系統,使用 BERT-QA 模型
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from typing import Dict, List

class QuestionAnsweringSystem:
    def __init__(self, model_name='deepset/roberta-base-squad2'):
        """
        初始化問答系統
        
        Args:
            model_name: 預訓練模型名稱
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {self.device}")
        
        # 使用 pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def answer(self, question: str, context: str, top_k: int = 1) -> Dict:
        """
        回答問題
        
        Args:
            question: 問題
            context: 上下文
            top_k: 返回前k個答案
            
        Returns:
            dict: 包含答案和信心分數
        """
        try:
            # 獲取答案
            result = self.qa_pipeline(
                question=question,
                context=context,
                top_k=top_k
            )
            
            # 如果 top_k=1,結果是字典;否則是列表
            if top_k == 1:
                answer = result['answer']
                score = result['score']
                start = result['start']
                end = result['end']
                
                return {
                    'question': question,
                    'answer': answer,
                    'confidence': round(score * 100, 2),
                    'start': start,
                    'end': end,
                    'context_snippet': self._get_context_snippet(context, start, end)
                }
            else:
                answers = []
                for res in result:
                    answers.append({
                        'answer': res['answer'],
                        'confidence': round(res['score'] * 100, 2),
                        'start': res['start'],
                        'end': res['end']
                    })
                
                return {
                    'question': question,
                    'answers': answers,
                    'top_answer': answers[0] if answers else None
                }
        
        except Exception as e:
            return {
                'error': str(e),
                'question': question,
                'answer': '無法回答',
                'confidence': 0.0
            }
    
    def _get_context_snippet(self, context: str, start: int, end: int, 
                            window: int = 50) -> str:
        """
        獲取答案周圍的上下文片段
        
        Args:
            context: 完整上下文
            start: 答案起始位置
            end: 答案結束位置
            window: 窗口大小
            
        Returns:
            str: 上下文片段
        """
        snippet_start = max(0, start - window)
        snippet_end = min(len(context), end + window)
        
        snippet = context[snippet_start:snippet_end]
        
        # 添加省略號
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(context):
            snippet = snippet + "..."
        
        return snippet
    
    def batch_answer(self, questions: List[str], context: str) -> List[Dict]:
        """
        批次回答多個問題
        
        Args:
            questions: 問題列表
            context: 共享上下文
            
        Returns:
            list: 答案列表
        """
        results = []
        for question in questions:
            results.append(self.answer(question, context))
        return results
    
    def multi_document_qa(self, question: str, documents: List[str]) -> Dict:
        """
        多文檔問答
        
        Args:
            question: 問題
            documents: 文檔列表
            
        Returns:
            dict: 最佳答案及來源
        """
        best_answer = None
        best_score = 0
        best_doc_idx = -1
        
        for idx, doc in enumerate(documents):
            result = self.answer(question, doc)
            score = result.get('confidence', 0)
            
            if score > best_score:
                best_score = score
                best_answer = result
                best_doc_idx = idx
        
        if best_answer:
            best_answer['source_document_index'] = best_doc_idx
            best_answer['source_document'] = documents[best_doc_idx][:200] + "..."
        
        return best_answer if best_answer else {
            'question': question,
            'answer': '無法在文檔中找到答案',
            'confidence': 0.0
        }
    
    def verify_answer(self, question: str, context: str, 
                     expected_answer: str) -> Dict:
        """
        驗證答案是否正確
        
        Args:
            question: 問題
            context: 上下文
            expected_answer: 預期答案
            
        Returns:
            dict: 驗證結果
        """
        result = self.answer(question, context)
        predicted_answer = result.get('answer', '')
        
        # 簡單的字符串匹配
        is_correct = expected_answer.lower() in predicted_answer.lower() or \
                     predicted_answer.lower() in expected_answer.lower()
        
        return {
            'question': question,
            'expected_answer': expected_answer,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'confidence': result.get('confidence', 0)
        }

# 便捷函數
def quick_qa(question: str, context: str) -> str:
    """快速問答"""
    qa_system = QuestionAnsweringSystem()
    result = qa_system.answer(question, context)
    return result.get('answer', '無法回答')

if __name__ == "__main__":
    # 測試代碼
    qa_system = QuestionAnsweringSystem()
    
    context = """
    The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical 
    rainforest in the Amazon biome that covers most of the Amazon basin of South America. 
    This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 
    (2,100,000 sq mi) are covered by the rainforest. This region includes territory 
    belonging to nine nations and 3,344 formally acknowledged indigenous territories.
    The majority of the forest is contained within Brazil, with 60% of the rainforest, 
    followed by Peru with 13%, and Colombia with 10%.
    """
    
    questions = [
        "What is the Amazon rainforest?",
        "How much area does the Amazon basin cover?",
        "Which country contains most of the forest?",
        "What percentage of the forest is in Peru?"
    ]
    
    print("=== 問答系統測試 ===")
    print(f"上下文: {context[:150]}...\n")
    
    for question in questions:
        result = qa_system.answer(question, context)
        print(f"問題: {result['question']}")
        print(f"答案: {result['answer']}")
        print(f"信心度: {result['confidence']}%")
        print(f"上下文片段: {result.get('context_snippet', '')}\n")
    
    # 多答案測試
    print("=== 多答案測試 ===")
    multi_result = qa_system.answer(questions[0], context, top_k=3)
    print(f"問題: {multi_result['question']}")
    print(f"前3個答案:")
    for ans in multi_result['answers']:
        print(f"  - {ans['answer']} (信心度: {ans['confidence']}%)")