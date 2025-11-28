"""
文本生成模組
使用 GPT-2 模型進行創意文本生成
"""

import torch
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict

class TextGenerator:
    def __init__(self, model_name='gpt2'):
        """
        初始化文本生成器
        
        Args:
            model_name: 模型名稱 ('gpt2', 'gpt2-medium', 'gpt2-large')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {self.device}")
        
        # 使用 pipeline
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.model_name = model_name
    
    def generate(self, prompt: str, max_length: int = 100, 
                num_return_sequences: int = 1, temperature: float = 0.7,
                top_k: int = 50, top_p: float = 0.95,
                do_sample: bool = True) -> Dict:
        """
        生成文本
        
        Args:
            prompt: 提示文本
            max_length: 最大長度
            num_return_sequences: 生成序列數量
            temperature: 溫度參數(控制隨機性)
            top_k: Top-K 採樣
            top_p: Top-P (nucleus) 採樣
            do_sample: 是否使用採樣
            
        Returns:
            dict: 生成結果
        """
        try:
            # 生成文本
            outputs = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=50256  # GPT-2 的 EOS token
            )
            
            # 格式化結果
            generated_texts = []
            for output in outputs:
                generated_text = output['generated_text']
                # 移除原始 prompt (可選)
                # generated_text = generated_text[len(prompt):].strip()
                generated_texts.append(generated_text)
            
            return {
                'prompt': prompt,
                'generated_texts': generated_texts,
                'num_sequences': len(generated_texts),
                'parameters': {
                    'max_length': max_length,
                    'temperature': temperature,
                    'top_k': top_k,
                    'top_p': top_p
                }
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'prompt': prompt,
                'generated_texts': []
            }
    
    def creative_writing(self, prompt: str, style: str = 'creative') -> str:
        """
        創意寫作
        
        Args:
            prompt: 提示
            style: 風格 ('creative', 'formal', 'casual')
            
        Returns:
            str: 生成的文本
        """
        # 根據風格調整參數
        if style == 'creative':
            temperature = 0.9
            top_p = 0.95
        elif style == 'formal':
            temperature = 0.5
            top_p = 0.9
        else:  # casual
            temperature = 0.7
            top_p = 0.92
        
        result = self.generate(
            prompt,
            max_length=150,
            temperature=temperature,
            top_p=top_p
        )
        
        return result['generated_texts'][0] if result['generated_texts'] else ""
    
    def complete_sentence(self, incomplete_sentence: str, 
                         num_completions: int = 3) -> List[str]:
        """
        完成句子
        
        Args:
            incomplete_sentence: 不完整的句子
            num_completions: 完成選項數量
            
        Returns:
            list: 完成選項列表
        """
        result = self.generate(
            incomplete_sentence,
            max_length=len(incomplete_sentence.split()) + 30,
            num_return_sequences=num_completions,
            temperature=0.8
        )
        
        return result.get('generated_texts', [])
    
    def story_continuation(self, story_beginning: str, 
                          continuation_length: int = 200) -> str:
        """
        故事續寫
        
        Args:
            story_beginning: 故事開頭
            continuation_length: 續寫長度
            
        Returns:
            str: 續寫的故事
        """
        result = self.generate(
            story_beginning,
            max_length=continuation_length,
            temperature=0.85,
            top_p=0.95,
            do_sample=True
        )
        
        return result['generated_texts'][0] if result['generated_texts'] else ""
    
    def generate_variations(self, text: str, num_variations: int = 5) -> List[str]:
        """
        生成文本變體
        
        Args:
            text: 原始文本
            num_variations: 變體數量
            
        Returns:
            list: 變體列表
        """
        # 使用較高的溫度來增加多樣性
        result = self.generate(
            text,
            max_length=len(text.split()) + 50,
            num_return_sequences=num_variations,
            temperature=1.0,
            top_k=100
        )
        
        return result.get('generated_texts', [])
    
    def controlled_generation(self, prompt: str, keywords: List[str],
                            max_length: int = 100) -> str:
        """
        受控生成(嘗試包含特定關鍵詞)
        
        Args:
            prompt: 提示
            keywords: 關鍵詞列表
            max_length: 最大長度
            
        Returns:
            str: 生成的文本
        """
        # 將關鍵詞加入提示
        enhanced_prompt = f"{prompt} (關鍵詞: {', '.join(keywords)})"
        
        result = self.generate(
            enhanced_prompt,
            max_length=max_length,
            temperature=0.7
        )
        
        return result['generated_texts'][0] if result['generated_texts'] else ""

# 便捷函數
def quick_generate(prompt: str, max_length: int = 100) -> str:
    """快速生成文本"""
    generator = TextGenerator()
    result = generator.generate(prompt, max_length=max_length)
    return result['generated_texts'][0] if result['generated_texts'] else ""

if __name__ == "__main__":
    # 測試代碼
    generator = TextGenerator(model_name='gpt2')
    
    print("=== 文本生成測試 ===\n")
    
    # 基本生成
    prompt1 = "Once upon a time in a distant galaxy"
    print(f"提示: {prompt1}")
    result1 = generator.generate(prompt1, max_length=80)
    print(f"生成: {result1['generated_texts'][0]}\n")
    
    # 多個序列
    prompt2 = "The future of artificial intelligence is"
    print(f"提示: {prompt2}")
    result2 = generator.generate(prompt2, max_length=60, num_return_sequences=3)
    print("生成的3個版本:")
    for i, text in enumerate(result2['generated_texts'], 1):
        print(f"{i}. {text}\n")
    
    # 創意寫作
    print("=== 創意寫作測試 ===")
    creative_prompt = "In the year 2050, technology has changed everything."
    creative_text = generator.creative_writing(creative_prompt, style='creative')
    print(f"創意文本: {creative_text}\n")
    
    # 句子完成
    print("=== 句子完成測試 ===")
    incomplete = "The most important thing in life is"
    completions = generator.complete_sentence(incomplete, num_completions=3)
    print(f"不完整句子: {incomplete}")
    print("完成選項:")
    for i, comp in enumerate(completions, 1):
        print(f"{i}. {comp}")