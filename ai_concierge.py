from llama_cpp import Llama
import os
import json

# Путь к локальной модели Llama
MODEL_PATH = os.path.join("models", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")

# Глобальный объект модели (ленивая инициализация)
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")
        
        print(f"🧠 Загрузка LLM {MODEL_PATH}...")
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,  # Контекстное окно
            n_threads=4, # Количество потоков (настрой под свой процессор)
            verbose=False
        )
    return _llm

def get_city_explanation(score, ndvi, infrastructure, user_type):
    """
    Генерирует человекопонятное объяснение оценки района.
    """
    llm = get_llm()
    
    # Формируем список объектов для промпта
    infra_list = [f"{item['name']} ({item['type']})" for item in infrastructure[:10]]
    infra_str = ", ".join(infra_list)
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ты — эксперт по урбанистике из команды CityVibe AI. Твоя задача — объяснить пользователю почему его район получил оценку {score}/100.
Отвечай кратко, дружелюбно и на русском языке.
Используй данные:
- Индекс озеленения (NDVI): {ndvi}%
- Тип пользователя: {user_type}
- Объекты рядом: {infra_str}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Объясни, почему мой район получил {score} баллов? Какие плюсы и минусы?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    response = llm(
        prompt,
        max_tokens=256,
        stop=["<|eot_id|>", "Task:"],
        echo=False
    )
    
    return response['choices'][0]['text'].strip()

if __name__ == "__main__":
    # Тест
    test_infra = [{"name": "Магнит", "type": "Магазин"}, {"name": "Парк Победы", "type": "Парк"}]
    explanation = get_city_explanation(85, 45, test_infra, "resident")
    print(f"🤖 AI: {explanation}")
