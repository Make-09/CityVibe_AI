from NDVI import get_ndvi_multi_radius
from infrastructure import get_cityvibe_input_data
from predict import predict_score, CityVibeNet

# Маппинг ID классов в читаемые названия
CLASS_NAMES = {
    0: "Магазин",
    1: "Аптека",
    2: "Кафе",
    3: "Клиника",
    4: "Остановка"
}

def evaluate_my_home(lat: float, lon: float, model=None):
    """
    Главная функция аудита локации.
    Возвращает: (score, ndvi, infrastructure_list)
    """
    # Инициализируем модель если не передана
    if model is None:
        import torch
        import os
        device = torch.device("cpu")
        model = CityVibeNet()
        MODEL_PATH = os.path.join("models", "cityvibe_model.pth")
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
    # 1. Получаем NDVI для разных радиусов
    ndvi_analysis = get_ndvi_multi_radius(lat, lon, radii=[100, 200, 300])
    
    # Берем среднее значение или приоритетное (например, 200м)
    ndvi_value = ndvi_analysis.get(200, 0.0)
    
    # 2. Получаем инфраструктуру в радиусе 15 минут ходьбы
    infra_data = get_cityvibe_input_data(lat, lon, ndvi_value)
    infrastructure_list = infra_data.get("infrastructure", [])
    
    # 3. Подготавливаем данные для нейросети
    objects_for_nn = [
        {"class": obj["class"], "walk_time": obj["walk_time"]}
        for obj in infrastructure_list
    ]
    
    # 4. Получаем предсказание от нейросети (0-100 баллов)
    score = predict_score(model, ndvi_value, objects_for_nn)
    
    # 5. Формируем список объектов для фронтенда
    formatted_infra = []
    for obj in infrastructure_list:
        class_id = obj["class"]
        type_name = CLASS_NAMES.get(class_id, "Объект")
        original_name = obj["name"]
        
        # Формируем отображаемое название
        # Если название совпадает с типом (например "Аптека") - показываем только тип
        # Иначе показываем "Тип (Название)"
        if original_name.lower() in [type_name.lower(), "без названия", ""]:
            display_name = type_name
        else:
            display_name = f"{type_name} ({original_name})"
        
        formatted_infra.append({
            "name": display_name,
            "type": type_name,
            "original_name": original_name,
            "class": class_id,
            "walk_time": obj["walk_time"],
            "lat": obj.get("lat"),  # Координаты объекта
            "lon": obj.get("lon")
        })
    
    return score, ndvi_value, formatted_infra


if __name__ == "__main__":
    # Тестовый запуск для Зеленого Квартала
    lat, lon = 51.27361102351295, 51.42923776746755
    
    print("🚀 Запуск аудита CityVibe AI...")
    score, ndvi, infra = evaluate_my_home(lat, lon)
    
    print(f"\n📊 Результаты:")
    print(f"   Балл CityVibe: {score}/100")
    print(f"   NDVI (озеленение): {ndvi:.3f} ({ndvi*100:.1f}%)")
    print(f"   Найдено объектов: {len(infra)}")
    
    if infra:
        print("\n🏪 Ближайшая инфраструктура:")
        for item in infra[:5]:
            print(f"   • {item['name']}: {item['walk_time']} мин")