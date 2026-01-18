import json
import os
import math

DB_PATH = os.path.join("data", "kazakhstan_infra.json")

# Маппинг категорий в ID для твоей нейросети
CATEGORIES_MAP = {
    "магазин": 0,
    "аптека": 1,
    "кафе": 2,
    "клиника": 3,
    "остановка": 4
}

def calculate_walk_time(lat1, lon1, lat2, lon2):
    """Вычисляет время пешком в минутах (80 м/мин)"""
    R = 6371000  # Радиус Земли в метрах
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    dist = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return round(dist / 80, 1)  # 80 м/мин средняя скорость ходьбы

def get_cityvibe_input_data(lat, lon, ndvi_value):
    """
    Возвращает список объектов инфраструктуры в радиусе 15 минут ходьбы.
    Формат: {"infrastructure": [{"name": str, "class": int, "walk_time": float}, ...]}
    """
    if not os.path.exists(DB_PATH):
        print(f"⚠️ Файл {DB_PATH} не найден!")
        return {"infrastructure": []}

    with open(DB_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    # Ограничиваем область поиска для скорости (~1.5 км)
    limit = 0.015 

    for feature in data.get('features', []):
        coords = feature['geometry']['coordinates']  # [lon, lat]
        
        # Быстрый фильтр (bounding box)
        if (lat - limit < coords[1] < lat + limit) and (lon - limit < coords[0] < lon + limit):
            w_time = calculate_walk_time(lat, lon, coords[1], coords[0])
            
            if w_time <= 15:  # Только объекты в 15 минутах ходьбы
                props = feature['properties']
                
                # Определяем категорию на основе OSM тегов
                cat_name = None
                if props.get('amenity') in ['pharmacy', 'chemist']: 
                    cat_name = "аптека"
                elif props.get('shop') in ['supermarket', 'convenience', 'mall']: 
                    cat_name = "магазин"
                elif props.get('amenity') in ['cafe', 'restaurant']: 
                    cat_name = "кафе"
                elif props.get('amenity') in ['clinic', 'hospital']: 
                    cat_name = "клиника"
                elif props.get('highway') == 'bus_stop': 
                    cat_name = "остановка"
                
                if cat_name:
                    results.append({
                        "name": props.get('name', cat_name.capitalize()),
                        "class": CATEGORIES_MAP[cat_name],
                        "walk_time": w_time,
                        "lat": coords[1],  # Широта
                        "lon": coords[0]   # Долгота
                    })

    # Сортируем по времени ходьбы (ближайшие первыми)
    results.sort(key=lambda x: x['walk_time'])
    
    # Берем максимум 15 ближайших объектов
    return {"infrastructure": results[:15]}


if __name__ == "__main__":
    # Тест для Зеленого Квартала
    lat, lon = 51.27361102351295, 51.42923776746755
    
    print("🔍 Поиск инфраструктуры вокруг координат...")
    result = get_cityvibe_input_data(lat, lon, 0.5)
    
    print(f"\n📊 Найдено объектов: {len(result['infrastructure'])}")
    
    if result['infrastructure']:
        print("\n🏪 Ближайшие объекты:")
        for obj in result['infrastructure'][:10]:
            emoji = {0: "🛒", 1: "💊", 2: "☕", 3: "🏥", 4: "🚌"}.get(obj['class'], "📍")
            print(f"   {emoji} {obj['name']}: {obj['walk_time']} мин (class={obj['class']})")