import json
import os
import math
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Маппинг категорий в ID для твоей нейросети
CATEGORIES_MAP = {
    "магазин": 0,
    "аптека": 1,
    "кафе": 2,
    "клиника": 3,
    "остановка": 4
}

# Зеркала Overpass API (если основной сервер перегружен — пробуем следующий)
OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# OSRM серверы для расчета маршрутов пешком
OSRM_SERVERS = [
    "https://router.project-osrm.org",
    "https://routing.openstreetmap.de/routed-foot",
]

# Кэш результатов (чтобы не перегружать API при повторных запросах)
_cache = {}
_osrm_cache = {}  # Кэш для OSRM маршрутов
CACHE_TTL = 300  # 5 минут

def calculate_walk_time(lat1, lon1, lat2, lon2):
    """Вычисляет время пешком в минутах (80 м/мин) — по прямой, для fallback"""
    R = 6371000  # Радиус Земли в метрах
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    dist = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return round(dist / 80, 1)  # 80 м/мин средняя скорость ходьбы

def calculate_walk_times_osrm_batch(origin_lat, origin_lon, destinations):
    """
    Batch расчет времени пешком через OSRM table endpoint.
    destinations: список [(lat, lon), ...]
    Возвращает список времен в минутах или None если OSRM недоступен.
    NOTE: разбиваем на чанки по 50 объектов, иначе публичный OSRM не справляется
    """
    if not destinations:
        return []
    
    CHUNK_SIZE = 50  # Максимум 50 точек за один запрос
    all_times = []
    
    # Разбиваем на чанки
    for i in range(0, len(destinations), CHUNK_SIZE):
        chunk = destinations[i:i + CHUNK_SIZE]
        chunk_times = _calculate_walk_times_chunk(origin_lat, origin_lon, chunk)
        
        if chunk_times is None:
            # Если OSRM упал — fallback на прямое расстояние для этого чанка
            chunk_times = []
            for lat, lon in chunk:
                straight = calculate_walk_time(origin_lat, origin_lon, lat, lon)
                chunk_times.append(round(straight * 1.2, 1))
        
        all_times.extend(chunk_times)
    
    return all_times

def _calculate_walk_times_chunk(origin_lat, origin_lon, chunk):
    """
    Вспомогательная функция для одного чанка (до 50 точек).
    """
    if not chunk:
        return []
    
    # Формируем строку координат: origin;dest1;dest2;...
    coords_str = f"{origin_lon},{origin_lat}"
    for lat, lon in chunk:
        coords_str += f";{lon},{lat}"
    
    for server_url in OSRM_SERVERS:
        try:
            url = f"{server_url}/table/v1/foot/{coords_str}"
            params = {
                "sources": "0",
                "annotations": "duration"
            }
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "Ok" and data.get("durations"):
                    durations = data["durations"][0][1:]  # Пропускаем origin
                    return [round(d / 60, 1) if d else None for d in durations]
        except requests.exceptions.Timeout:
            print(f"⏱️ OSRM chunk таймаут от {server_url}")
            continue
        except Exception as e:
            print(f"⚠️ OSRM chunk ошибка {server_url}: {e}")
            continue
    
    return None  # Все серверы недоступны

def get_osm_data(lat, lon, radius=1200):
    """
    Получает актуальные данные из OpenStreetMap через Overpass API.
    Использует зеркала при сбоях и кэширует результаты.
    """
    # Проверяем кэш (округляем координаты до ~100м для попадания в кэш)
    cache_key = f"{round(lat, 3)}_{round(lon, 3)}"
    if cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if time.time() - cached_time < CACHE_TTL:
            print(f"📦 Инфраструктура из кэша ({len(cached_data)} объектов)")
            return cached_data

    # Запрос всех нужных категорий в радиусе
    query = f"""
    [out:json][timeout:30];
    (
      node["amenity"~"pharmacy|clinic|hospital|cafe|restaurant|fast_food"](around:{radius},{lat},{lon});
      way["amenity"~"pharmacy|clinic|hospital|cafe|restaurant|fast_food"](around:{radius},{lat},{lon});
      node["shop"~"supermarket|convenience|mall"](around:{radius},{lat},{lon});
      way["shop"~"supermarket|convenience|mall"](around:{radius},{lat},{lon});
      node["highway"="bus_stop"](around:{radius},{lat},{lon});
    );
    out center;
    """
    
    # Пробуем каждый сервер по очереди
    for server_url in OVERPASS_SERVERS:
        try:
            print(f"🌐 Запрос к {server_url.split('/')[2]}...")
            response = requests.post(server_url, data=query, timeout=25)
            if response.status_code == 200:
                elements = response.json().get('elements', [])
                # Сохраняем в кэш
                _cache[cache_key] = (time.time(), elements)
                print(f"✅ Получено {len(elements)} объектов из OSM")
                return elements
            else:
                print(f"⚠️ {server_url.split('/')[2]} вернул {response.status_code}, пробуем следующий...")
        except requests.exceptions.Timeout:
            print(f"⏱️ Таймаут от {server_url.split('/')[2]}, пробуем следующий...")
        except Exception as e:
            print(f"⚠️ Ошибка {server_url.split('/')[2]}: {e}")
    
    print("❌ Все сервера Overpass недоступны")
    return []

def get_cityvibe_input_data(lat, lon, ndvi_value):
    """
    Возвращает актуальный список объектов инфраструктуры в радиусе 15 минут ходьбы.
    Использует живые данные OSM + OSRM batch для реального времени пешком по дорогам.
    """
    elements = get_osm_data(lat, lon)
    results = []
    
    # Собираем координаты всех объектов с категориями
    destinations = []
    valid_elements = []
    
    for el in elements:
        el_lat = el.get('lat') or el.get('center', {}).get('lat')
        el_lon = el.get('lon') or el.get('center', {}).get('lon')
        
        if not el_lat or not el_lon:
            continue
        
        # Проверяем категорию сразу
        tags = el.get('tags', {})
        cat_name = None
        amenity = tags.get('amenity', '')
        shop = tags.get('shop', '')
        highway = tags.get('highway', '')
        
        if amenity in ['pharmacy']: 
            cat_name = "аптека"
        elif amenity in ['clinic', 'hospital', 'doctors', 'dentist']: 
            cat_name = "клиника"
        elif amenity in ['cafe', 'restaurant', 'fast_food', 'bar', 'pub']: 
            cat_name = "кафе"
        elif shop in ['supermarket', 'convenience', 'mall', 'grocery']: 
            cat_name = "магазин"
        elif highway == 'bus_stop' or tags.get('public_transport') == 'platform': 
            cat_name = "остановка"
        
        if cat_name:
            destinations.append((el_lat, el_lon))
            valid_elements.append((el, cat_name))
    
    if not destinations:
        return {"infrastructure": []}
    
    print(f"🚶 Batch расчет маршрутов через OSRM для {len(destinations)} объектов...")
    
    # Один batch запрос на все объекты
    osrm_times = calculate_walk_times_osrm_batch(lat, lon, destinations)
    
    # Если OSRM недоступен — fallback на прямое расстояние
    if osrm_times is None:
        print("⚠️ OSRM недоступен, используем прямое расстояние + 20%")
        osrm_times = []
        for dest_lat, dest_lon in destinations:
            straight_time = calculate_walk_time(lat, lon, dest_lat, dest_lon)
            osrm_times.append(round(straight_time * 1.2, 1))
    
    # Собираем результаты
    for i, (w_time, (el, cat_name)) in enumerate(zip(osrm_times, valid_elements)):
        if w_time is None or w_time > 15:
            continue
        
        tags = el.get('tags', {})
        el_lat, el_lon = destinations[i]
        
        name = tags.get('name') or tags.get('name:ru') or tags.get('name:en') or cat_name.capitalize()
        
        results.append({
            "name": name,
            "class": CATEGORIES_MAP[cat_name],
            "type": cat_name.capitalize(),
            "walk_time": w_time,
            "lat": el_lat,
            "lon": el_lon
        })
    
    print(f"✅ Обработано {len(results)} объектов в радиусе 15 минут")
    
    # Сортируем: сначала ближайшие
    results.sort(key=lambda x: x['walk_time'])
    
    # Лимит 20 объектов
    return {"infrastructure": results[:20]}


if __name__ == "__main__":
    # Тест для Зеленого Квартала (Астана)
    lat, lon = 51.128207, 71.430411 
    
    print(f"🔍 Запрос актуальных данных OSM для {lat}, {lon}...")
    try:
        data = get_cityvibe_input_data(lat, lon, 0.5)
        print(f"📊 Найдено объектов: {len(data['infrastructure'])}")
        for item in data['infrastructure'][:5]:
            print(f"   • {item['name']} ({item['type']}): {item['walk_time']} мин")
    except Exception as e:
        print(f"Ошибка: {e}")