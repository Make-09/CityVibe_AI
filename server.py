from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import requests

# Импортируем архитектуру и функции из твоих файлов
from predict import CityVibeNet
from final import evaluate_my_home
import uvicorn

app = FastAPI()

# Google Places API Key (замени на свой или оставь None для OSM-режима)
GOOGLE_PLACES_API_KEY = None  # Получи на https://console.cloud.google.com

# Если ключа нет, используем данные из OpenStreetMap
USE_GOOGLE_PLACES = GOOGLE_PLACES_API_KEY is not None

# Разрешаем фронтенду (index.html) подключаться к серверу
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Инициализация нейросети ---
device = torch.device("cpu")
model = CityVibeNet()

MODEL_PATH = "cityvibe_model.pth"

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"✅ Модель CityVibe AI успешно загружена из {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Ошибка при загрузке весов модели: {e}")
else:
    print(f"⚠️ ВНИМАНИЕ: Файл {MODEL_PATH} не найден. Сначала запусти train_cityvibe.py")

@app.get("/audit")
async def get_audit(lat: float, lon: float):
    """
    Эндпоинт для проведения полного урбанистического аудита.
    Принимает координаты, возвращает балл CityVibe, NDVI и список объектов.
    """
    try:
        # Вызываем логику из final.py
        # Теперь infra уже содержит форматированные названия типа "Магазин (Экстра)"
        score, ndvi, infra = evaluate_my_home(lat, lon) 
        
        # Передаем данные напрямую - форматирование уже сделано в final.py
        formatted_infra = []
        for item in infra:
            formatted_infra.append({
                "name": item["name"],  # Уже отформатированное название
                "type": item["type"],
                "class": item["class"],
                "walk_time": item["walk_time"],
                "lat": item.get("lat"),  # Координаты объекта
                "lon": item.get("lon")
            })

        return {
            "status": "success",
            "score": score,
            "ndvi_percent": round(ndvi * 100, 1),
            "infrastructure": formatted_infra,
            "coords": [lat, lon]
        }
    except Exception as e:
        print(f"🔥 Ошибка при аудите: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "message": str(e)
        }

@app.get("/")
async def root():
    """Проверка работы сервера"""
    return {"status": "CityVibe AI Server Running", "version": "1.0"}

@app.get("/place-details")
async def get_place_details(name: str, lat: float, lon: float):
    """
    Получает детальную информацию о месте.
    Использует Google Places API если доступен, иначе OpenStreetMap.
    """
    try:
        if USE_GOOGLE_PLACES:
            return await get_place_details_google(name, lat, lon)
        else:
            return await get_place_details_osm(name, lat, lon)
    except Exception as e:
        print(f"🔥 Ошибка при получении деталей места: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }

async def get_place_details_osm(name: str, lat: float, lon: float):
    """
    Получает информацию из OpenStreetMap (бесплатно, без API ключа)
    """
    try:
        # Ищем объект в OSM по координатам
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        (
          node(around:50,{lat},{lon});
          way(around:50,{lat},{lon});
        );
        out body;
        """
        
        response = requests.post(overpass_url, data=query, timeout=10)
        data = response.json()
        
        # Ищем подходящий объект
        place_data = None
        for element in data.get("elements", []):
            tags = element.get("tags", {})
            if tags.get("name", "").lower() in name.lower() or name.lower() in tags.get("name", "").lower():
                place_data = tags
                break
        
        if not place_data:
            # Берем первый объект с именем
            for element in data.get("elements", []):
                if element.get("tags", {}).get("name"):
                    place_data = element.get("tags", {})
                    break
        
        if not place_data:
            return {
                "status": "success",
                "place": {
                    "name": name,
                    "rating": None,
                    "address": f"📍 {lat:.5f}, {lon:.5f}",
                    "opening_hours": [],
                    "photo_url": None,
                    "reviews": []
                }
            }
        
        # Формируем информацию
        place_info = {
            "name": place_data.get("name", name),
            "rating": None,
            "user_ratings_total": None,
            "address": place_data.get("addr:full") or f"{place_data.get('addr:street', '')} {place_data.get('addr:housenumber', '')}".strip() or None,
            "opening_hours": [],
            "photo_url": None,
            "reviews": []
        }
        
        # Парсим часы работы из OSM
        opening_hours = place_data.get("opening_hours")
        if opening_hours:
            # OSM формат: Mo-Fr 08:00-18:00
            place_info["opening_hours"] = [opening_hours]
        
        # Добавляем дополнительную информацию
        if place_data.get("phone"):
            place_info["phone"] = place_data.get("phone")
        if place_data.get("website"):
            place_info["website"] = place_data.get("website")
        
        return {
            "status": "success",
            "place": place_info,
            "source": "OpenStreetMap"
        }
        
    except Exception as e:
        print(f"Ошибка OSM запроса: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

async def get_place_details_google(name: str, lat: float, lon: float):
    """
    Получает детальную информацию из Google Places API
    """
    try:
        # 1. Ищем место по названию и координатам
        search_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        search_params = {
            "location": f"{lat},{lon}",
            "radius": 100,
            "keyword": name,
            "language": "ru",
            "key": GOOGLE_PLACES_API_KEY
        }
        
        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_data = search_response.json()
        
        if search_data.get("status") != "OK" or not search_data.get("results"):
            return {
                "status": "error",
                "message": "Место не найдено в Google Places"
            }
        
        place_id = search_data["results"][0]["place_id"]
        
        # 2. Получаем детали места
        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            "place_id": place_id,
            "fields": "name,rating,user_ratings_total,formatted_address,opening_hours,photos,reviews",
            "language": "ru",
            "key": GOOGLE_PLACES_API_KEY
        }
        
        details_response = requests.get(details_url, params=details_params, timeout=10)
        details_data = details_response.json()
        
        if details_data.get("status") != "OK":
            return {
                "status": "error",
                "message": "Не удалось получить детали места"
            }
        
        result = details_data.get("result", {})
        
        # 3. Формируем ответ
        place_info = {
            "name": result.get("name"),
            "rating": result.get("rating"),
            "user_ratings_total": result.get("user_ratings_total"),
            "address": result.get("formatted_address"),
            "opening_hours": result.get("opening_hours", {}).get("weekday_text", []),
            "photo_url": None,
            "reviews": []
        }
        
        # Получаем URL первой фотографии
        if result.get("photos"):
            photo_reference = result["photos"][0].get("photo_reference")
            place_info["photo_url"] = (
                f"https://maps.googleapis.com/maps/api/place/photo"
                f"?maxwidth=600&photo_reference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
            )
        
        # Первые 3 отзыва
        if result.get("reviews"):
            place_info["reviews"] = [
                {
                    "author_name": r.get("author_name"),
                    "rating": r.get("rating"),
                    "text": r.get("text", "")[:200] + ("..." if len(r.get("text", "")) > 200 else "")
                }
                for r in result["reviews"][:3]
            ]
        
        return {
            "status": "success",
            "place": place_info,
            "source": "Google Places"
        }
        
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "message": "Таймаут запроса к Google Places API"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    # Запуск сервера на локальном хосте, порт 8000
    print("🚀 Запуск сервера CityVibe AI...")
    print("📍 API доступен на http://127.0.0.1:8000")
    print("📖 Документация: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)