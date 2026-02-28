from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import requests
from typing import Optional
from pydantic import BaseModel, Field, validator

# Импорты из модулей проекта
from predict import CityVibeNet
from final import evaluate_my_home
from ai_concierge import get_city_explanation
import uvicorn
import random
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Модели валидации
class Coordinates(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Широта от -90 до 90")
    lon: float = Field(..., ge=-180, le=180, description="Долгота от -180 до 180")
    
    @validator('lat', 'lon')
    def validate_coordinates(cls, v):
        if v is None or (isinstance(v, float) and (v != v)):  # NaN check
            raise ValueError('Координаты не могут быть None или NaN')
        return v

app = FastAPI()

# Ключ Google Places API; если ключ не задан, используется режим OpenStreetMap
GOOGLE_PLACES_API_KEY = None

# Флаг выбора источника данных для place-details
USE_GOOGLE_PLACES = GOOGLE_PLACES_API_KEY is not None

# Настройки CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация модели
device = torch.device("cpu")
model = CityVibeNet()

MODEL_PATH = os.path.join("models", "cityvibe_model.pth")

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Model weights load error: {e}")
        model = None
else:
    logger.warning(f"Model file not found: {MODEL_PATH}. Run train_cityvibe.py first.")
    model = None

@app.get("/audit")
async def get_audit(lat: float, lon: float):
    """
    Эндпоинт для проведения полного урбанистического аудита.
    Принимает координаты, возвращает балл CityVibe, NDVI и список объектов.
    """
    try:
        # Валидация координат
        coords = Coordinates(lat=lat, lon=lon)
        
        if model is None:
            return {
                "status": "error",
                "message": "Модель не загружена. Проверьте файл модели."
            }
        
        # Расчет показателей и инфраструктуры выполняется в final.py
        score, ndvi, infra = evaluate_my_home(coords.lat, coords.lon, model) 
        
        # Форматирование на стороне final.py; здесь только сборка ответа
        formatted_infra = []
        for item in infra:
            formatted_infra.append({
                "name": item["name"],
                "type": item["type"],
                "class": item["class"],
                "walk_time": item["walk_time"],
                "lat": item.get("lat"),
                "lon": item.get("lon")
            })

        return {
            "status": "success",
            "score": score,
            "ndvi_percent": round(ndvi * 100, 1),
            "infrastructure": formatted_infra,
            "coords": [coords.lat, coords.lon],
            "user_type": "resident"
        }
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return {
            "status": "error",
            "message": f"Ошибка валидации: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Audit error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "message": str(e)
        }

@app.post("/explain")
async def explain_audit(data: dict):
    """
    Эндпоинт для объяснения результатов аудита с помощью LLM.
    """
    try:
        explanation = get_city_explanation(
            score=data.get("score"),
            ndvi=data.get("ndvi_percent"),
            infrastructure=data.get("infrastructure", []),
            user_type=data.get("user_type", "resident")
        )
        return {"status": "success", "explanation": explanation}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/listings")
async def get_listings(lat: float, lon: float):
    """
    Получает реальные жилые здания из OpenStreetMap (Overpass API).
    Здания типа residential, apartments, house, dormitory в радиусе 800м.
    """
    try:
        # Валидация координат
        coords = Coordinates(lat=lat, lon=lon)
        
        OVERPASS_SERVERS = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        ]
        query = f"""
        [out:json][timeout:25];
        (
          way["building"](around:800,{coords.lat},{coords.lon});
          relation["building"](around:800,{coords.lat},{coords.lon});
        );
        out center geom;
        """
        elements = []
        for server_url in OVERPASS_SERVERS:
            try:
                response = requests.post(server_url, data=query, timeout=20)
                if response.status_code == 200:
                    elements = response.json().get("elements", [])
                    break
            except Exception:
                continue

        listings = []
        for el in elements[:60]: 
            tags = el.get('tags', {})
            b_type = tags.get('building', 'yes')
            
            # Получаем координаты центра
            if el.get('center'):
                el_lat = el['center']['lat']
                el_lon = el['center']['lon']
            elif el.get('lat') and el.get('lon'):
                el_lat = el['lat']
                el_lon = el['lon']
            else:
                geom = el.get('geometry', [])
                if not geom: continue
                el_lat = sum(p['lat'] for p in geom) / len(geom)
                el_lon = sum(p['lon'] for p in geom) / len(geom)
            
            # Получаем геометрию
            poly_geom = []
            if el.get('type') == 'way':
                poly_geom = [[p['lat'], p['lon']] for p in el.get('geometry', [])]
            elif el.get('type') == 'relation':
                members = el.get('members', [])
                max_points = 0
                for member in members:
                    if member.get('type') == 'way' and member.get('geometry'):
                        points = member.get('geometry')
                        if len(points) > max_points:
                            poly_geom = [[p['lat'], p['lon']] for p in points]
                            max_points = len(points)
            
            if not poly_geom: continue
            
            floors = tags.get('building:levels', '?')
            street = tags.get('addr:street', '')
            house_num = tags.get('addr:housenumber', '')
            address = f"{street} {house_num}".strip() or None
            
            # Моковая ссылка на объявление
            mock_url = f"https://krisha.kz/search/prodazha/kvartiry/?lat={el_lat}&lon={el_lon}"

            listings.append({
                "id": el.get('id', len(listings)),
                "title": f"Объект • {floors} эт." if floors != '?' else "Здание",
                "price": f"{random.randint(15, 80)} млн ₸",
                "lat": el_lat,
                "lon": el_lon,
                "score": random.randint(40, 95),
                "is_residential": b_type in {'residential', 'apartments', 'house', 'dormitory', 'flat', 'yes'},
                "building_type": b_type,
                "floors": floors,
                "address": address,
                "geometry": poly_geom,
                "url": mock_url
            })

        # Fallback если OSM ничего не нашёл (упрощенный без геометрии для фейковых данных)
        if not listings:
            for i in range(5):
                d_lat = (random.random() - 0.5) * 0.009
                d_lon = (random.random() - 0.5) * 0.009
                fake_lat = coords.lat + d_lat
                fake_lon = coords.lon + d_lon
                mock_url = f"https://krisha.kz/search/prodazha/kvartiry/?lat={fake_lat}&lon={fake_lon}"
                listings.append({
                    "id": i,
                    "title": f"Квартира {random.randint(1, 4)}-комн.",
                    "price": f"{random.randint(15, 60)} млн ₸",
                    "lat": fake_lat,
                    "lon": fake_lon,
                    "score": random.randint(40, 95),
                    "is_residential": True,
                    "geometry": [],
                    "url": mock_url
                })

        return {"status": "success", "listings": listings}
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return {
            "status": "error",
            "message": f"Ошибка валидации: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Listings error: {e}")
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
        # Валидация координат
        coords = Coordinates(lat=lat, lon=lon)
        
        if USE_GOOGLE_PLACES:
            return await get_place_details_google(name, coords.lat, coords.lon)
        else:
            return await get_place_details_osm(name, coords.lat, coords.lon)
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return {
            "status": "error",
            "message": f"Ошибка валидации: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Place details error: {e}")
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
        # Поиск объекта в OSM по координатам
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
            element_name = tags.get("name", "").lower()
            if name.lower() in element_name or element_name in name.lower():
                place_data = tags
                break
        
        if not place_data:
            # Fallback: первый объект с заполненным именем
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
                    "address": f"{lat:.5f}, {lon:.5f}",
                    "opening_hours": [],
                    "photo_url": None,
                    "reviews": []
                }
            }
        
        # Сборка ответа
        place_info = {
            "name": place_data.get("name", name),
            "rating": None,
            "user_ratings_total": None,
            "address": place_data.get("addr:full") or f"{place_data.get('addr:street', '')} {place_data.get('addr:housenumber', '')}".strip() or None,
            "opening_hours": [],
            "photo_url": None,
            "reviews": []
        }
        
        # Часы работы в формате OSM
        opening_hours = place_data.get("opening_hours")
        if opening_hours:
            place_info["opening_hours"] = [opening_hours]
        
        # Дополнительные поля (если есть)
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
        print(f"OSM request error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

async def get_place_details_google(name: str, lat: float, lon: float):
    """
    Получает детальную информацию из Google Places API
    """
    try:
        # Поиск места по названию и координатам
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
        
        # Запрос детальной информации
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
        
        # Сборка ответа
        place_info = {
            "name": result.get("name"),
            "rating": result.get("rating"),
            "user_ratings_total": result.get("user_ratings_total"),
            "address": result.get("formatted_address"),
            "opening_hours": result.get("opening_hours", {}).get("weekday_text", []),
            "photo_url": None,
            "reviews": []
        }
        
        # URL первой фотографии
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
    # Запуск сервера на localhost:8000
    print("Starting CityVibe server...")
    print("API: http://127.0.0.1:8000")
    print("Docs: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)