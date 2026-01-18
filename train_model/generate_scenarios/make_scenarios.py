# save_scenarios.py
import random

def generate_synthetic_dataset(count=200):
    MIN_WALK_TIME = 2.0  
    categories = {
        "остановка": 10.0,
        "магазин": 15.0,
        "кафе": 15.0,
        "аптека": 18.0,
        "клиника": 25.0
    }

    CLASS_TO_INT = {cat: i for i, cat in enumerate(categories.keys())}

    dataset = []

    for i in range(count):
        ndvi = round(random.triangular(0.05, 0.7, 0.3), 3)
        objects = []

        for cat, max_limit in categories.items():
            if random.random() > 0.2:
                first_dist = round(random.uniform(MIN_WALK_TIME, max_limit), 1)
                objects.append({"class": CLASS_TO_INT[cat], "walk_time": first_dist})
                if random.random() > 0.6:
                    second_dist = round(first_dist + random.uniform(1.5, 7.0), 1)
                    if second_dist <= 25.0:
                        objects.append({"class": CLASS_TO_INT[cat], "walk_time": second_dist})

        objects.sort(key=lambda x: x["walk_time"])
        dataset.append({"ndvi": ndvi, "objects": objects})

    return dataset

if __name__ == "__main__":
    data = generate_synthetic_dataset(500)
    print("✅ Сгенерировано 500 примеров")
