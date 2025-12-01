
import numpy as np
import pandas as pd

np.random.seed(42)

n = 2000
# Features typical of nutrition data (per 100g)
calories = np.random.normal(150, 90, n).clip(10, 900)
protein = np.random.normal(5, 6, n).clip(0, 100)
carbs = np.random.normal(20, 15, n).clip(0, 200)
fat = np.random.normal(8, 10, n).clip(0, 100)
sugar = np.random.normal(6, 8, n).clip(0, 120)
fiber = np.random.normal(2, 3, n).clip(0, 50)
sodium = np.random.normal(200, 300, n).clip(0, 5000)

# Simple label logic for synthetic data: 4 classes
labels = []
for c,p,k,f,sb,fi,so in zip(calories, protein, carbs, fat, sugar, fiber, sodium):
    if p > 15 and c > 150:
        labels.append("Protein-Rich")
    elif k > 30 and sb > 10:
        labels.append("Carbohydrate-Rich")
    elif f > 20:
        labels.append("Fat-Rich")
    else:
        labels.append("Low-Calorie")

df = pd.DataFrame({
    "calories": calories,
    "protein": protein,
    "carbs": carbs,
    "fat": fat,
    "sugar": sugar,
    "fiber": fiber,
    "sodium": sodium,
    "label": labels
})

df.to_csv("C:\\Nutri\\src\\data\\synthetic_food_dataset_imbalanced.csv", index=False)
print("Saved C:\\Nutri\\src\\data\\synthetic_food_dataset_imbalanced.csv with", len(df), "rows")