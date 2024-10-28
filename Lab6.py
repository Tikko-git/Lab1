import numpy as np

F = np.array([
    [8, 2, 4],  # x1
    [6, 7, 4],  # x2
    [4, 7, 5],  # x3
    [3, 5, 6]   # x4
])

# Ймовірності станів економічного середовища
P = np.array([1/2, 1/3, 1/6])

# 1. Обчислення критерію Баєса (середнього очікуваного прибутку)
bayes_criterion = np.dot(F, P)

# 2. Обчислення дисперсії для кожного проекту
mean_values = bayes_criterion  # Середні значення прибутку для кожного проекту
variances = np.dot((F - mean_values[:, np.newaxis]) ** 2, P)

for i in range(len(bayes_criterion)):
    print(f"Проект x{i+1}:")
    print(f"  Критерій Баєса (середній очікуваний прибуток): {bayes_criterion[i]:.2f}")
    print(f"  Дисперсія (рівень ризику): {variances[i]:.2f}")

# 3. Компромісне рішення: проект з максимальним значенням Баєса та мінімальною дисперсією
# Для простоти беремо середнє значення двох критеріїв як компроміс
compromise_values = (bayes_criterion - variances)  

best_project = np.argmax(compromise_values) + 1

print(f"Найкращий компромісний проект: x{best_project}")
