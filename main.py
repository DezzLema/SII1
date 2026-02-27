import random
import matplotlib.pyplot as plt
from deap import base, creator, tools
import collections
import itertools
import random

# Производители: (название, производственная мощность y)
producers = [
    ("Москва",         1500),
    ("Санкт-Петербург", 900),
    ("Екатеринбург",   1000),
    ("Новосибирск",     800),
]

# Города: (название, спрос x)
cities = [
    ("Краснодар",   400),
    ("Воронеж",     350),
    ("Пермь",       300),
    ("Красноярск",  320),
    ("Саратов",     250),
]

n = len(producers)   # количество производителей
k = len(cities)      # количество городов

# Матрица расстояний: distances[индекс_производителя][индекс_города] в км (примерные реальные расстояния)
distances = [
    # Москва
    [1100,  500, 1400, 4000,  850],
    # Санкт-Петербург
    [1700, 1100, 2000, 4500, 1400],
    # Екатеринбург
    [2200, 1800,  350, 1800, 1400],
    # Новосибирск
    [3400, 2800, 1400,  800, 2500],
]

# Общий спрос всех городов
total_demand = sum(x for _, x in cities)

# Ограничения
MAX_COST = 1000000   # Произвольный верхний предел транспортных затрат (можно подстраивать)
PENALTY = 100000     # Большой штраф за нарушения ограничений


# Функция приспособленности (fitness)
def evaluate(individual):
    assigned = collections.defaultdict(int)  # сколько отправлено с каждого производителя
    cost = 0
    for city_idx, prod_idx in enumerate(individual):
        assigned[prod_idx] += cities[city_idx][1]           # добавляем спрос города
        cost += distances[prod_idx][city_idx] * cities[city_idx][1]  # транспортные расходы

    penalty = 0
    used_prods = set(assigned.keys())  # какие производители задействованы
    excess = sum(producers[p][1] for p in used_prods) - total_demand  # избыток мощности

    # Штраф за превышение мощности любого производителя
    for prod, used in assigned.items():
        if used > producers[prod][1]:
            penalty += PENALTY

    # Штраф, если суммарная мощность задействованных заводов меньше спроса
    if excess < 0:
        penalty += PENALTY

    # Штраф за слишком большие транспортные расходы
    if cost > MAX_COST:
        penalty += PENALTY

    # Итоговая приспособленность: минимизируем стоимость + штраф за избыток + жёсткие штрафы
    fitness = cost + 2000 * max(0, excess) + penalty
    return (fitness,)


# Настройка DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))           # минимизация
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, n - 1)                # ген — номер производителя
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=k)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)          # турнирный отбор

# Операторы скрещивания
crossovers = {
    "One Point": tools.cxOnePoint,                                    # одноточечное
    "Two Points": tools.cxTwoPoint,                                   # двухточечное
    "Uniform": lambda ind1, ind2: tools.cxUniform(ind1, ind2, indpb=0.5)  # равномерное
}

# Операторы мутации (с разной вероятностью изменения гена)
mutations = {
    "UniformInt 5%":  lambda ind: tools.mutUniformInt(ind, low=0, up=n-1, indpb=0.05),
    "UniformInt 10%": lambda ind: tools.mutUniformInt(ind, low=0, up=n-1, indpb=0.1),
    "UniformInt 20%": lambda ind: tools.mutUniformInt(ind, low=0, up=n-1, indpb=0.2),
}

# Параметры генетического алгоритма
POP_SIZE = 60        # размер популяции
GENERATIONS = 80     # количество поколений
CXPB = 0.8           # вероятность скрещивания
MUTPB = 0.2          # вероятность мутации особи

# Эксперименты с разными комбинациями скрещивания и мутации
results = {}
for cx_name, cx_op in crossovers.items():
    for mut_name, mut_op in mutations.items():
        toolbox.register("mate", cx_op)
        toolbox.register("mutate", mut_op)

        population = toolbox.population(n=POP_SIZE)
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        best_history = []  # история лучшего значения в каждом поколении
        for gen in range(GENERATIONS):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Скрещивание
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Мутация
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Оценка новых особей
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring
            best = min(population, key=lambda ind: ind.fitness.values[0]).fitness.values[0]
            best_history.append(best)

        results[f"{cx_name} + {mut_name}"] = best_history

# Построение графика
plt.figure(figsize=(13, 8))
for label, history in results.items():
    plt.plot(history, label=label)
plt.xlabel("Поколение")
plt.ylabel("Приспособленность (fitness)")
plt.title("Сравнение операторов скрещивания и мутации")
plt.legend(fontsize=8)
plt.grid()
plt.show()

# Полный перебор для проверки оптимального решения
min_fitness = float('inf')
best_assignment = None
for assignment in itertools.product(range(n), repeat=k):
    ind = list(assignment)
    fitness = evaluate(ind)[0]
    if fitness < min_fitness:
        min_fitness = fitness
        best_assignment = ind

print("Оптимальное значение fitness (полный перебор):", min_fitness)
print("Оптимальное распределение (индексы производителей для городов):", best_assignment)

# Подробности лучшего решения
assigned = collections.defaultdict(int)
cost = 0
for city_idx, prod_idx in enumerate(best_assignment):
    assigned[prod_idx] += cities[city_idx][1]
    cost += distances[prod_idx][city_idx] * cities[city_idx][1]

used_prods = set(assigned.keys())
excess = sum(producers[p][1] for p in used_prods) - total_demand
print("Оптимальная транспортная стоимость:", cost)
print("Избыток мощности:", excess)
print("Распределённый спрос по производителям:", dict(assigned))