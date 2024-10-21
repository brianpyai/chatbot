import random
import math
import numpy as np
from collections import defaultdict

# 1. 決策樹
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_decision_tree(X, y, max_depth=3):
    if max_depth == 0 or len(set(y)) == 1:
        return DecisionTreeNode(value=max(set(y), key=y.count))
    
    best_feature, best_threshold = find_best_split(X, y)
    left_indices = [i for i in range(len(X)) if X[i][best_feature] <= best_threshold]
    right_indices = [i for i in range(len(X)) if X[i][best_feature] > best_threshold]
    
    if not left_indices or not right_indices:
        return DecisionTreeNode(value=max(set(y), key=y.count))
    
    left = build_decision_tree([X[i] for i in left_indices], [y[i] for i in left_indices], max_depth-1)
    right = build_decision_tree([X[i] for i in right_indices], [y[i] for i in right_indices], max_depth-1)
    
    return DecisionTreeNode(feature=best_feature, threshold=best_threshold, left=left, right=right)

def find_best_split(X, y):
    best_feature = random.randint(0, len(X[0])-1)
    best_threshold = sum(x[best_feature] for x in X) / len(X)
    return best_feature, best_threshold

def predict(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict(node.left, x)
    else:
        return predict(node.right, x)

def test_decision_tree():
    X = [[1, 2], [2, 3], [3, 1], [4, 4]]
    y = [0, 0, 1, 1]
    tree = build_decision_tree(X, y)
    print("決策樹測試:")
    print("預測 [1.5, 2.5]:", predict(tree, [1.5, 2.5]))
    print("預測 [3.5, 3.5]:", predict(tree, [3.5, 3.5]))

# 2. 遺傳算法
def genetic_algorithm(population, fitness_func, generations, mutation_rate=0.01):
    for _ in range(generations):
        new_population = []
        for _ in range(len(population)):
            parent1 = tournament_selection(population, fitness_func)
            parent2 = tournament_selection(population, fitness_func)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)
        population = new_population
    return max(population, key=fitness_func)

def tournament_selection(population, fitness_func, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=fitness_func)

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1)-1)
    return parent1[:crossover_point] + parent2[crossover_point:]

def mutate(individual):
    mutation_point = random.randint(0, len(individual)-1)
    individual[mutation_point] = random.randint(0, 1)
    return individual

def test_genetic_algorithm():
    def fitness(individual):
        return sum(individual)

    population = [[random.randint(0, 1) for _ in range(10)] for _ in range(50)]
    best = genetic_algorithm(population, fitness, generations=100)
    print("\n遺傳算法測試:")
    print("最佳個體:", best)
    print("適應度:", fitness(best))

# 3. K-means聚類
def kmeans(X, k, max_iterations=100):
    centroids = random.sample(X, k)
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for x in X:
            closest_centroid = min(range(k), key=lambda i: euclidean_distance(x, centroids[i]))
            clusters[closest_centroid].append(x)
        new_centroids = [vector_mean(cluster) for cluster in clusters if cluster]
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return centroids, clusters

def euclidean_distance(v1, v2):
    return math.sqrt(sum((a-b)**2 for a, b in zip(v1, v2)))

def vector_mean(vectors):
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(len(vectors[0]))]

def test_kmeans():
    X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]
    k = 2
    centroids, clusters = kmeans(X, k)
    print("\nK-means聚類測試:")
    print("中心點:", centroids)
    print("簇:", clusters)

# 4. 專家系統
class ExpertSystem:
    def __init__(self):
        self.rules = []

    def add_rule(self, condition, action):
        self.rules.append((condition, action))

    def infer(self, facts):
        for condition, action in self.rules:
            if all(fact in facts for fact in condition):
                return action
        return None

def test_expert_system():
    expert_system = ExpertSystem()
    expert_system.add_rule(["發燒", "咳嗽"], "可能是感冒")
    expert_system.add_rule(["頭痛", "噁心"], "可能是偏頭痛")

    print("\n專家系統測試:")
    print("症狀: 發燒, 咳嗽 - 診斷:", expert_system.infer(["發燒", "咳嗽"]))
    print("症狀: 頭痛, 噁心 - 診斷:", expert_system.infer(["頭痛", "噁心"]))
    print("症狀: 發燒 - 診斷:", expert_system.infer(["發燒"]))

# 5. 隨機森林
def random_forest(X, y, n_trees=10, max_depth=3):
    forest = []
    for _ in range(n_trees):
        indices = random.choices(range(len(X)), k=len(X))
        X_subset = [X[i] for i in indices]
        y_subset = [y[i] for i in indices]
        tree = build_decision_tree(X_subset, y_subset, max_depth)
        forest.append(tree)
    return forest

def predict_forest(forest, x):
    predictions = [predict(tree, x) for tree in forest]
    return max(set(predictions), key=predictions.count)

def test_random_forest():
    X = [[1, 2], [2, 3], [3, 1], [4, 4], [2, 2], [3, 3]]
    y = [0, 0, 1, 1, 0, 1]
    forest = random_forest(X, y, n_trees=5)
    print("\n隨機森林測試:")
    print("預測 [1.5, 2.5]:", predict_forest(forest, [1.5, 2.5]))
    print("預測 [3.5, 3.5]:", predict_forest(forest, [3.5, 3.5]))

# 6. 蒙特卡羅模擬
def monte_carlo_simulation(num_simulations, probability_func):
    successes = 0
    for _ in range(num_simulations):
        if probability_func():
            successes += 1
    return successes / num_simulations

def estimate_pi():
    def is_in_circle():
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        return x*x + y*y <= 1
    
    return 4 * monte_carlo_simulation(1000000, is_in_circle)

def test_monte_carlo():
    estimated_pi = estimate_pi()
    print("\n蒙特卡羅模擬測試:")
    print(f"估算的π值: {estimated_pi}")
    print(f"與實際π值的誤差: {abs(estimated_pi - math.pi)}")

# 7. 貝葉斯定理
def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

def test_bayes_theorem():
    prior = 0.01
    likelihood = 0.95
    false_positive_rate = 0.05
    evidence = likelihood * prior + false_positive_rate * (1 - prior)

    posterior = bayes_theorem(prior, likelihood, evidence)
    
    print("\n貝葉斯定理測試:")
    print(f"測試呈陽性時實際患病的概率: {posterior:.4f}")

# 8. Q-learning
def q_learning(states, actions, rewards, learning_rate=0.1, discount_factor=0.9, episodes=1000):
    Q = {(s, a): 0 for s in states for a in actions}
    
    for _ in range(episodes):
        state = random.choice(states)
        while state is not None:
            action = max(actions, key=lambda a: Q[(state, a)])
            next_state = random.choice(states + [None])
            reward = rewards.get((state, action, next_state), 0)
            
            if next_state is None:
                Q[(state, action)] += learning_rate * (reward - Q[(state, action)])
            else:
                best_next_action = max(actions, key=lambda a: Q[(next_state, a)])
                Q[(state, action)] += learning_rate * (reward + discount_factor * Q[(next_state, best_next_action)] - Q[(state, action)])
            
            state = next_state
    
    return Q

def test_q_learning():
    states = ['A', 'B', 'C']
    actions = ['left', 'right']
    rewards = {
        ('A', 'right', 'B'): 1,
        ('B', 'right', 'C'): 2,
        ('C', 'left', 'B'): 1,
        ('B', 'left', 'A'): 0,
    }

    Q = q_learning(states, actions, rewards)
    
    print("\nQ-learning測試:")
    for state in states:
        best_action = max(actions, key=lambda a: Q[(state, a)])
        print(f"在狀態 {state} 的最佳行動: {best_action}")

# 9. 模糊邏輯
class FuzzySet:
    def __init__(self, membership_func):
        self.membership_func = membership_func
    
    def membership(self, x):
        return self.membership_func(x)

def fuzzy_and(a, b):
    return min(a, b)

def fuzzy_or(a, b):
    return max(a, b)

def fuzzy_not(a):
    return 1 - a

def test_fuzzy_logic():
    cold = FuzzySet(lambda x: max(0, min(1, (20 - x) / 10)))
    warm = FuzzySet(lambda x: max(0, min(1, (x - 15) / 10, (35 - x) / 10)))
    hot = FuzzySet(lambda x: max(0, min(1, (x - 30) / 10)))

    temp = 25
    print("\n模糊邏輯測試:")
    print(f"溫度 {temp}°C:")
    print(f"冷的程度: {cold.membership(temp):.2f}")
    print(f"溫暖的程度: {warm.membership(temp):.2f}")
    print(f"熱的程度: {hot.membership(temp):.2f}")

    not_cold_and_not_hot = fuzzy_and(fuzzy_not(cold.membership(temp)), fuzzy_not(hot.membership(temp)))
    print(f"不冷且不熱的程度: {not_cold_and_not_hot:.2f}")

# 10. 演化算法 (Differential Evolution)
def differential_evolution(objective_func, bounds, population_size=50, F=0.8, CR=0.7, generations=1000):
    dimensions = len(bounds)
    population = [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)] for _ in range(population_size)]
    
    for _ in range(generations):
        for i in range(population_size):
            a, b, c = random.sample([p for j, p in enumerate(population) if j != i], 3)
            mutant = [0] * dimensions
            R = random.randint(0, dimensions - 1)
            for j in range(dimensions):
                if random.random() < CR or j == R:
                    mutant[j] = a[j] + F * (b[j] - c[j])
                else:
                    mutant[j] = population[i][j]
                mutant[j] = max(min(mutant[j], bounds[j][1]), bounds[j][0])
            
            if objective_func(mutant) < objective_func(population[i]):
                population[i] = mutant
    
    return min(population, key=objective_func)

def test_differential_evolution():
    def objective_func(x):
        return x[0]**2 + x[1]**2

    bounds = [(-5, 5), (-5, 5)]
    result = differential_evolution(objective_func, bounds)
    
    print("\n演化算法 (Differential Evolution) 測試:")
    print(f"找到的最優解: x = {result[0]:.4f}, y = {result[1]:.4f}")
    print(f"目標函數值: {objective_func(result):.4f}")

# 11. 自然語言處理 - 詞頻分析
def word_frequency(text):
    words = text.lower().split()
    frequency = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1
    return frequency

def test_word_frequency():
    text = "這是一個測試文本。這個文本用於測試詞頻分析。分析文本中的詞頻是自然語言處理的基礎任務之一。"
    freq = word_frequency(text)
    
    print("\n詞頻分析測試:")
    for word, count in sorted(freq.items(), key=lambda x: x[1], reverse=True):
        print(f"{word}: {count}")

# 12. 圖像處理 - 邊緣檢測 (Sobel 算子)
def sobel_edge_detection(image):
    height, width = len(image), len(image[0])
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            px = sum(sobel_x[di+1][dj+1] * image[i+di][j+dj] for di in [-1, 0, 1] for dj in [-1, 0, 1])
            py = sum(sobel_y[di+1][dj+1] * image[i+di][j+dj] for di in [-1, 0, 1] for dj in [-1, 0, 1])
            output[i][j] = min(int(math.sqrt(px*px + py*py)), 255)
    
    return output

def test_edge_detection():
    image = [
        [0, 0, 0, 0, 0],
        [0, 100, 100, 100, 0],
        [0, 100, 100, 100, 0],
        [0, 100, 100, 100, 0],
        [0, 0, 0, 0, 0]
    ]
    
    edges = sobel_edge_detection(image)
    
    print("\n邊緣檢測測試:")
    print("原始圖像:")
    for row in image:
        print(' '.join(f"{pixel:3d}" for pixel in row))
    
    print("\n邊緣檢測結果:")
    for row in edges:
        print(' '.join(f"{pixel:3d}" for pixel in row))

# 13. 啟發式算法 - A*搜索
class Node:
    def __init__(self, position, g=0, h=0, parent=None):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    start_node = Node(start)
    open_list = [start_node]
    closed_set = set()

    while open_list:
        current_node = min(open_list, key=lambda x: x.f)
        open_list.remove(current_node)

        if current_node.position == goal:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current_node.position)

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor_pos = (current_node.position[0] + dx, current_node.position[1] + dy)
            if (0 <= neighbor_pos[0] < len(grid) and 
                0 <= neighbor_pos[1] < len(grid[0]) and 
                grid[neighbor_pos[0]][neighbor_pos[1]] == 0 and 
                neighbor_pos not in closed_set):
                
                neighbor = Node(neighbor_pos, current_node.g + 1, heuristic(neighbor_pos, goal), current_node)
                
                if neighbor not in open_list:
                    open_list.append(neighbor)
                else:
                    idx = open_list.index(neighbor)
                    if neighbor.g < open_list[idx].g:
                        open_list[idx] = neighbor

    return None

def test_astar():
    grid = [
        [0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0]
    ]
    start = (0, 0)
    goal = (4, 4)
    path = astar(grid, start, goal)
    print("\nA*搜索測試:")
    print("網格:")
    for row in grid:
        print(row)
    print("起點:", start)
    print("終點:", goal)
    print("找到的路徑:", path)

# 14. 貝葉斯網絡
class BayesianNetwork:
    def __init__(self):
        self.network = {}

    def add_node(self, node, parents, cpt):
        self.network[node] = {'parents': parents, 'cpt': cpt}

    def probability(self, node, value, evidence):
        if not self.network[node]['parents']:
            return self.network[node]['cpt'][value]
        
        parent_values = tuple(evidence[parent] for parent in self.network[node]['parents'])
        return self.network[node]['cpt'][parent_values][value]

def test_bayesian_network():
    bn = BayesianNetwork()

    bn.add_node('Rain', [], {'True': 0.2, 'False': 0.8})
    bn.add_node('Sprinkler', ['Rain'], {
        ('True',): {'True': 0.01, 'False': 0.99},
        ('False',): {'True': 0.4, 'False': 0.6}
    })
    bn.add_node('GrassWet', ['Rain', 'Sprinkler'], {
        ('True', 'True'): {'True': 0.99, 'False': 0.01},
        ('True', 'False'): {'True': 0.8, 'False': 0.2},
        ('False', 'True'): {'True': 0.9, 'False': 0.1},
        ('False', 'False'): {'True': 0.0, 'False': 1.0}
    })

    print("\n貝葉斯網絡測試:")
    evidence = {'Rain': 'False', 'Sprinkler': 'True'}
    prob_grass_wet = bn.probability('GrassWet', 'True', evidence)
    print(f"P(GrassWet=True | Rain=False, Sprinkler=True) = {prob_grass_wet}")

# 15. 神經網絡 (簡單的前向傳播網絡)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

def test_neural_network():
    np.random.seed(0)
    nn = SimpleNeuralNetwork(2, 3, 1)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    print("\n神經網絡測試:")
    for x in X:
        output = nn.forward(x.reshape(1, -1))
        print(f"輸入: {x}, 輸出: {output[0][0]:.4f}")

# 16. 符號AI (簡單的專家系統)
class SymbolicAI:
    def __init__(self):
        self.rules = []

    def add_rule(self, condition, conclusion):
        self.rules.append((condition, conclusion))

    def infer(self, facts):
        conclusions = set()
        for condition, conclusion in self.rules:
            if all(fact in facts for fact in condition):
                conclusions.add(conclusion)
        return conclusions

def test_symbolic_ai():
    expert_system = SymbolicAI()
    expert_system.add_rule(("下雨", "地面濕"), "樹葉濕")
    expert_system.add_rule(("下雨",), "雲層厚")
    expert_system.add_rule(("雲層厚", "濕度高"), "可能下雨")

    facts = {"下雨", "地面濕"}
    conclusions = expert_system.infer(facts)

    print("\n符號AI (專家系統) 測試:")
    print("已知事實:", facts)
    print("推理結論:", conclusions)

# 17. 生成對抗網絡 (GAN) 的簡化版本
class SimpleGAN:
    def __init__(self, data_dim):
        self.data_dim = data_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        return lambda z: z * np.random.randn(self.data_dim)

    def build_discriminator(self):
        return lambda x: 1 if np.sum(x) > 0 else 0

    def generate(self, num_samples):
        return [self.generator(np.random.randn()) for _ in range(num_samples)]

    def train(self, real_data, epochs):
        for epoch in range(epochs):
            real_labels = [self.discriminator(d) for d in real_data]
            fake_data = self.generate(len(real_data))
            fake_labels = [self.discriminator(d) for d in fake_data]
            
            z = np.random.randn(len(real_data))
            generated_data = [self.generator(zi) for zi in z]

def test_gan():
    data_dim = 5
    gan = SimpleGAN(data_dim)
    real_data = [np.random.randn(data_dim) for _ in range(100)]
    
    print("\n生成對抗網絡 (GAN) 測試:")
    print("訓練前生成的數據:")
    print(gan.generate(3))
    
    gan.train(real_data, epochs=1000)
    
    print("訓練後生成的數據:")
    print(gan.generate(3))

# 18. 語音識別 (簡化版，僅作為概念演示)
def simple_speech_recognition(audio_signal):
    energy = sum(abs(x) for x in audio_signal)
    if energy > 1000:
        return "有人說話"
    elif energy > 500:
        return "背景噪音"
    else:
        return "安靜"

def test_speech_recognition():
    quiet = [random.randint(0, 10) for _ in range(100)]
    noise = [random.randint(0, 100) for _ in range(100)]
    speech = [random.randint(0, 1000) for _ in range(100)]

    print("\n語音識別測試:")
    print("安靜環境:", simple_speech_recognition(quiet))
    print("有噪音環境:", simple_speech_recognition(noise))
    print("有人說話:", simple_speech_recognition(speech))

# 19. 3D建模 (簡單的線框模型)
class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Wireframe:
    def __init__(self):
        self.points = []
        self.edges = []

    def add_point(self, point):
        self.points.append(point)

    def add_edge(self, start, end):
        self.edges.append((start, end))

def test_3d_modeling():
    cube = Wireframe()
    for x in [0, 1]:
        for y in [0, 1]:
            for z in [0, 1]:
                cube.add_point(Point3D(x, y, z))

    for i in range(4):
        cube.add_edge(i, (i+1)%4)
        cube.add_edge(i+4, ((i+1)%4)+4)
        cube.add_edge(i, i+4)

    print("\n3D建模測試 (線框立方體):")
    print("頂點數量:", len(cube.points))
    print("邊的數量:", len(cube.edges))
    print("第一個頂點坐標:", f"({cube.points[0].x}, {cube.points[0].y}, {cube.points[0].z})")

# 運行所有測試
if __name__ == "__main__":
    test_decision_tree()
    test_genetic_algorithm()
    test_kmeans()
    test_expert_system()
    test_random_forest()
    test_monte_carlo()
    test_bayes_theorem()
    test_q_learning()
    test_fuzzy_logic()
    test_differential_evolution()
    test_word_frequency()
    test_edge_detection()
    test_astar()
    test_bayesian_network()
    test_neural_network()
    test_symbolic_ai()
    test_gan()
    test_speech_recognition()
    test_3d_modeling()
