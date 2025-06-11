import timeit
import random
import matplotlib.pyplot as plt
import numpy as np
from heapq import heappush, heappop
import statistics

def insertion_sort(arr):
    """Сортування вставками"""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def merge_sort(arr):
    """Сортування злиттям"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    """Злиття двох відсортованих масивів"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def timsort_wrapper(arr):
    """Обгортка для Timsort (вбудований sorted)"""
    return sorted(arr)

def generate_test_data(size, data_type="random"):
    """Генерує тестові дані різних типів"""
    if data_type == "random":
        return [random.randint(1, 1000) for _ in range(size)]
    elif data_type == "sorted":
        return list(range(size))
    elif data_type == "reverse":
        return list(range(size, 0, -1))
    elif data_type == "partially_sorted":
        arr = list(range(size))
        # Перемішуємо тільки 10% елементів
        for _ in range(size // 10):
            i, j = random.randint(0, size-1), random.randint(0, size-1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr

def measure_time(algorithm, data, number=5):
    """Вимірює час виконання алгоритму з кількома запусками для точності"""
    times = []
    
    for _ in range(number):
        if algorithm.__name__ == 'timsort_wrapper':
            time_taken = timeit.timeit(lambda: sorted(data.copy()), number=1)
        else:
            time_taken = timeit.timeit(lambda: algorithm(data.copy()), number=1)
        times.append(time_taken)
    
    return {
        'mean': statistics.mean(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'times': times
    }

def run_performance_analysis():
    """Основний аналіз продуктивності з візуалізацією"""
    sizes = [100, 500, 1000, 2000, 5000, 10000]
    data_types = ["random", "sorted", "reverse", "partially_sorted"]
    algorithms = [insertion_sort, merge_sort, timsort_wrapper]
    
    results = {}
    
    print("🔬 АНАЛІЗ ПРОДУКТИВНОСТІ АЛГОРИТМІВ СОРТУВАННЯ")
    print("=" * 80)
    
    for data_type in data_types:
        print(f"\n📊 Тип даних: {data_type.upper()}")
        print("-" * 60)
        
        results[data_type] = {}
        
        for size in sizes:
            print(f"\nРозмір масиву: {size}")
            test_data = generate_test_data(size, data_type)
            
            for algorithm in algorithms:
                # Пропускаємо insertion sort для великих масивів
                if algorithm.__name__ == 'insertion_sort' and size > 5000:
                    print(f"  {algorithm.__name__:20}: Пропущено (занадто повільно)")
                    continue
                
                try:
                    time_stats = measure_time(algorithm, test_data)
                    
                    if algorithm.__name__ not in results[data_type]:
                        results[data_type][algorithm.__name__] = []
                    
                    results[data_type][algorithm.__name__].append((size, time_stats['mean']))
                    
                    print(f"  {algorithm.__name__:20}: {time_stats['mean']*1000:.2f} ± {time_stats['stdev']*1000:.2f} мс")
                    
                except Exception as e:
                    print(f"  {algorithm.__name__:20}: Помилка - {e}")
    
    # Створюємо графіки
    create_performance_plots(results)
    
    return results

def create_performance_plots(results):
    """Створює графіки продуктивності"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Порівняння алгоритмів сортування', fontsize=16)
    
    data_types = ["random", "sorted", "reverse", "partially_sorted"]
    titles = ["Випадкові дані", "Відсортовані дані", "Зворотно відсортовані", "Частково відсортовані"]
    
    for idx, (data_type, title) in enumerate(zip(data_types, titles)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        for alg_name, data_points in results[data_type].items():
            if data_points:
                sizes, times = zip(*data_points)
                ax.loglog(sizes, [t*1000 for t in times], 'o-', label=alg_name, linewidth=2, markersize=6)
        
        ax.set_xlabel('Розмір масиву')
        ax.set_ylabel('Час виконання (мс)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sorting_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📈 Графік збережено як 'sorting_performance.png'")

def create_complexity_plot():
    """Створює графік для демонстрації теоретичної складності"""
    sizes = np.array([100, 200, 500, 1000, 2000, 5000])
    
    # Теоретичні складності (нормалізовані)
    linear = sizes
    nlogn = sizes * np.log2(sizes)
    quadratic = sizes ** 2
    
    # Нормалізація для порівняння
    linear = linear / linear[0]
    nlogn = nlogn / nlogn[0]  
    quadratic = quadratic / quadratic[0]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(sizes, linear, 'g-', label='O(n)', linewidth=2)
    plt.loglog(sizes, nlogn, 'b-', label='O(n log n)', linewidth=2)
    plt.loglog(sizes, quadratic, 'r-', label='O(n²)', linewidth=2)
    
    plt.xlabel('Розмір масиву')
    plt.ylabel('Відносний час виконання')
    plt.title('Теоретична складність алгоритмів')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('complexity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Графік складності збережено як 'complexity_comparison.png'")
def analyze_complexity():
    """Емпіричний аналіз складності з візуалізацією"""
    print(f"\n🧮 ЕМПІРИЧНИЙ АНАЛІЗ СКЛАДНОСТІ")
    print("=" * 50)
    
    sizes = [100, 200, 400, 800, 1600, 3200]
    algorithms = [insertion_sort, merge_sort, timsort_wrapper]
    
    # Створюємо графік теоретичної складності
    create_complexity_plot()
    
    for algorithm in algorithms:
        print(f"\n{algorithm.__name__.upper()}:")
        times = []
        
        for size in sizes:
            # Пропускаємо insertion sort для дуже великих масивів
            if algorithm.__name__ == 'insertion_sort' and size > 1600:
                print(f"  Розмір {size:4}: Пропущено (занадто повільно)")
                continue
                
            test_data = generate_test_data(size, "random")
            time_stats = measure_time(algorithm, test_data, 3)
            time_taken = time_stats['mean']
            times.append(time_taken)
            
            # Теоретична складність
            if algorithm.__name__ == 'insertion_sort':
                theoretical = size * size
            else:
                theoretical = size * np.log2(size)
            
            ratio = time_taken / theoretical * 1000000
            
            print(f"  Розмір {size:4}: {time_taken*1000:.2f} мс (коеф. {ratio:.2f})")

def merge_k_lists_heap(lists):
    """
    Оптимізована версія merge_k_lists з використанням heap
    Складність: O(N log k) для всіх випадків
    """
    if not lists:
        return []
    
    result = []
    heap = []
    
    # Ініціалізуємо heap першими елементами кожного списку
    for i, lst in enumerate(lists):
        if lst:
            heappush(heap, (lst[0], i, 0))
    
    while heap:
        val, lst_idx, elem_idx = heappop(heap)
        result.append(val)
        
        # Додаємо наступний елемент з того ж списку
        elem_idx += 1
        if elem_idx < len(lists[lst_idx]):
            heappush(heap, (lists[lst_idx][elem_idx], lst_idx, elem_idx))
    
    return result

def demonstrate_timsort_efficiency():
    """Демонстрація ефективності Timsort"""
    print(f"\n⚡ ДЕМОНСТРАЦІЯ ПЕРЕВАГ TIMSORT")
    print("=" * 50)
    
    test_cases = [
        ("Частково відсортований", "partially_sorted"),
        ("Повністю відсортований", "sorted"),
        ("Зворотно відсортований", "reverse"),
        ("Випадковий", "random")
    ]
    
    size = 2000
    
    for name, data_type in test_cases:
        print(f"\n{name} масив ({size} елементів):")
        test_data = generate_test_data(size, data_type)
        
        # Тестуємо тільки merge_sort та timsort для справедливого порівняння
        merge_time = measure_time(merge_sort, test_data, 3)
        timsort_time = measure_time(timsort_wrapper, test_data, 3)
        
        # Обчислюємо прискорення за середнім часом виконання
        speedup = merge_time['mean'] / timsort_time['mean']
        
        print(f"  Merge Sort: {merge_time['mean']*1000:.2f} мс")
        print(f"  Timsort:    {timsort_time['mean']*1000:.2f} мс")
        print(f"  Прискорення: {speedup:.1f}x")

def merge_k_lists(lists):
    """
    НЕОБОВ'ЯЗКОВЕ ЗАВДАННЯ
    Злиття k відсортованих списків у один відсортований список
    """
    if not lists:
        return []
    
    while len(lists) > 1:
        merged_lists = []
        
        # Зливаємо списки попарно
        for i in range(0, len(lists), 2):
            left = lists[i]
            right = lists[i + 1] if i + 1 < len(lists) else []
            merged_lists.append(merge(left, right))
        
        lists = merged_lists
    
    return lists[0]

def test_merge_k_lists():
    """Тестування обох версій merge_k_lists"""
    print(f"\n🔗 ТЕСТУВАННЯ MERGE_K_LISTS")
    print("=" * 40)
    
    test_cases = [
        [[1, 4, 5], [1, 3, 4], [2, 6]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[1, 3], [], [2, 4, 5]],
        [[10, 20, 30], [5, 15, 25], [1, 11, 21], [2, 12, 22]]
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nТест {i}:")
        print(f"  Вхід: {test_case}")
        
        result1 = merge_k_lists(test_case)
        result2 = merge_k_lists_heap(test_case)
        
        print(f"  Попарне злиття: {result1}")
        print(f"  Heap злиття:    {result2}")
        print(f"  Збігаються: {'✅' if result1 == result2 else '❌'}")
    
    # Тест продуктивності
    print(f"\n⚡ ТЕСТ ПРОДУКТИВНОСТІ:")
    large_lists = [[i*10 + j for j in range(100)] for i in range(50)]
    
    import time
    
    start = time.time()
    result1 = merge_k_lists(large_lists)
    time1 = time.time() - start
    
    start = time.time()  
    result2 = merge_k_lists_heap(large_lists)
    time2 = time.time() - start
    
    print(f"  50 списків по 100 елементів:")
    print(f"  Попарне: {time1*1000:.2f} мс")
    print(f"  Heap:    {time2*1000:.2f} мс")
    print(f"  Прискорення: {time1/time2:.1f}x")

def print_conclusions():
    """Висновки дослідження"""
    print(f"\n📝 ВИСНОВКИ")
    print("=" * 50)
    
    conclusions = [
        "1. INSERTION SORT:",
        "   • Найкращий для малих масивів (< 50 елементів)",
        "   • O(n) для відсортованих даних, O(n²) для випадкових",
        "   • Простий у реалізації, стабільний",
        "",
        "2. MERGE SORT:",
        "   • Гарантована O(n log n) складність",
        "   • Стабільний, передбачувана продуктивність",
        "   • Потребує O(n) додаткової пам'яті",
        "",
        "3. TIMSORT (Python sorted/sort):",
        "   • Гібрид merge sort + insertion sort",
        "   • Адаптивний - O(n) для частково відсортованих",
        "   • Найефективніший для реальних даних",
        "   • Причина використання вбудованих функцій Python",
        "",
        "🎯 КЛЮЧОВИЙ ВИСНОВОК:",
        "Timsort показує найкращу продуктивність завдяки:",
        "• Використанню insertion sort для малих підмасивів",
        "• Виявленню наявних відсортованих послідовностей",
        "• Оптимізації для реальних, часто частково впорядкованих даних",
        "",
        "Саме тому програмісти використовують sorted() замість власних реалізацій!"
    ]
    
    for conclusion in conclusions:
        print(conclusion)

if __name__ == "__main__":
    # Основне завдання
    results = run_performance_analysis()
    analyze_complexity()
    demonstrate_timsort_efficiency()
    
    # Необов'язкове завдання
    test_merge_k_lists()
    
    # Висновки
    print_conclusions()
    
    print(f"\n✅ АНАЛІЗ ЗАВЕРШЕНО!")
    print("Детальні висновки збережено у файлі README.md")