import random
import pandas as pd
import blosum
import copy
import time
import matplotlib.pyplot as plt # Importar matplotlib

# --- Configuración Inicial ---
blosum62 = blosum.BLOSUM(62)
NFE = 0  # Número de Evaluaciones de Fitness (Variable Global)

# --- Funciones Base (Comunes a ambos AGs) ---

def get_sequences():
    """Devuelve las secuencias de aminoácidos base."""
    seq1 = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRDLYDDDDKDRWGKLVVLGAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV"
    seq2 = "MKTLLVAAAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQKELQKQLGQKAKEL"
    seq3 = "MAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKALCVFAIN"
    return [list(seq1), list(seq2), list(seq3)]

def crear_individuo():
    return get_sequences()

def crear_poblacion_inicial(n=10):
    individuo_base = crear_individuo()
    poblacion = [[row[:] for row in individuo_base] for _ in range(n)]
    return poblacion

def mutar_poblacion_v2(poblacion, num_gaps=1):
    """Mutación inicial: inserta gaps al azar en la población."""
    poblacion_mutada = []
    for individuo in poblacion:
        nuevo_individuo = []
        for fila in individuo:
            fila_mutada = fila[:]
            posiciones = set()
            for _ in range(num_gaps):
                pos = random.randint(0, len(fila_mutada))
                while pos in posiciones:
                    pos = random.randint(0, len(fila_mutada))
                posiciones.add(pos)
                fila_mutada.insert(pos, '-')
            nuevo_individuo.append(fila_mutada)
        poblacion_mutada.append(nuevo_individuo)
    return poblacion_mutada

def igualar_longitud_secuencias(individuo, gap='-'):
    max_len = max(len(fila) for fila in individuo)
    individuo_igualado = [fila + [gap]*(max_len - len(fila)) for fila in individuo]
    return individuo_igualado

def evaluar_individuo_blosum62(individuo):
    """Calcula el fitness de un individuo usando BLOSUM62."""
    global NFE
    NFE += 1
    score = 0
    n_seqs = len(individuo)
    seq_len = len(individuo[0])
    for col in range(seq_len):
        for i in range(n_seqs):
            for j in range(i+1, n_seqs):
                a = individuo[i][col]
                b = individuo[j][col]
                if a == '-' or b == '-':
                    score -= 4
                else:
                    try:
                        score += blosum62[a][b]
                    except KeyError:
                        score -= 4 
    return score

def eliminar_peores(poblacion, scores, porcentaje=0.5):
    idx_ordenados = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    n_seleccionados = int(len(poblacion) * porcentaje)
    ind_seleccionados = [poblacion[i] for i in idx_ordenados[:n_seleccionados]]
    scores_seleccionados = [scores[i] for i in idx_ordenados[:n_seleccionados]]
    return ind_seleccionados, scores_seleccionados

def obtener_best(scores, poblacion):
    idx_mejor = scores.index(max(scores))
    fitness_best = scores[idx_mejor]
    best = copy.deepcopy(poblacion[idx_mejor])
    return best, fitness_best

def validar_poblacion_sin_gaps(poblacion, originales):
    """
    Función de validación de integridad.
    """
    for individuo in poblacion:
        for seq, seq_orig in zip(individuo, originales):
            seq_sin_gaps = [a for a in seq if a != '-']
            seq_orig_sin_gaps = [a for a in seq_orig if a != '-']
            if seq_sin_gaps != seq_orig_sin_gaps:
                print("¡ERROR DE INTEGRIDAD DETECTADO!")
                print("Original:", "".join(seq_orig_sin_gaps))
                print("Mutada:  ", "".join(seq_sin_gaps))
                return False
    return True

# --- FUNCIÓN DE CRUZA SEGURA (CORREGIDA) ---
# Esta función es usada por AMBOS algoritmos para una comparación justa.
def cruzar_individuos_integrity_safe(ind1, ind2):
    """
    Cruza de doble punto que PRESERVA LA INTEGRIDAD.
    Extrae aminoácidos, los recombina y los reinserta en la
    estructura de gaps del padre.
    """
    hijo1 = []
    hijo2 = []
    for seq1, seq2 in zip(ind1, ind2):
        # 1. Obtener índices de solo aminoácidos
        aa_indices = [i for i, a in enumerate(seq1) if a != '-']
        
        # Si no hay suficientes aminoácidos para cruzar, clonar y continuar
        if len(aa_indices) < 6:
            hijo1.append(seq1[:])
            hijo2.append(seq2[:])
            continue

        # 2. Seleccionar puntos de corte basados en la *lista de aminoácidos*
        # (p.ej., cortar en el 10mo y 20mo aminoácido, no en el índice 10 y 20)
        p1_idx_aa, p2_idx_aa = sorted(random.sample(range(len(aa_indices)), 2))

        # 3. Asegurar que el segmento tenga al menos 5 aminoácidos
        intentos = 0
        while (p2_idx_aa - p1_idx_aa) < 5 and intentos < 10:
             p1_idx_aa, p2_idx_aa = sorted(random.sample(range(len(aa_indices)), 2))
             intentos += 1

        # 4. Definir la función de cruza interna
        def cruza(seqA, seqB, idx1, idx2):
            # Extrae solo aminoácidos
            aaA = [a for a in seqA if a != '-']
            aaB = [a for a in seqB if a != '-']
            
            # Recombina las *listas de aminoácidos*
            nueva_aa_list = aaA[:idx1] + aaB[idx1:idx2] + aaA[idx2:]
            
            # Reconstruye la secuencia con la estructura de gaps de seqA
            resultado = []
            idx_aa = 0
            for char in seqA:
                if char == '-':
                    resultado.append('-')
                else:
                    # Asegurarse de que no nos pasemos del límite si aaB era más corta
                    if idx_aa < len(nueva_aa_list):
                        resultado.append(nueva_aa_list[idx_aa])
                        idx_aa += 1
                    else:
                        # Si la nueva lista de AA es más corta, rellenar con gaps
                        # (esto no debería pasar si aaA y aaB vienen de la misma original)
                        resultado.append('-') 
            return resultado

        # 5. Crear los nuevos hijos
        nueva_seq1 = cruza(seq1, seq2, p1_idx_aa, p2_idx_aa)
        nueva_seq2 = cruza(seq2, seq1, p1_idx_aa, p2_idx_aa)

        hijo1.append(nueva_seq1)
        hijo2.append(nueva_seq2)

    return hijo1, hijo2


# --- IMPLEMENTACIÓN DEL ALGORITMO GENÉTICO ORIGINAL (para comparación) ---

def mutar_individuo_original(individuo, p_mut_seq=0.8, num_gaps=1):
    """
    Mutación simple del AG Original: SOLO AÑADE gaps.
    """
    nuevo_individuo = []
    for secuencia in individuo:
        sec = secuencia[:]
        if random.random() < p_mut_seq: # Probabilidad por secuencia
            posiciones = set()
            for _ in range(num_gaps):
                pos = random.randint(0, len(sec))
                while pos in posiciones:
                    pos = random.randint(0, len(sec))
                posiciones.add(pos)
                sec.insert(pos, '-')
        nuevo_individuo.append(sec)
    return nuevo_individuo

def run_original_ga(N_POBLACION_INICIAL, N_GENERACIONES, secuencias_originales):
    """
    Ejecuta una simulación del AG Original.
    Usa selección aleatoria de padres y mutación simple (solo añadir gaps).
    """
    global NFE
    
    # --- Inicialización ---
    poblacion = crear_poblacion_inicial(N_POBLACION_INICIAL)
    poblacion = mutar_poblacion_v2(poblacion, num_gaps=5)
    poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
    scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
    
    # Quedarse con los mejores (población se reduce a N/2)
    poblacion, scores = eliminar_peores(poblacion, scores, porcentaje=0.5)
    
    best_fitness_per_gen = []
    fitnessVeryBest = -float('inf')
    
    # --- Bucle Evolutivo ---
    for gen in range(N_GENERACIONES):
        
        # 1. REPRODUCCIÓN (Cruza aleatoria + Elitismo simple)
        # (Simulando la lógica de 'cruzar_poblacion_doble_punto' de AG10.py)
        
        # Elitismo: Los padres (supervivientes) se copian a la sig. gen
        nueva_poblacion = [copy.deepcopy(ind) for ind in poblacion]
        
        # Generar N/2 hijos
        n_hijos_a_generar = len(poblacion)
        
        while len(nueva_poblacion) < n_hijos_a_generar * 2:
            # Selección aleatoria de padres entre los supervivientes
            padre1 = random.choice(poblacion)
            padre2 = random.choice(poblacion)
            
            # Cruza (USA LA FUNCIÓN SEGURA)
            hijo1, hijo2 = cruzar_individuos_integrity_safe(padre1, padre2)
            
            # 2. MUTACIÓN (Simple, solo añadir gaps)
            hijo1 = mutar_individuo_original(hijo1, p_mut_seq=0.8, num_gaps=1)
            hijo2 = mutar_individuo_original(hijo2, p_mut_seq=0.8, num_gaps=1)
            
            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < n_hijos_a_generar * 2:
                nueva_poblacion.append(hijo2)
        
        poblacion = nueva_poblacion
        
        # 3. EVALUACIÓN
        poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
        scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
        
        # 4. SELECCIÓN
        poblacion, scores = eliminar_peores(poblacion, scores, porcentaje=0.5)
        
        # 5. Reporte
        _, fitness_actual = obtener_best(scores, poblacion)
        if fitness_actual > fitnessVeryBest:
            fitnessVeryBest = fitness_actual
            
        best_fitness_per_gen.append(fitnessVeryBest)
        
    print(f"\n[AG Original] Mejor Fitness Final: {fitnessVeryBest:.2f}")
    # Validar integridad al final
    print(f"[AG Original] Validación de Integridad: {validar_poblacion_sin_gaps(poblacion, secuencias_originales)}")
    
    return best_fitness_per_gen


# --- IMPLEMENTACIÓN DEL ALGORITMO GENÉTICO MEJORADO ---

# --- MEJORA 1: Selección por Torneo ---
def seleccion_por_torneo(poblacion, scores, k=3):
    indices_torneo = [random.randint(0, len(poblacion)-1) for _ in range(k)]
    mejor_score = -float('inf')
    idx_ganador = -1
    for idx in indices_torneo:
        if scores[idx] > mejor_score:
            mejor_score = scores[idx]
            idx_ganador = idx
    return copy.deepcopy(poblacion[idx_ganador])

def crear_nueva_generacion_torneo(poblacion_superviviente, scores_superviviente, k=3):
    """
    Genera población usando Elitismo + Cruza por Torneo.
    """
    nueva_poblacion_total = []
    n_supervivientes = len(poblacion_superviviente)
    
    # 1. Elitismo: Todos los supervivientes (padres) pasan
    nueva_poblacion_total.extend([copy.deepcopy(ind) for ind in poblacion_superviviente])
    
    # 2. Cruza: Generar N/2 nuevos hijos usando torneo
    num_hijos_a_generar = n_supervivientes
    
    while len(nueva_poblacion_total) < num_hijos_a_generar * 2:
        padre1 = seleccion_por_torneo(poblacion_superviviente, scores_superviviente, k)
        padre2 = seleccion_por_torneo(poblacion_superviviente, scores_superviviente, k)
        
        # Cruza (USA LA FUNCIÓN SEGURA)
        hijo1, hijo2 = cruzar_individuos_integrity_safe(padre1, padre2)
        
        nueva_poblacion_total.append(hijo1)
        if len(nueva_poblacion_total) < num_hijos_a_generar * 2:
            nueva_poblacion_total.append(hijo2)
            
    return nueva_poblacion_total


# --- MEJORA 2: Mutación Flexible (Añadir, Quitar, Mover Gaps) ---
def mutar_individuo_flexible(individuo, p_mut_seq, p_add, p_rm, p_shift):
    """
    Muta un individuo con probabilidad 'p_mut_seq' por secuencia.
    La mutación puede ser: Añadir, Eliminar o Mover un gap.
    """
    nuevo_individuo = []
    for secuencia in individuo:
        sec = secuencia[:]
        if random.random() < p_mut_seq:
            gap_indices = [i for i, a in enumerate(sec) if a == '-']
            op = random.random()
            
            # 1. Añadir Gap
            if op < p_add or not gap_indices: 
                pos = random.randint(0, len(sec))
                sec.insert(pos, '-')
            
            # 2. Eliminar Gap
            elif op < p_add + p_rm:
                pos = random.choice(gap_indices)
                sec.pop(pos)
            
            # 3. Mover Gap
            else:
                pos_gap = random.choice(gap_indices)
                direction = random.choice([-1, 1])
                pos_new = pos_gap + direction
                
                if 0 <= pos_new < len(sec) and sec[pos_new] != '-':
                    sec[pos_gap], sec[pos_new] = sec[pos_new], sec[pos_gap]

        nuevo_individuo.append(sec)
    return nuevo_individuo


def run_mejorado_ga(N_POBLACION_INICIAL, N_GENERACIONES, secuencias_originales):
    """
    Ejecuta el AG Mejorado.
    Usa Selección por Torneo y Mutación Flexible.
    """
    global NFE
    
    # --- Parámetros del AG Mejorado ---
    K_TORNEO = 3
    P_MUTACION_SEQ = 0.3
    P_ADD_GAP = 0.4
    P_RM_GAP = 0.4
    P_SHIFT_GAP = 0.2
    
    # --- Inicialización ---
    poblacion = crear_poblacion_inicial(N_POBLACION_INICIAL)
    poblacion = mutar_poblacion_v2(poblacion, num_gaps=5)
    poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
    scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
    poblacion, scores = eliminar_peores(poblacion, scores, porcentaje=0.5)
    
    best_fitness_per_gen = []
    veryBest = None
    fitnessVeryBest = -float('inf')
    
    print(f"Iniciando AG Mejorado. Supervivientes: {len(poblacion)}. NFE: {NFE}")
    
    # --- Bucle Evolutivo ---
    for gen in range(N_GENERACIONES):
        
        # 1. REPRODUCCIÓN (Torneo + Elitismo)
        poblacion_nueva_generacion = crear_nueva_generacion_torneo(
            poblacion, scores, k=K_TORNEO
        )
        
        # 2. MUTACIÓN (Flexible)
        poblacion_mutada = []
        for ind in poblacion_nueva_generacion:
            ind_mutado = mutar_individuo_flexible(
                ind, 
                p_mut_seq=P_MUTACION_SEQ, 
                p_add=P_ADD_GAP, 
                p_rm=P_RM_GAP, 
                p_shift=P_SHIFT_GAP
            )
            poblacion_mutada.append(ind_mutado)
        
        poblacion = poblacion_mutada
        
        # 3. EVALUACIÓN
        poblacion = [igualar_longitud_secuencias(ind) for ind in poblacion]
        scores = [evaluar_individuo_blosum62(ind) for ind in poblacion]
        
        # 4. SELECCIÓN
        poblacion, scores = eliminar_peores(poblacion, scores, porcentaje=0.5)
        
        # 5. Reporte
        best_actual, fitness_actual = obtener_best(scores, poblacion)
        if fitness_actual > fitnessVeryBest:
            veryBest = best_actual
            fitnessVeryBest = fitness_actual
            
        best_fitness_per_gen.append(fitnessVeryBest)

    print("\n--- ¡Evolución Mejorada Terminada! ---")
    print(f"Mejor Fitness Global Encontrado: {fitnessVeryBest:.2f}")
    
    print("\nValidando integridad de la población final (Mejorado)...")
    validacion = validar_poblacion_sin_gaps(poblacion, secuencias_originales)
    print(f"Resultado de Validación de Integridad: {validacion}")
    
    print("\nMejor Alineamiento Encontrado (veryBest Mejorado):")
    for seq in veryBest:
        print("".join(seq))
        
    return best_fitness_per_gen, veryBest


# --- Bucle Principal (Main) para ejecutar ambos y graficar ---
if __name__ == "__main__":
    
    # --- Parámetros Comunes del Algoritmo ---
    N_POBLACION_INICIAL = 20  # Población total (se reduce a 10 supervivientes)
    N_GENERACIONES = 100
    
    secuencias_originales = get_sequences() # Guardar para validación
    
    # --- Ejecutar el Algoritmo Genético ORIGINAL ---
    print("--- Ejecutando AG Original ---")
    NFE = 0 # Reset NFE
    start_time_original = time.time()
    best_fitness_original = run_original_ga(
        N_POBLACION_INICIAL, N_GENERACIONES, secuencias_originales
    )
    end_time_original = time.time()
    print(f"[AG Original] NFE: {NFE}")
    print(f"Tiempo total AG Original: {end_time_original - start_time_original:.2f}s\n")
    
    
    # --- Ejecutar el Algoritmo Genético MEJORADO ---
    print("--- Ejecutando AG Mejorado ---")
    NFE = 0 # Reset NFE
    start_time_mejorado = time.time()
    best_fitness_mejorado, veryBest_mejorado = run_mejorado_ga(
        N_POBLACION_INICIAL, N_GENERACIONES, secuencias_originales
    )
    end_time_mejorado = time.time()
    print(f"[AG Mejorado] NFE: {NFE}")
    print(f"Tiempo total AG Mejorado: {end_time_mejorado - start_time_mejorado:.2f}s\n")

    
    # --- Generar la Gráfica ---
    print("\n--- Generando Gráfica Comparativa ---")
    
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, N_GENERACIONES + 1), best_fitness_original, label='Original (Selección Aleatoria + Mutación Simple)', linestyle='--', marker='o', markersize=2, alpha=0.7)
    plt.plot(range(1, N_GENERACIONES + 1), best_fitness_mejorado, label='Mejorado (Torneo + Mutación Flexible)', linestyle='-', marker='.', markersize=2)
    
    plt.xlabel('Generación')
    plt.ylabel('Best Fitness (más alto es mejor)')
    plt.title('Comparativa de Fitness: Original vs Mejorado')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("Gráfica generada y mostrada.")