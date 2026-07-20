"""
DRAW BOT — desenha uma imagem controlando o mouse.

A ideia é transformar uma imagem em movimentos de mouse. O caminho, do começo
ao fim, é:

    1. Ler a imagem e pegar os pixels escuros como pontos (x, y).
    2. Agrupar os pontos em "traços" (grupos de pontos vizinhos).
    3. Para cada traço:
         - ordenar os pontos numa rota curta (vizinho mais próximo);
         - simplificar a linha jogando fora pontos redundantes (Douglas-Peucker).
    4. Reproduzir cada traço movendo o mouse: desce a caneta, percorre os
       pontos, levanta a caneta e passa para o próximo traço.

Como usar:
    - ajuste o nome da imagem e o LIMIAR lá embaixo no bloco __main__;
    - rode o script, deixe o mouse sobre o canto onde o desenho deve começar;
    - pressione a tecla "d" para iniciar o desenho.
"""

import math
import sys
import time
import warnings

import cv2
import keyboard
import numpy as np
import pydirectinput as mouse
from numba import njit, prange
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")  # silencia avisos do numba/scipy
mouse.PAUSE = 0.003  # pausa mínima entre comandos do pydirectinput


# ---------------------------------------------------------------------------
# 1. IMAGEM -> PONTOS
# ---------------------------------------------------------------------------
def carregar_pontos_da_imagem(caminho, limiar=128, amostragem=1):
    """Lê a imagem em tons de cinza e devolve os pixels escuros como pontos.

    - `limiar`: pixels com valor abaixo dele são considerados "tinta".
    - `amostragem`: pula pixels para reduzir a densidade (1 = usa todos).
    Retorna um array Nx2 com as coordenadas (x, y).
    """
    imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível carregar {caminho}")

    if amostragem > 1:
        imagem = imagem[::amostragem, ::amostragem]

    # `mask` é True onde o pixel é escuro o suficiente para virar traço.
    mask = imagem < limiar
    y, x = np.where(mask)
    if amostragem > 1:
        x *= amostragem
        y *= amostragem
    return np.column_stack((x, y))


# ---------------------------------------------------------------------------
# 2. AGRUPAR PONTOS EM TRAÇOS (componentes conectados)
# ---------------------------------------------------------------------------
def agrupar_em_tracos(coords, distancia_max=20):
    """Junta pontos próximos em grupos (traços) usando union-find.

    Dois pontos ficam no mesmo traço se estão a menos de `distancia_max` um do
    outro (direta ou indiretamente). Usa uma KD-Tree para achar os vizinhos
    rápido. Retorna uma lista de traços; cada traço é uma lista de pontos.
    """
    coords = np.asarray(coords, dtype=np.float32)
    n = coords.shape[0]
    if n < 2:
        return [coords.tolist()] if n else []

    tree = cKDTree(coords)
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        # Sobe até a raiz comprimindo o caminho no meio do trajeto.
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Liga todos os pares de pontos dentro do raio. `query_pairs` faz essa
    # busca de uma vez em C, bem mais rápido que consultar ponto a ponto.
    for i, j in tree.query_pairs(distancia_max):
        union(i, j)

    # Agrupa os pontos por raiz (cada raiz = um traço).
    grupos = {}
    for i in range(n):
        grupos.setdefault(find(i), []).append(coords[i])

    return [np.array(g, dtype=np.float32).tolist() for g in grupos.values() if len(g) > 1]


# ---------------------------------------------------------------------------
# 3. ORDENAR E SIMPLIFICAR CADA TRAÇO
# ---------------------------------------------------------------------------
@njit(parallel=True)
def _matriz_de_distancias(coords):
    """Distância de cada ponto para cada outro (matriz NxN)."""
    n = coords.shape[0]
    dist = np.empty((n, n), dtype=np.float32)
    for i in prange(n):
        dist[i, i] = 0.0
        for j in range(i + 1, n):
            d = math.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
            dist[i, j] = d
            dist[j, i] = d
    return dist


@njit
def _rota_vizinho_mais_proximo(dist):
    """Ordena os pontos sempre indo ao mais próximo ainda não visitado.

    É uma heurística simples para o "caixeiro viajante": não dá a rota ótima,
    mas é rápida e boa o bastante para desenhar sem ficar pulando à toa.
    """
    n = dist.shape[0]
    visitado = np.zeros(n, dtype=np.bool_)
    rota = np.empty(n, dtype=np.int64)

    atual = 0
    visitado[0] = True
    rota[0] = 0
    for k in range(1, n):
        melhor, melhor_d = -1, 1e9
        for j in range(n):
            if not visitado[j] and dist[atual, j] < melhor_d:
                melhor_d, melhor = dist[atual, j], j
        rota[k] = melhor
        visitado[melhor] = True
        atual = melhor
    return rota


# Acima deste tamanho, a matriz NxN pesaria demais na RAM
# (n=5000 -> ~100 MB; n=20000 -> ~1,6 GB). Aí usamos a KD-Tree.
LIMITE_MATRIZ = 4000


def _rota_vizinho_mais_proximo_kdtree(pontos):
    """Mesma heurística do vizinho-mais-próximo, mas sem matriz NxN.

    Em vez de guardar a distância de todos para todos (memória n²), consulta
    a KD-Tree ponto a ponto. Usa memória O(n) — o custo vira CPU (as buscas na
    árvore), que é o que a gente quer para nuvens gigantes.
    """
    n = pontos.shape[0]
    tree = cKDTree(pontos)
    visitado = np.zeros(n, dtype=np.bool_)
    rota = np.empty(n, dtype=np.int64)

    atual = 0
    visitado[0] = True
    rota[0] = 0
    for k in range(1, n):
        # Pede os vizinhos mais próximos e pega o primeiro ainda não visitado.
        # Se todos vierem visitados, dobra o alcance da busca.
        vizinhos = min(n, 16)
        proximo = -1
        while True:
            idxs = np.atleast_1d(tree.query(pontos[atual], k=vizinhos)[1])
            for idx in idxs:
                if not visitado[idx]:
                    proximo = idx
                    break
            if proximo != -1 or vizinhos >= n:
                break
            vizinhos = min(n, vizinhos * 4)

        if proximo == -1:  # raro: varre o que sobrou e pega o mais próximo
            resto = np.where(~visitado)[0]
            d = np.hypot(
                pontos[resto, 0] - pontos[atual, 0],
                pontos[resto, 1] - pontos[atual, 1],
            )
            proximo = resto[int(np.argmin(d))]

        rota[k] = proximo
        visitado[proximo] = True
        atual = proximo
    return rota


def _ordenar_rota(pontos):
    """Escolhe o método de ordenação conforme o tamanho do componente."""
    if pontos.shape[0] <= LIMITE_MATRIZ:
        return _rota_vizinho_mais_proximo(_matriz_de_distancias(pontos))
    return _rota_vizinho_mais_proximo_kdtree(pontos)


@njit
def _douglas_peucker(pontos, epsilon=1.0):
    """Simplifica uma linha removendo pontos que quase não a alteram.

    Mantém só os pontos que ficam a mais de `epsilon` da reta traçada entre as
    pontas. Menos pontos = desenho mais rápido, sem perder o formato.
    """
    if len(pontos) < 3:
        return pontos

    def distancia_ate_reta(p, a, b):
        # Distância perpendicular do ponto `p` à reta que passa por `a` e `b`.
        num = abs((b[1] - a[1]) * p[0] - (b[0] - a[0]) * p[1] + b[0] * a[1] - b[1] * a[0])
        den = math.hypot(b[1] - a[1], b[0] - a[0])
        return num / den if den > 0 else 0.0

    manter = np.zeros(len(pontos), dtype=np.bool_)
    manter[0] = True
    manter[-1] = True

    # Pilha explícita (em vez de recursão) para dividir a linha em pedaços.
    pilha = [(0, len(pontos) - 1)]
    while pilha:
        ini, fim = pilha.pop()
        if fim - ini < 2:
            continue
        maior_d, idx = 0.0, ini
        for i in range(ini + 1, fim):
            d = distancia_ate_reta(pontos[i], pontos[ini], pontos[fim])
            if d > maior_d:
                maior_d, idx = d, i
        if maior_d > epsilon:
            manter[idx] = True
            pilha.append((ini, idx))
            pilha.append((idx, fim))
    return pontos[manter]


def _quebrar_em_saltos(pontos, salto_max):
    """Divide a rota em pedaços sempre que o pulo entre dois pontos é grande.

    O vizinho-mais-próximo, ao terminar uma região, precisa saltar para o
    próximo ponto — que às vezes está longe. Se desenhássemos esse salto com a
    caneta descida, apareceria uma linha reta cortando o desenho. Aqui a gente
    corta a rota nesses saltos, para a caneta levantar em vez de riscar.
    """
    pedacos = []
    atual = [pontos[0]]
    for anterior, ponto in zip(pontos, pontos[1:]):
        if math.hypot(ponto[0] - anterior[0], ponto[1] - anterior[1]) > salto_max:
            pedacos.append(atual)
            atual = []
        atual.append(ponto)
    pedacos.append(atual)
    return pedacos


def ordenar_e_simplificar(traco, salto_max):
    """Ordena os pontos numa rota curta e devolve uma lista de sub-traços.

    Retorna vários traços porque a rota é quebrada nos saltos grandes (ver
    `_quebrar_em_saltos`).
    """
    pontos = np.asarray(traco, dtype=np.float32)
    if pontos.shape[0] < 2:
        return []
    if pontos.shape[0] == 2:
        return [pontos.tolist()]

    pontos = pontos[_ordenar_rota(pontos)]

    sub_tracos = []
    for pedaco in _quebrar_em_saltos(pontos, salto_max):
        pedaco = np.asarray(pedaco, dtype=np.float32)
        if pedaco.shape[0] < 2:
            continue
        # Só vale simplificar quando há muitos pontos.
        if pedaco.shape[0] > 100:
            pedaco = _douglas_peucker(pedaco, epsilon=1.5)
        sub_tracos.append(pedaco.tolist())
    return sub_tracos


def preparar_desenho(coords, distancia_max=3, salto_max=6):
    """Transforma a nuvem de pontos numa lista de traços prontos para desenhar.

    - `distancia_max`: só junta pixels realmente vizinhos num mesmo traço.
    - `salto_max`: acima disso a caneta levanta em vez de riscar uma linha.
    """
    tracos = []
    for grupo in agrupar_em_tracos(coords, distancia_max=distancia_max):
        tracos.extend(ordenar_e_simplificar(grupo, salto_max=salto_max))
    return tracos


# ---------------------------------------------------------------------------
# 4. DESENHAR (mover o mouse)
# ---------------------------------------------------------------------------
def desenhar(tracos, atraso_inicial=3):
    """Desenha os traços movendo o mouse a partir da posição atual dele.

    Para cada traço: leva o mouse ao primeiro ponto, aperta o botão (desce a
    caneta), percorre os pontos e solta o botão (levanta a caneta).
    """
    time.sleep(atraso_inicial)
    ox, oy = mouse.position()  # tudo é desenhado relativo a este canto

    for traco in tracos:
        x0, y0 = traco[0]
        mouse.moveTo(int(x0 + ox), int(y0 + oy))
        time.sleep(0.05)
        mouse.mouseDown()

        for i, (x, y) in enumerate(traco[1:], start=1):
            mouse.moveTo(int(x + ox), int(y + oy))
            if i % 50 == 0:
                time.sleep(0.005)  # respiro para o programa alvo acompanhar

        time.sleep(0.05)
        mouse.mouseUp()


# ---------------------------------------------------------------------------
# EXECUÇÃO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    IMAGEM = "images/image67.png"
    LIMIAR = 250  # quanto maior, mais pixels contam como traço
    AMOSTRAGEM = 1  # 1 = usa todos os pixels
    DISTANCIA_MAX = 2  # une pixels a até N px no mesmo traço
    SALTO_MAX = 5  # acima disso a caneta levanta em vez de riscar

    try:
        print("=== DRAW BOT DE FUBA ===")

        coords = carregar_pontos_da_imagem(IMAGEM, LIMIAR, AMOSTRAGEM)
        if coords.size == 0:
            print("Nenhum ponto encontrado!")
            sys.exit(1)
        print(f"Total de pontos: {coords.shape[0]}")

        t0 = time.time()
        tracos = preparar_desenho(coords, DISTANCIA_MAX, SALTO_MAX)
        total = sum(len(t) for t in tracos)
        print(f"Otimização em {time.time() - t0:.2f}s | {coords.shape[0]}→{total} pontos")

        print("Posicione o mouse e pressione 'd' para desenhar...")
        keyboard.wait("d")

        t1 = time.time()
        desenhar(tracos)
        dt = time.time() - t1
        print(f"Desenho em {dt:.2f}s | {total / dt:.0f} pontos/s")
    except Exception as e:
        print(f"Erro: {e}")
