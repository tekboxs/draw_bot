import numpy as np
import math
import pydirectinput as dir
import time
import sys
import keyboard
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from numba import njit, prange
from tqdm import tqdm
import cv2
import warnings

warnings.filterwarnings("ignore")
sys.setrecursionlimit(100000)
dir.PAUSE = 0.003


@njit(parallel=True)
def fast_distance_matrix(coords):
    n = coords.shape[0]
    D = np.empty((n, n), dtype=np.float32)
    for i in prange(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            d = math.hypot(dx, dy)
            D[i, j] = d
            D[j, i] = d
    return D


@njit
def nearest_neighbor_numba(dist):
    n = dist.shape[0]
    visited = np.zeros(n, dtype=np.bool_)
    path = np.empty(n, dtype=np.int64)
    current = 0
    visited[0] = True
    path[0] = 0
    for k in range(1, n):
        best = -1
        best_d = 1e9
        for j in range(n):
            if not visited[j] and dist[current, j] < best_d:
                best_d = dist[current, j]
                best = j
        path[k] = best
        visited[best] = True
        current = best
    return path


def nearest_neighbor_jit(coords):
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[0] < 2:
        return coords.tolist()
    D = fast_distance_matrix(coords)
    idx_path = nearest_neighbor_numba(D)
    return coords[idx_path].tolist()


@njit
def douglas_peucker_numba(points, epsilon=1.0):
    if len(points) < 3:
        return points

    def perp(p, a, b):
        x0, y0 = p
        x1, y1 = a
        x2, y2 = b
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = math.hypot(y2 - y1, x2 - x1)
        return num / den if den > 0 else 0.0

    keep = np.zeros(len(points), dtype=np.bool_)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        max_d = 0.0
        idx = start
        for i in range(start + 1, end):
            d = perp(points[i], points[start], points[end])
            if d > max_d:
                max_d = d
                idx = i
        if max_d > epsilon:
            keep[idx] = True
            stack.append((start, idx))
            stack.append((idx, end))
    return points[keep]


def generate_image_coordinates_optimized(image_path, threshold=128, sampling_rate=1):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Não foi possível carregar {image_path}")
    if sampling_rate > 1:
        image = image[::sampling_rate, ::sampling_rate]
    mask = image < threshold
    y_coords, x_coords = np.where(mask)
    if sampling_rate > 1:
        x_coords *= sampling_rate
        y_coords *= sampling_rate
    return np.column_stack((x_coords, y_coords))


def advanced_clustering_dbscan(coords, eps=2.0, min_samples=2):
    coords = np.asarray(coords)
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(coords)
    centroids = []
    for label in set(labels):
        pts = coords[labels == label]
        if label == -1:
            centroids.extend(pts.tolist())
        else:
            centroids.append(np.mean(pts, axis=0).tolist())
    return np.array(centroids, dtype=np.float32)


def connected_components_optimized(coords, max_distance=8):
    coords = np.asarray(coords, dtype=np.float32)
    n = coords.shape[0]
    if n < 2:
        return [coords.tolist()] if n else []
    tree = cKDTree(coords)
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa = find(a)
        pb = find(b)
        if pa == pb:
            return
        if rank[pa] < rank[pb]:
            pa, pb = pb, pa
        parent[pb] = pa
        if rank[pa] == rank[pb]:
            rank[pa] += 1

    for i in range(n):
        for j in tree.query_ball_point(coords[i], max_distance):
            if i != j:
                union(i, j)
    comps = {}
    for i in range(n):
        root = find(i)
        comps.setdefault(root, []).append(coords[i])
    return [
        np.array(c, dtype=np.float32).tolist() for c in comps.values() if len(c) > 1
    ]


def process_component(component):
    c = np.asarray(component, dtype=np.float32)
    n = c.shape[0]
    if n < 3:
        return component
    path = nearest_neighbor_jit(c.tolist())
    if len(path) > 100:
        arr = np.array(path, dtype=np.float32)
        path = douglas_peucker_numba(arr, epsilon=1.5).tolist()
    return path


def optimize_drawing_path_advanced(coords, break_threshold=10.0):
    coords = np.asarray(coords, dtype=np.float32)

    # Agrupa pontos próximos em clusters
    coords = advanced_clustering_dbscan(coords, eps=0.1, min_samples=1)

    # Encontra componentes conectados
    components = connected_components_optimized(coords, max_distance=20)
    final = []
    counter = 1
    prev_point = None
    for path in components:
        subpath = process_component(path)
        if prev_point is not None:
            # Verifica distância entre último ponto e primeiro do próximo subpath
            dist = math.hypot(
                subpath[0][0] - prev_point[0], subpath[0][1] - prev_point[1]
            )
            if dist > break_threshold:
                final.append((-counter, -counter))
                counter += 1
        final.extend(subpath)
        prev_point = subpath[-1]
    return final


def draw_optimized_path_fast(coordinates):
    time.sleep(3)
    start = dir.position()
    dir.moveTo(int(coordinates[0][0] + start[0]), int(coordinates[0][1] + start[1]))
    time.sleep(0.05)
    dir.mouseDown()
    pts = 0

    skip = False

    for x, y in coordinates[1:]:
        if skip:
            skip = False
            continue
        if x < 0 and y < 0:
            # time.sleep(2)
            time.sleep(0.05)
            dir.mouseUp()
            index = coordinates.index((x, y))
            newLine = coordinates[index + 1]
            dir.moveTo(int(newLine[0] + start[0]), int(newLine[1] + start[1]))
            time.sleep(0.05)
            dir.mouseDown()
            skip = True
            continue
        dir.moveTo(int(x + start[0]), int(y + start[1]))
        pts += 1
        if pts % 50 == 0:
            time.sleep(0.005)
    dir.mouseUp()


if __name__ == "__main__":
    try:
        print("=== DRAW BOT DE FUBA ===")
        threshold, sampling_rate = 220, 1
        coords = generate_image_coordinates_optimized(
            "image5.png", threshold, sampling_rate
        )
        if coords.size == 0:
            print("Nenhum ponto encontrado!")
            sys.exit(1)
        print(f"Total de pontos: {coords.shape[0]}")
        start_time = time.time()
        optimized_path = optimize_drawing_path_advanced(coords)
        print(
            f"Otimização em {time.time() - start_time:.2f}s | {coords.shape[0]}→{len(optimized_path)} pontos"
        )
        keyboard.wait("d")
        draw_start = time.time()
        draw_optimized_path_fast(optimized_path)
        print(
            f"Desenho em {time.time() - draw_start:.2f}s | {len(optimized_path)/(time.time() - draw_start):.0f} p/s"
        )
    except Exception as e:
        print(f"Erro: {e}")

