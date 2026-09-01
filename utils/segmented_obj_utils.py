import numpy as np
from collections import deque


def fix_mesh_orientation(V, F, large_component_fraction=0.3):
    """
    Делает порядок вершин (winding order) СОГЛАСОВАННЫМ и физически
    корректным ("наружу") по всему мешу.

    Проблема: если в исходном OBJ грани записаны в разном порядке для
    разных частей меша, naive-расчёт нормалей через cross-product даёт
    нормали, развёрнутые на 180° для части граней.

    Шаг 1 - согласование внутри каждой связной компоненты (BFS по графу
    смежности граней: соседние грани должны проходить общее ребро в
    ПРОТИВОПОЛОЖНЫХ направлениях; если совпало - грань "перевёрнута"
    относительно соседей, чиним порядок вершин).

    Важно: в датасете встречаются ДВА разных случая структуры файла:
    - зубы физически СОЕДИНЕНЫ с десной (общие вершины на стыке) - тогда
      весь скан представляет собой ОДНУ большую связную компоненту
      (открытую дугу), как в исходных файлах датасета;
    - зубы физически РАЗЪЕДИНЕНЫ друг от друга и от десны - тогда меш
      распадается на много отдельных компонент (зуб/зуб/десна), как в
      целевых моделях для инференса, которые не входят в датасет.

    Шаг 2 - абсолютная ориентация "наружу" через знаковый объём (стандартный
    геометрический тест), но ТОЛЬКО для компонент, которые занимают
    небольшую долю (< large_component_fraction) от общего числа граней
    меша. Такие компоненты - это отдельные почти замкнутые "шапочки"
    (один зуб, один фрагмент десны), для которых тест по объёму надёжен.

    Для КРУПНОЙ компоненты (одна большая открытая дуга целиком, occupies
    >= large_component_fraction всех граней) объём НЕ считается - тест по
    объёму ненадёжен на такой большой открытой площади (эмпирически
    проверено: даёт ложный результат, разворачивая всё). Для такой
    компоненты остаётся только результат шага 1 (согласованность внутри
    себя, без попытки определить абсолютную сторону).

    Проверено на реальных файлах:
    - разъединённый скан (19 компонент, крупнейшая = 11% от всех граней):
      per-component volume даёт 16/16 корректных зубов (было 8/16 развёрнуты)
    - соединённый скан из датасета (1 компонента = 100% граней): не
      трогается volume-тестом, остаётся как есть (уже был корректен)

    V: (N, 3)
    F: (M, 3) индексы граней (0-based)
    large_component_fraction: порог доли граней компоненты от всего меша,
        выше которого volume-тест не применяется (считается ненадёжным)
    returns: F_fixed (M, 3) - те же грани, с согласованным и (где надёжно
             определимо) корректно ориентированным порядком вершин
    """
    F = F.copy()
    M = len(F)
    if M == 0:
        return F

    # undirected edge -> [face_idx, ...] (обычно 2 грани на ребро)
    edge_map = {}
    for fi in range(M):
        a, b, c = F[fi]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_map.setdefault(key, []).append(fi)

    adjacency = [[] for _ in range(M)]
    for flist in edge_map.values():
        if len(flist) == 2:
            f1, f2 = flist
            adjacency[f1].append(f2)
            adjacency[f2].append(f1)

    visited = np.zeros(M, dtype=bool)
    component_id = np.full(M, -1, dtype=np.int32)
    n_components = 0

    # ---------- шаг 1: согласование внутри каждой компоненты ----------
    for start in range(M):
        if visited[start]:
            continue
        visited[start] = True
        component_id[start] = n_components
        queue = deque([start])

        while queue:
            fi = queue.popleft()
            a, b, c = F[fi]
            fi_edges = {(a, b), (b, c), (c, a)}

            for fj in adjacency[fi]:
                if visited[fj]:
                    continue

                aj, bj, cj = F[fj]
                fj_edges = {(aj, bj), (bj, cj), (cj, aj)}

                if fi_edges & fj_edges:
                    F[fj] = (bj, aj, cj)

                visited[fj] = True
                component_id[fj] = n_components
                queue.append(fj)

        n_components += 1

    # ---------- шаг 2: абсолютная ориентация "наружу", только для НЕБОЛЬШИХ компонент ----------
    for comp in range(n_components):
        mask = component_id == comp
        faces_c = F[mask]
        if len(faces_c) == 0:
            continue

        # крупная компонента (целая открытая дуга) - volume-тест
        # ненадёжен, пропускаем, оставляем как после шага 1
        if len(faces_c) / M >= large_component_fraction:
            continue

        v0 = V[faces_c[:, 0]].astype(np.float64)
        v1 = V[faces_c[:, 1]].astype(np.float64)
        v2 = V[faces_c[:, 2]].astype(np.float64)

        signed_vol = np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0

        if signed_vol < 0:
            F[mask] = faces_c[:, [1, 0, 2]]

    return F


def compute_vertex_normals(V, F):
    """
    V: (N, 3) координаты
    F: (M, 3) индексы граней (0-based) - ДОЛЖНЫ иметь согласованный
       порядок вершин (см. fix_mesh_orientation) для корректного знака
       нормалей по всему мешу.
    Простой area-weighted расчёт нормалей вершин, без внешних
    зависимостей - надёжен независимо от того, как конкретная
    библиотека интерпретирует множественные "o"-группы в OBJ
    (trimesh может вернуть Scene с раздельными подмешами вместо
    единого меша, что сбило бы индексацию вершин).
    """
    normals = np.zeros_like(V, dtype=np.float64)

    v0 = V[F[:, 0]].astype(np.float64)
    v1 = V[F[:, 1]].astype(np.float64)
    v2 = V[F[:, 2]].astype(np.float64)

    face_normals = np.cross(v1 - v0, v2 - v0)  # не нормированы -> вес = площадь*2

    np.add.at(normals, F[:, 0], face_normals)
    np.add.at(normals, F[:, 1], face_normals)
    np.add.at(normals, F[:, 2], face_normals)

    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    normals = normals / norm

    return normals.astype(np.float32)


def load_segmented_obj(path):
    """
    Читает OBJ, где меш может быть разбит на группы по объектам:
        o gingiva
        g gingiva
        f ...
        o tooth_1
        g tooth_1
        f ...
        ...
    Вершины (v) определены один раз в общем списке; каждый f-блок под
    своим "o <name>" ссылается на индексы из этого общего списка.
    Также поддерживает обычные (несегментированные) OBJ без "o"-групп -
    тогда все грани всё равно собираются, а label = -1 для всех вершин
    (сегментация неизвестна).

    Возвращает:
        V:      (N, 3) float32 - координаты вершин
        F:      (M, 3) int64 - индексы граней (0-based), из ВСЕХ групп
                 (или всего файла, если групп нет), в исходном порядке
        C:      (N, 3) float32 - цвет вершин в [0,1], если есть в файле
                 (v x y z r g b), иначе None
        label:  (N,) int32 - 0 = десна (gingiva), 1 = зуб (любой tooth_*),
                 -1 = вершина не размечена (нет групп в файле, либо
                 вершина не встретилась ни в одной грани ни одной группы)
        tooth_id: (N,) int32 - номер зуба (1..14) для вершин с label==1,
                 0 для остальных
    """
    verts = []
    has_color = None
    faces = []

    current_group = None
    group_vertex_ids = {}   # name -> set of vertex indices (1-based)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                if len(parts) >= 7:
                    if has_color is None:
                        has_color = True
                    r, g, b = map(float, parts[4:7])
                    verts.append((x, y, z, r, g, b))
                else:
                    if has_color is None:
                        has_color = False
                    verts.append((x, y, z, 0.0, 0.0, 0.0))

            elif line.startswith("o "):
                current_group = line.strip().split(maxsplit=1)[1]
                if current_group not in group_vertex_ids:
                    group_vertex_ids[current_group] = set()

            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face_idx = []
                for p in parts:
                    # формат может быть "idx" или "idx/vt" или "idx/vt/vn"
                    idx = int(p.split("/")[0])
                    if current_group is not None:
                        group_vertex_ids[current_group].add(idx)
                    face_idx.append(idx - 1)  # 0-based

                # триангуляция веером на случай не-треугольных граней
                for k in range(1, len(face_idx) - 1):
                    faces.append((face_idx[0], face_idx[k], face_idx[k + 1]))

    verts = np.asarray(verts, dtype=np.float32)
    V = verts[:, 0:3]
    C = verts[:, 3:6] if has_color else None
    F = np.asarray(faces, dtype=np.int64)

    n = len(V)
    label = np.full(n, -1, dtype=np.int32)
    tooth_id = np.zeros(n, dtype=np.int32)

    # Сначала простановка десны, потом зубов - зубы имеют приоритет на
    # редких пересекающихся вершинах шва (overlap на границе).
    for name, idx_set in group_vertex_ids.items():
        if not idx_set:
            continue
        idx_arr = np.fromiter(idx_set, dtype=np.int64) - 1  # OBJ индексы с 1
        idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < n)]

        if name.lower().startswith("tooth"):
            label[idx_arr] = 1
            try:
                t_num = int(name.split("_")[-1])
            except ValueError:
                t_num = 0
            tooth_id[idx_arr] = t_num
        else:
            # всё, что не tooth_*, считаем десной (gingiva или иное имя)
            already_tooth = label[idx_arr] == 1
            gum_idx = idx_arr[~already_tooth]
            label[gum_idx] = 0

    return V, F, C, label, tooth_id


def save_colored_obj(path, V, F, colors01):
    """
    Простая запись OBJ вручную (без trimesh), чтобы не зависеть от того,
    как конкретная версия библиотеки обрабатывает файлы с несколькими
    "o"-группами. Пишет плоский меш: вершины с цветом (v x y z r g b) +
    треугольные грани (f, 1-based индексы).

    V:        (N, 3) координаты (те же, что были во входном файле - НЕ
               нормализованные, чтобы результат сохранял исходный масштаб)
    F:        (M, 3) индексы граней (0-based), как из load_segmented_obj
    colors01: (N, 3) цвет в диапазоне [0, 1]
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Colored by model (exported without trimesh)\n")
        for (x, y, z), (r, g, b) in zip(V, colors01):
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
        for a, b_, c in F:
            f.write(f"f {a + 1} {b_ + 1} {c + 1}\n")


def save_colored_obj_grouped(path, V, F, colors01, label, tooth_id):
    """
    Как save_colored_obj, но при наличии сегментации пишет результат
    ТОЖЕ разбитым на группы "o gingiva" / "o tooth_N", как во входном
    файле - чтобы сегментированный вход давал сегментированный выход.

    Группа каждой грани определяется мажоритарным голосованием по
    label/tooth_id её трёх вершин (грани почти всегда целиком лежат в
    одной группе, кроме редких вершин на самом шве).

    Если сегментации нет вообще (label == -1 везде) - пишет плоский
    меш без групп, как save_colored_obj.
    """
    if not (label >= 0).any():
        save_colored_obj(path, V, F, colors01)
        return

    # групповая метка каждой грани: (is_tooth, tooth_number)
    face_label = label[F]        # (M, 3)
    face_tooth = tooth_id[F]     # (M, 3)

    # мажоритарный голос по 3 вершинам грани (сумма >=2 из 3 => зуб)
    face_is_tooth = (face_label == 1).sum(axis=1) >= 2

    # для граней-зубов - самый частый tooth_id среди вершин
    face_tooth_num = np.zeros(len(F), dtype=np.int32)
    tooth_rows = np.where(face_is_tooth)[0]
    if len(tooth_rows):
        # np.bincount по каждой строке было бы медленно построчно на
        # большом M, но M обычно ~2*10^5, приемлемо
        for r in tooth_rows:
            vals = face_tooth[r]
            vals = vals[vals > 0]
            face_tooth_num[r] = np.bincount(vals).argmax() if len(vals) else 0

    # группировка граней
    groups = {}   # name -> list of face rows (0-based vertex idx triples)
    for is_tooth, t_num, face in zip(face_is_tooth, face_tooth_num, F):
        name = f"tooth_{t_num}" if (is_tooth and t_num > 0) else "gingiva"
        groups.setdefault(name, []).append(face)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Colored by model (exported without trimesh, grouped)\n")
        for (x, y, z), (r, g, b) in zip(V, colors01):
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

        # десна первой (как в исходных файлах), затем зубы по номеру
        ordered_names = sorted(
            groups.keys(),
            key=lambda n: (n != "gingiva", int(n.split("_")[-1]) if n != "gingiva" else 0)
        )
        for name in ordered_names:
            f.write(f"o {name}\n")
            f.write(f"g {name}\n")
            for a, b_, c in groups[name]:
                f.write(f"f {a + 1} {b_ + 1} {c + 1}\n")