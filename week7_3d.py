"""Bai thuc hanh so 7: STL -> point cloud -> loc Z -> depth image.

Mo hinh mac dinh: ../b1/Logitech c270.stl
Thu vien: NumPy, Matplotlib (khong dung Open3D de phan chieu depth).
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Results:
    triangle_count: int
    raw_point_count: int
    unique_point_count: int
    z_min: float
    z_max: float
    z_mid: float
    kept_count: int
    removed_count: int
    depth_pixels: int
    centroid_count: int
    augmented_count: int
    augmented_depth_pixels: int

    @property
    def additional_pixels(self) -> int:
        return self.augmented_depth_pixels - self.depth_pixels


def read_stl(path: Path) -> np.ndarray:
    """Doc STL binary/ASCII va tra ve mang tam giac (T, 3, 3)."""
    raw = path.read_bytes()
    if len(raw) < 15:
        raise ValueError(f"File STL rong hoac khong hop le: {path}")

    binary = False
    triangle_count = 0
    if len(raw) >= 84:
        triangle_count = struct.unpack_from("<I", raw, 80)[0]
        binary = 84 + 50 * triangle_count == len(raw)

    if binary:
        dtype = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]
        )
        records = np.frombuffer(raw, dtype=dtype, count=triangle_count, offset=84)
        triangles = records["vertices"].astype(np.float64)
    else:
        vertices: list[list[float]] = []
        for line in raw.decode("utf-8", errors="ignore").splitlines():
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertices.append([float(value) for value in fields[1:]])
        if not vertices or len(vertices) % 3 != 0:
            raise ValueError(f"Khong doc duoc cac tam giac trong {path}")
        triangles = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)

    if not np.isfinite(triangles).all():
        raise ValueError("STL chua toa do NaN hoac vo cuc.")
    return triangles


def merge_unique_vertices(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gop dinh thanh N x 3 va loai cac hang trung toa do."""
    all_points = triangles.reshape(-1, 3)
    unique_points = np.unique(all_points, axis=0)
    return all_points, unique_points


def project_depth(
    points: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_min: float,
    resolution: int = 64,
) -> np.ndarray:
    """Chieu diem bang camera nhin +Z, tu cap nhat tung o theo de bai.

    X -> cot u, Y -> hang v. clip bao dam diem o bien tren vao o cuoi.
    Gia tri nho hon duoc giu lai, tuong duong Z-buffer khi nhin theo +Z.
    """
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Khoang X/Y cua point cloud phai co do dai duong.")

    u = np.floor((points[:, 0] - x_min) / (x_max - x_min) * resolution).astype(int)
    v = np.floor((points[:, 1] - y_min) / (y_max - y_min) * resolution).astype(int)
    u = np.clip(u, 0, resolution - 1)
    v = np.clip(v, 0, resolution - 1)

    depth = np.full((resolution, resolution), np.nan, dtype=np.float64)
    for point, pixel_u, pixel_v in zip(points, u, v):
        new_depth = point[2] - z_min
        old_depth = depth[pixel_v, pixel_u]
        if np.isnan(old_depth) or new_depth < old_depth:
            depth[pixel_v, pixel_u] = new_depth
    return depth


def set_equal_3d(axis, points: np.ndarray) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float((high - low).max()) / 2.0, 1e-9)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.view_init(elev=23, azim=-58)


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")


def plot_question_1(points: np.ndarray, output: Path) -> plt.Figure:
    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=6, c=points[:, 2], cmap="viridis")
    axis.set_title(f"Câu 1 - Point cloud từ các đỉnh STL ({len(points):,} điểm)")
    set_equal_3d(axis, points)
    figure.tight_layout()
    save_figure(figure, output)
    return figure


def plot_question_2(
    kept: np.ndarray, removed: np.ndarray, z_mid: float, output: Path
) -> plt.Figure:
    all_points = np.vstack((kept, removed))
    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        removed[:, 0], removed[:, 1], removed[:, 2],
        s=7, c="#BDBDBD", alpha=0.65, label=f"Bị loại: z < z_mid ({len(removed):,})",
    )
    axis.scatter(
        kept[:, 0], kept[:, 1], kept[:, 2],
        s=8, c="#E63946", alpha=0.9, label=f"Giữ lại: z ≥ z_mid ({len(kept):,})",
    )
    axis.set_title(f"Câu 2 - Lọc theo Z, z_mid = {z_mid:.3f}")
    axis.legend(loc="upper right")
    set_equal_3d(axis, all_points)
    figure.tight_layout()
    save_figure(figure, output)
    return figure


def depth_colormap() -> plt.Colormap:
    colormap = plt.get_cmap("turbo").copy()
    colormap.set_bad("#E6E6E6")
    return colormap


def plot_question_3(
    depth: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    max_depth: float,
    output: Path,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8.5, 7))
    image = axis.imshow(
        depth,
        origin="lower",
        cmap=depth_colormap(),
        vmin=0,
        vmax=max_depth,
        extent=(x_limits[0], x_limits[1], y_limits[0], y_limits[1]),
        interpolation="nearest",
        aspect="equal",
    )
    axis.set_title(f"Câu 3 - Ảnh độ sâu 64×64 ({np.count_nonzero(~np.isnan(depth)):,} pixel có dữ liệu)")
    axis.set_xlabel("X → cột u")
    axis.set_ylabel("Y → hàng v")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Độ sâu z − z_min")
    figure.tight_layout()
    save_figure(figure, output)
    return figure


def plot_question_4(
    depth_original: np.ndarray,
    depth_augmented: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    max_depth: float,
    output: Path,
) -> plt.Figure:
    count_original = int(np.count_nonzero(~np.isnan(depth_original)))
    count_augmented = int(np.count_nonzero(~np.isnan(depth_augmented)))
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)
    common = dict(
        origin="lower",
        cmap=depth_colormap(),
        vmin=0,
        vmax=max_depth,
        extent=(x_limits[0], x_limits[1], y_limits[0], y_limits[1]),
        interpolation="nearest",
        aspect="equal",
    )
    image = axes[0].imshow(depth_original, **common)
    axes[0].set_title(f"Chỉ đỉnh STL\n{count_original:,} pixel có dữ liệu")
    axes[1].imshow(depth_augmented, **common)
    axes[1].set_title(
        f"Đỉnh + trọng tâm tam giác\n{count_augmented:,} pixel (+{count_augmented-count_original:,})"
    )
    for axis in axes:
        axis.set_xlabel("X → cột u")
        axis.set_ylabel("Y → hàng v")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.88)
    colorbar.set_label("Độ sâu z − z_min (cùng thang màu)")
    figure.suptitle("Câu 4 - So sánh ảnh độ sâu trên cùng lưới chiếu 64×64", fontsize=14)
    save_figure(figure, output)
    return figure


def question_4_comment(results: Results) -> str:
    percent = 100.0 * results.additional_pixels / results.depth_pixels if results.depth_pixels else 0.0
    return (
        f"Sau khi thêm {results.centroid_count:,} trọng tâm tam giác, số pixel có dữ liệu "
        f"tăng từ {results.depth_pixels:,} lên {results.augmented_depth_pixels:,}, tức tăng "
        f"{results.additional_pixels:,} pixel ({percent:.2f}% so với ban đầu).\n\n"
        "Các trọng tâm bổ sung mẫu ở phần giữa mỗi tam giác nên lấp được nhiều ô trống, đặc biệt "
        "ở vùng tròn quanh ống kính và trên thân camera kéo dài. Tuy nhiên, một tam giác chỉ được "
        "bổ sung một điểm tại trọng tâm. Tam giác lớn hoặc dài có thể phủ qua nhiều ô 64×64 nhưng "
        "vẫn chỉ đóng góp một mẫu mới; các điểm cũng phân bố không đều theo mật độ lưới tam giác. "
        "Vì vậy nhiều ô trên hình chiếu không chứa đỉnh hay trọng tâm nào và vẫn là NaN. Muốn bề "
        "mặt kín hơn cần lấy nhiều mẫu theo diện tích bên trong mỗi tam giác hoặc raster hóa toàn bộ "
        "tam giác, thay vì chỉ dùng đỉnh và một trọng tâm."
    )


def write_text_results(path: Path, source: Path, results: Results) -> None:
    text = f"""BÀI THỰC HÀNH SỐ 7 - LOGITECH C270

File STL: {source}

Câu 1
- Số tam giác: {results.triangle_count}
- Số điểm trước loại trùng: {results.raw_point_count}
- Số điểm sau loại trùng: {results.unique_point_count}

Câu 2
- z_min: {results.z_min:.6f}
- z_max: {results.z_max:.6f}
- z_mid: {results.z_mid:.6f}
- Số điểm giữ lại: {results.kept_count}
- Số điểm bị loại: {results.removed_count}

Câu 3
- Kích thước depth map: 64 x 64
- Số pixel có dữ liệu: {results.depth_pixels}

Câu 4
- Số trọng tâm thêm vào: {results.centroid_count}
- Tổng số điểm sau bổ sung: {results.augmented_count}
- Số pixel có dữ liệu mới: {results.augmented_depth_pixels}
- Số pixel tăng thêm: {results.additional_pixels}

NHẬN XÉT CÂU 4
{question_4_comment(results)}
"""
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    # File STL được đặt cùng thư mục với chương trình để repository chạy độc lập.
    default_stl = Path(__file__).resolve().parent / "Logitech c270.stl"
    parser = argparse.ArgumentParser(description="Bai thuc hanh 7 voi mo hinh Logitech C270 STL")
    parser.add_argument("input", nargs="?", type=Path, default=default_stl, help="Duong dan STL")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--resolution", type=int, default=64, help="Do phan giai depth (de bai: 64)")
    parser.add_argument("--no-show", action="store_true", help="Chi xuat file, khong mo cua so")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resolution != 64:
        print(f"Lưu ý: đề yêu cầu 64×64, hiện đang chạy {args.resolution}×{args.resolution}.")

    # ======================================================================
    # CÂU 1. TẠO ĐÁM MÂY ĐIỂM TỪ MÔ HÌNH STL
    # ======================================================================
    # Câu 1a: Đọc file STL và lấy danh sách tam giác.
    triangles = read_stl(source)

    # Câu 1b: Gộp ba đỉnh của mỗi tam giác thành mảng N x 3,
    # sau đó loại các hàng có tọa độ trùng nhau.
    raw_points, points = merge_unique_vertices(triangles)

    # Câu 1c được thực hiện ở hàm plot_question_1() phía dưới:
    # vẽ point cloud bằng scatter 3D và ghi tên ba trục X, Y, Z.

    # ======================================================================
    # CÂU 2. LỌC ĐÁM MÂY ĐIỂM THEO Z
    # ======================================================================
    # Câu 2a: Tính z_min, z_max và mặt phẳng phân chia z_mid.
    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max())
    z_mid = (z_min + z_max) / 2.0

    # Câu 2b: Giữ các điểm có z >= z_mid, phần còn lại là điểm bị loại.
    # Hai nhóm được vẽ bằng hai màu trong hàm plot_question_2().
    keep_mask = points[:, 2] >= z_mid
    kept = points[keep_mask]
    removed = points[~keep_mask]

    # ======================================================================
    # CÂU 3. TẠO ẢNH ĐỘ SÂU TỪ ĐÁM MÂY ĐIỂM
    # ======================================================================
    # Câu 3a: Lấy giới hạn X, Y để chia mỗi chiều thành 64 ô đều.
    # Việc đổi (x, y) thành (u, v), kể cả xử lý điểm ở biên trên,
    # nằm trong hàm project_depth().
    x_limits = (float(points[:, 0].min()), float(points[:, 0].max()))
    y_limits = (float(points[:, 1].min()), float(points[:, 1].max()))

    # Câu 3b: Khởi tạo depth bằng NaN và tự cập nhật từng ô.
    # Camera nhìn theo +Z nên mỗi ô giữ giá trị (z - z_min) nhỏ nhất.
    depth = project_depth(points, x_limits, y_limits, z_min, args.resolution)

    # Câu 3c được thực hiện ở hàm plot_question_3():
    # hiển thị depth map, thang màu và đếm số pixel có dữ liệu.

    # ======================================================================
    # CÂU 4. BỔ SUNG ĐIỂM TRÊN BỀ MẶT
    # ======================================================================
    # Câu 4a: Tính trọng tâm từng tam giác bằng trung bình ba đỉnh,
    # rồi ghép các trọng tâm vào point cloud của Câu 1.
    centroids = triangles.mean(axis=1)
    augmented_points = np.vstack((points, centroids))

    # Câu 4b: Chiếu lại bằng đúng lưới X-Y, z_min và độ phân giải cũ.
    # Hai ảnh dùng chung thang màu trong hàm plot_question_4().
    augmented_depth = project_depth(
        augmented_points, x_limits, y_limits, z_min, args.resolution
    )

    # Câu 4c: Thống kê số pixel có dữ liệu trước/sau khi bổ sung.
    # Hiệu hai giá trị được cung cấp bởi results.additional_pixels.
    results = Results(
        triangle_count=len(triangles),
        raw_point_count=len(raw_points),
        unique_point_count=len(points),
        z_min=z_min,
        z_max=z_max,
        z_mid=z_mid,
        kept_count=len(kept),
        removed_count=len(removed),
        depth_pixels=int(np.count_nonzero(~np.isnan(depth))),
        centroid_count=len(centroids),
        augmented_count=len(augmented_points),
        augmented_depth_pixels=int(np.count_nonzero(~np.isnan(augmented_depth))),
    )

    q1_path = output_dir / "cau1_point_cloud.png"
    q2_path = output_dir / "cau2_loc_z.png"
    q3_path = output_dir / "cau3_depth_64x64.png"
    q4_path = output_dir / "cau4_depth_comparison.png"
    figures = [
        plot_question_1(points, q1_path),
        plot_question_2(kept, removed, z_mid, q2_path),
        plot_question_3(depth, x_limits, y_limits, z_max - z_min, q3_path),
        plot_question_4(depth, augmented_depth, x_limits, y_limits, z_max - z_min, q4_path),
    ]

    result_path = output_dir / "ket_qua_va_nhan_xet.txt"
    write_text_results(result_path, source, results)

    print(f"File STL: {source}")
    print("\nCÂU 1")
    print(f"Số tam giác               : {results.triangle_count:,}")
    print(f"Số điểm trước loại trùng  : {results.raw_point_count:,}")
    print(f"Số điểm sau loại trùng    : {results.unique_point_count:,}")
    print("\nCÂU 2")
    print(f"z_min / z_max / z_mid     : {z_min:.3f} / {z_max:.3f} / {z_mid:.3f}")
    print(f"Giữ lại / bị loại         : {len(kept):,} / {len(removed):,}")
    print("\nCÂU 3")
    print(f"Pixel có dữ liệu           : {results.depth_pixels:,} / {args.resolution ** 2:,}")
    print("\nCÂU 4")
    print(f"Số trọng tâm thêm vào     : {results.centroid_count:,}")
    print(f"Pixel sau bổ sung          : {results.augmented_depth_pixels:,}")
    print(f"Số pixel tăng thêm         : {results.additional_pixels:,}")
    print(f"\nKết quả: {output_dir}")

    if args.no_show:
        for figure in figures:
            plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
