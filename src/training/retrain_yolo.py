from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO


def build_dataset_split(data_root: Path, val_ratio: float = 0.15, seed: int = 42, force_resplit: bool = False) -> None:
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"No existe {train_dir}. Debe contener images/ y labels/.")

    train_images = train_dir / "images"
    train_labels = train_dir / "labels"
    if not train_images.exists() or not train_labels.exists():
        raise FileNotFoundError("Debe existir: data/train/images y data/train/labels")

    val_images = val_dir / "images"
    val_labels = val_dir / "labels"

    # Si se pide forzar resplit y val_dir existe, reintegrar todo en train primero
    if force_resplit and val_images.exists() and any(val_images.iterdir()):
        print(f"🔄 Reintegrando imágenes de {val_dir} en {train_dir} para un nuevo split...")
        for img_p in list(val_images.iterdir()):
            shutil.move(str(img_p), str(train_images / img_p.name))
            lbl_p = val_labels / f"{img_p.stem}.txt"
            if lbl_p.exists():
                shutil.move(str(lbl_p), str(train_labels / lbl_p.name))

    if not force_resplit and val_dir.exists() and val_images.exists() and any(val_images.iterdir()):
        val_count = len(list(val_images.iterdir()))
        train_count = len(list(train_images.iterdir()))
        print(f"El split ya existe en {val_dir} (Train={train_count}, Val={val_count}); se reutiliza.")
        return

    files = sorted(p for p in train_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not files:
        raise FileNotFoundError(f"No hay imágenes válidas en {train_images}.")

    rng = random.Random(seed)
    rng.shuffle(files)

    split_idx = max(1, int(len(files) * (1 - val_ratio)))
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    for src_img in val_files:
        stem = src_img.stem
        src_lbl = train_labels / f"{stem}.txt"
        dst_img = val_images / src_img.name
        dst_lbl = val_labels / f"{stem}.txt"

        shutil.move(str(src_img), str(dst_img))
        if src_lbl.exists():
            shutil.move(str(src_lbl), str(dst_lbl))

    print(f"✅ Split realizado con ratio {val_ratio*100:.0f}%: train={len(train_files)}; val={len(val_files)}")


def write_data_yaml(data_root: Path) -> Path:
    project_root = data_root.parent
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_path = (data_root / "train" / "images").resolve()
    val_path = (data_root / "val" / "images").resolve()
    yaml_path = config_dir / "data.yaml"
    yaml_content = f"""train: {train_path}
val: {val_path}

nc: 1
names: [\"plate\"]
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Archivo YAML creado: {yaml_path}")
    return yaml_path


def parse_args() -> argparse.Namespace:
    import torch
    default_device = "0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Retrain YOLOv8 para detección de placas vehiculares")
    parser.add_argument("--model", default="yolov8n.pt", help="Modelo base para fine-tuning")
    parser.add_argument("--epochs", type=int, default=80, help="Número de épocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default=default_device, help="Dispositivo para entrenamiento ('0' para GPU, 'cpu' para procesador)")
    parser.add_argument("--patience", type=int, default=20, help="Patience para early stopping")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fracción para validación (ej. 0.15 para 15%)")
    parser.add_argument("--force-resplit", action="store_true", help="Forzar re-división del conjunto de datos")
    parser.add_argument("--skip-split", action="store_true", help="No hacer split y reutilizar data/train + data/val si ya existen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"

    print(f"📂 Raíz del proyecto: {project_root}")
    print(f"📂 Directorio de datos: {data_root}")

    if not args.skip_split:
        build_dataset_split(data_root, val_ratio=args.val_ratio, force_resplit=args.force_resplit)

    yaml_path = write_data_yaml(data_root)

    model = YOLO(args.model)
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(project_root / "runs"),
        name="retrain_plate_v1",
        seed=42,
        workers=0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.0,
    )

    run_best_path = project_root / "runs" / "retrain_plate_v1" / "weights" / "best.pt"
    models_best_path = project_root / "models" / "best.pt"

    if run_best_path.exists():
        models_best_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_best_path, models_best_path)
        print(f"✅ ¡Entrenamiento finalizado con éxito!")
        print(f"🏆 Nuevo modelo guardado en: {models_best_path}")
    else:
        print(f"Entrenamiento finalizado. Modelo generado en: {project_root / 'runs' / 'retrain_plate_v1'}")


if __name__ == "__main__":
    main()
