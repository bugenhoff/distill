# YOLO26 Distillation Workspace

Проект для Knowledge Distillation моделей YOLO26 с поддержкой трёх задач: **detect**, **segment**, **pose**.
Экспорт артефактов в ONNX (FP32) и OpenVINO (FP16/INT8) с последующей проверкой численной сходимости (parity check).

## Возможности

- **Три задачи дистилляции**: detect (bbox), segment (bbox + masks), pose (bbox + keypoints)
- **PySide6 GUI** с переключателем задачи и пресетами параметров под каждую задачу
- **NDJSON → YOLO dataset**: автоматическая конвертация Ultralytics NDJSON-экспорта в локальный YOLO-датасет со скачиванием изображений (параллельно, до 32 worker'ов)
- **OpenVINO INT8**: экспорт в INT8-квантованную модель OpenVINO
- **Export parity check**: сравнение предсказаний PyTorch vs ONNX/OpenVINO на валидационных данных
- **Детальное логирование**: структурированные `[DISTILL]` и `[PROGRESS]` логи каждого этапа

## Зависимости

Управление через `uv`:
- Python 3.11–3.12
- `ultralytics>=8.3.0`, `torch>=2.7.0` (CUDA 12.8), `torchvision>=0.22.0`
- `PySide6>=6.8.0` (GUI)
- `onnx`, `onnxslim`, `openvino` (экспорт)

## Быстрый старт

```bash
# Установка uv (если нет)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Создание окружения
uv sync

# Проверка CUDA
uv run python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# Запуск GUI
uv run distill_yolo26_seg_gui.py
```

## CLI: примеры

### Segment (стандартная дистилляция сегментации)

```bash
uv run distill_yolo26_seg.py \
  --task segment \
  --teacher yolo26x-seg.pt \
  --student-init yolo26n-seg.pt \
  --data /path/to/data.yaml \
  --imgsz 320 --epochs 120 --batch 16 --workers 8 --device auto \
  --outdir ./runs_distill
```

### Detect (bbox) через NDJSON

```bash
uv run distill_yolo26_seg.py \
  --task detect \
  --teacher yolo26x.pt \
  --data /path/to/detect-export.ndjson \
  --imgsz 320 --epochs 120 --batch 16 --workers 8 --device auto \
  --outdir ./runs_detect_distill
```

### Pose (keypoints) через NDJSON

```bash
uv run distill_yolo26_seg.py \
  --task pose \
  --teacher yolo26x-pose.pt \
  --data /path/to/pose-export.ndjson \
  --imgsz 416 --epochs 350 --batch 16 --workers 8 --device auto \
  --outdir ./runs_pose_distill
```

### OpenVINO INT8

```bash
uv run distill_yolo26_seg.py \
  --task detect \
  --teacher yolo26x.pt --data ... \
  --export onnx openvino --openvino-int8 \
  --outdir ./runs_int8
```

## NDJSON → YOLO dataset

При передаче `.ndjson` файла в `--data`, конвертация происходит автоматически перед стартом обучения:

- `utils/detect_ndjson_to_yolo.py` — конвертация detect NDJSON
- `utils/pose_ndjson_to_yolo.py` — конвертация pose NDJSON
- `--ndjson-download-workers` (по умолчанию 32) — параллельная загрузка изображений

Если загрузка хотя бы одного изображения не удалась — обучение не начнётся (для detect/pose).

### Объединение нескольких NDJSON датасетов

```bash
uv run python utils/merge_detect_ndjson_to_yolo.py --clean --workers 24
```

Скрипт скачает изображения, объединит классы по имени, перемапит id в YOLO-labels и создаст датасет:

```
merge/merged_detect_yolo/data.yaml
```

Итоговые классы для текущих файлов: `mini_game`, `bobber`, `mini_fish`, `nick`, `self`.

## KD Loss: как работает

Для каждой задачи используется свой критерий дистилляции:

| Задача   | Класс criterion               | KD-термы                          |
|----------|-------------------------------|-----------------------------------|
| detect   | `DetectionDistillationCriterion`  | cls + box                         |
| segment  | `SegmentationDistillationCriterion` | cls + box + mask + proto + semseg |
| pose     | `PoseDistillationCriterion`       | cls + box + kpt                   |

Общая формула: `loss = α · hard_loss + (1-α) · kd_gain · KD`, где KD — взвешенная сумма лоссов по головам.

## GUI (PySide6)

GUI написан на PySide6 (Qt) с переключением задачи через выпадающий список:

- **Task**: выбор задачи (detect/segment/pose)
- **Presets**: кнопки "Apply detect defaults" и "Apply pose accuracy defaults" — подставляют оптимальные гиперпараметры
- **KD params**: все веса для каждой задачи, включая `kd_kpt_w` для pose
- **Export**: ONNX (FP32), OpenVINO (FP16/INT8)
- **NDJSON**: автоматическая подготовка датасета перед запуском, с прогресс-баром в логе
- **Heartbeat**: если процесс не выводит лог >60с, GUI пишет `[GUI] Process is still running`
- **Сохранение настроек**: все поля сохраняются в `~/.distill_yolo26_seg_gui.json`

### Preset: Pose accuracy

```yaml
student: yolo26n-pose.pt
imgsz: 416
epochs: 350  patience: 80  batch: 16
alpha: 0.65  temperature: 3.0  kd_gain: 1.1
kd_box_w: 0.40  kd_kpt_w: 0.90
OpenVINO: FP16 (INT8 выключен)
```

### Preset: Detect

```yaml
student: yolo26n.pt
imgsz: 320  epochs: 120  batch: 16
alpha: 0.8  temperature: 2.0  kd_gain: 1.0
kd_cls_w: 1.0  kd_box_w: 0.35  (mask/proto/semseg/kpt = 0)
```

## Структура проекта

```
├── distill_yolo26_seg.py           # CLI entrypoint (detect/segment/pose)
├── distill_yolo26_seg_gui.py       # PySide6 GUI
├── utils/
│   ├── yolo_distill.py             # Ядро: конфиг, критерии, трейнеры, экспорт, parity check
│   ├── detect_ndjson_to_yolo.py    # Detect NDJSON → YOLO dataset
│   ├── pose_ndjson_to_yolo.py      # Pose NDJSON → YOLO dataset
│   └── merge_detect_ndjson_to_yolo.py # Объединение нескольких detect NDJSON
├── pyproject.toml                  # Зависимости (uv, CUDA 12.8)
├── .gitignore
└── README.md
```

## Чтение логов

Ищи строки с префиксами:

- `[DISTILL] Stage=cli_start|prepare|train_begin|train_done|export_begin|export_done|parity_begin|parity_done`
- `[PROGRESS] Stage=train_start|epoch_start|epoch_end|val_end|train_end`
- `[PROGRESS] Stage=epoch_end epoch=X/Y train_loss=... kd_loss=... lr=...`
- `[PROGRESS] Stage=val_end mAP50=... mAP50-95=...`
- `[GUI] Process is still running; no output for ...s`
- `[DATA] Converting ... NDJSON ... downloaded X/Y`
- `[DATA] Prepared dataset: images_total=... downloaded=... failed=...`
- `[DEVICE] auto -> 0 (CUDA)`

## Экспорт и проверка

После обучения автоматически экспортируются ONNX (FP32) и OpenVINO (FP16 по умолчанию, или INT8 с флагом `--openvino-int8`).

Parity check сравнивает количество детекций, confidence и (в зависимости от задачи) маски или keypoints между PyTorch и экспортированными моделями.

## Системные требования (NVIDIA/CUDA)

```bash
nvidia-smi
sudo pacman -Syu --needed nvidia-utils cuda cudnn
sudo reboot
```

Для wheel-сборок PyTorch из `cu128` достаточно совместимого драйвера NVIDIA, полный CUDA toolkit не обязателен.

### Диагностика

```bash
uv run python -c "import ultralytics, onnx, openvino, torch; print('ok')"
```
