# YOLO26 Distillation Workspace

Проект для дистилляции сегментационной модели YOLO26 в nano-версию и экспорта в ONNX/OpenVINO.

## Что уже настроено

- Управление зависимостями через `uv`.
- `pyproject.toml` с CUDA-сборками `torch/torchvision` (индекс PyTorch `cu128`).
- GUI-лаунчер и CLI-скрипт дистилляции.

## Быстрый старт

1. Установить `uv` (если еще нет):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Создать/синхронизировать окружение:

```bash
uv sync
```

3. Проверить CUDA в окружении:

```bash
uv run python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

4. Запуск GUI:

```bash
uv run distill_yolo26_seg_gui.py
```

5. Запуск CLI (пример):

```bash
uv run distill_yolo26_seg.py \
  --teacher /abs/path/to/teacher.pt \
  --student-init yolo26n-seg.pt \
  --data /abs/path/to/data.yaml \
  --imgsz 320 --epochs 120 --batch 16 --workers 8 --device auto \
  --outdir ./runs_distill
```

## Системная часть (NVIDIA/CUDA) для Arch/CachyOS

Проверь драйвер:

```bash
nvidia-smi
```

Если драйвер не установлен, поставь базовый стек:

```bash
sudo pacman -Syu --needed nvidia-utils cuda cudnn
sudo reboot
```

После перезагрузки снова проверь:

```bash
nvidia-smi
```

Важно:

- Для wheel-сборок PyTorch из `cu128` достаточно совместимого драйвера NVIDIA.
- Полный `cuda` toolkit полезен для разработческих задач и некоторых экспортов, но не всегда обязателен только для обучения/инференса через wheel.

## Диагностика

Проверка ключевых пакетов:

```bash
uv run python -c "import ultralytics, onnx, openvino, torch; print('ok')"
```

Если GUI пишет про недоступный PyTorch/CUDA:

- Запускай GUI именно через `uv run distill_yolo26_seg_gui.py`.
- В GUI оставляй `device=auto`, тогда будет проверка CUDA в целевом интерпретаторе.
