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

### Почему в GUI не видно `pacman` логов

- Команда `sudo pacman -Syu --needed nvidia-utils cuda cudnn` выполняется отдельно от процесса обучения.
- Лог в GUI показывает только вывод процесса дистилляции, а не системные команды установки пакетов.

### Как читать этапы обучения в логе

Ищи строки с префиксами:

- `[DISTILL] Stage=prepare|train_begin|train_done|export_begin|export_done|parity_begin|parity_done`
- `[PROGRESS] Stage=epoch_start epoch=X/Y`
- `[PROGRESS] Stage=epoch_end epoch=X/Y train_loss=... kd_loss=... lr=...`
- `[PROGRESS] Stage=val_end mAP50=... mAP50-95=...`

## Рекомендованные пресеты KD (цель: качество + OpenVINO)

Ниже настройки для GUI, если основная цель - точность ближе к учителю при `imgsz=320`:

- `epochs=700`
- `batch=16` (или больше, если VRAM позволяет)
- `patience=100`
- `alpha=0.65`
- `temperature=3.0`
- `kd_gain=1.1`
- `kd_cls_w=1.0`
- `kd_box_w=0.40`
- `kd_mask_w=0.75`
- `kd_proto_w=0.70`
- `kd_semseg_w=0.50`

Для более агрессивной скорости на инференсе после обучения можно пробовать OpenVINO INT8-квантизацию (обычно быстрее, но иногда с просадкой mAP).
