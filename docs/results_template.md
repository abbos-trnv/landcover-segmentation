# Шаблон фиксации результатов

Этот файл заполняется только значениями, сохранёнными в `artifacts/runs/`.
Не переносить показатели из stdout вручную без соответствующего CSV/JSON.

## Основной controlled benchmark

| Decoder | Best epoch by validation | Test foreground mIoU | Buildings IoU | Woodland IoU | Water IoU | Roads IoU | Macro Dice | Parameters, M | Latency, ms | Peak VRAM, MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net |  |  |  |  |  |  |  |  |  |  |
| FPN |  |  |  |  |  |  |  |  |  |  |
| DeepLabV3+ |  |  |  |  |  |  |  |  |  |  |

## Обязательные графики

- loss и validation foreground mIoU по эпохам для каждой модели;
- примеры: входной тайл, истинная маска, предсказания трёх моделей;
- boxplot или столбчатая диаграмма foreground mIoU по шести test-сценам;
- точечная диаграмма «test foreground mIoU — latency».

## Правила интерпретации

- Лучшую эпоху выбираем только по validation foreground mIoU.
- Test используется после фиксации конфигурации, один раз на модель.
- Нельзя объявлять модель лучшей только по Pixel Accuracy.
- Разницу mIoU интерпретируем вместе с IoU редких классов, latency, VRAM и
  разбросом качества по шести тестовым сценам.
