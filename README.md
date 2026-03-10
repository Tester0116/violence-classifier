# Violence Video Classifier

Бинарная классификация видео: **Violence** / **NonViolence**.

Датасет: [Real-Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) — 2000 видео, по 1000 на класс.

Точность: **94% test**

---

## Детали реализации

**Базовая модель:** MobileNetV3-Large с весами ImageNet V2. Оригинальный классификатор на 1000 классов заменён на новый под 2 класса.

**Обработка видео:** из каждого видео берётся 16 кадров равномерно → каждый кадр прогоняется через backbone → векторы усредняются → один вектор на видео → классификатор.

**Классификатор:**
```
Dropout(0.4) → Linear(960→512) → Hardswish → Dropout(0.2) → Linear(512→2)
```

**Стратегия fine-tuning:** первые 3 эпохи backbone заморожен, обучается только классификатор. С 4-й эпохи размораживается вся сеть. Это защищает предобученные веса от разрушения на старте.

**Ключевые параметры:** AdamW, LR backbone 3e-5 / classifier 3e-4, OneCycleLR, batch 16, dropout 0.4, label smoothing 0.05, early stopping (patience 6).

**Балансировка:** WeightedRandomSampler — оба класса видятся одинаково часто.

**Аугментации:** RandomCrop, HorizontalFlip, ColorJitter.

---

## Результаты тестирования

**Валидация на датасете (test split, 100 видео):**

| Класс       | Precision | Recall | F1   |
|-------------|-----------|--------|------|
| NonViolence | 0.89      | 1.00   | 0.94 |
| Violence    | 1.00      | 0.88   | 0.94 |
| **Overall** |           |        | **0.94** |

Confusion matrix:
```
[[50  0]
 [ 6 44]]
```
Из 100 видео ошиблась только на 6 — все ошибки это Violence распознанный как NonViolence.

**Тестирование на реальных видео из интернета:**

| Видео | Ожидалось | Предсказано | Верно |
|-------|-----------|-------------|-------|
| Драка на улице (YouTube) | Violence | Violence | ✅ |
| Футбольный матч | NonViolence | NonViolence | ✅ |
| Уличный конфликт | Violence | Violence | ✅ |
| Прогулка в парке | NonViolence | NonViolence | ✅ |
| Массовая драка | Violence | Violence | ✅ |

---

## Файлы

```
├── training.ipynb                   — обучение и тестирование
├── checkpoints/
│   └── best_model.safetensors       — веса финальной модели
├── history.json                     — loss/accuracy по эпохам
└── README.md
```

---

## Как обучить

1. Открыть `training.ipynb` в [Kaggle](https://kaggle.com)
2. Add Data → `real-life-violence-situations-dataset`
3. Session options → Accelerator → GPU T4
4. Run All (~2 часа)

---

## Как использовать модель

```python
import torch, torch.nn as nn
from safetensors.torch import load_file
import torchvision.models as models

class ViolenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.mobilenet_v3_large(weights=None)
        self.features = bb.features
        self.avgpool  = bb.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(960, 512),
            nn.Hardswish(), nn.Dropout(0.2), nn.Linear(512, 2),
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        f = torch.flatten(self.avgpool(self.features(x)), 1)
        return self.classifier(f.view(B, T, -1).mean(1))

model = ViolenceClassifier()
model.load_state_dict(load_file("checkpoints/best_model.safetensors"))
model.eval()
# 0 = NonViolence, 1 = Violence
```
