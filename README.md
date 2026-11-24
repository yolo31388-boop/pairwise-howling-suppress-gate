# pairwise-howling-suppress-gate

## 领域

游戏、图形与多媒体

## 主题

啸叫抑制与噪声门

## 说明

自适应陷波、谐波判定与门限跟随。

初始状态下 `media/howling_suppress_gate.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
