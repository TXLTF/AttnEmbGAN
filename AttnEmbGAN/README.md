# AttnEmbGAN

AttnEmbGAN 是一个基于残差注意力网络的深度学习模型，用于生成多针迹刺绣图像。

## 文件结构

- `data/`: 包含数据集相关的代码。
- `models/`: 包含生成器和判别器的模型定义。
- `utils/`: 包含工具函数和日志记录。
- `train.py`: 训练模型的脚本。
- `test.py`: 测试模型的脚本。

## 运行方法

1. 准备数据集并将其放置在 `data/` 目录中。
2. 安装所需的包：
    ```bash
    pip install -r requirements.txt
    ```
3. 训练模型：
    ```bash
    python train.py
    ```
4. 测试模型：
    ```bash
    python test.py
    ```

## 参考文献

- [AttentionGAN](https://arxiv.org/abs/1903.12296)
- [CycleGAN](https://arxiv.org/abs/1703.10593)