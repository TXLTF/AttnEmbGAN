# AttnEmbGAN

AttnEmbGAN is a deep learning model for generating multi-stitch embroidery images using a residual attention network.

## File Structure

- `data/`: Contains dataset-related code.
- `models/`: Contains model definitions for the generator and discriminator.
- `utils/`: Contains utility functions and logging.
- `train.py`: Script to train the model.
- `test.py`: Script to test the model.

## How to Run

1. Prepare the dataset and place it in the `data/` directory.
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Train the model:
    ```bash
    python train.py
    ```
4. Test the model:
    ```bash
    python test.py
    ```

## References

- [AttentionGAN](https://arxiv.org/abs/1903.12296)
- [CycleGAN](https://arxiv.org/abs/1703.10593)