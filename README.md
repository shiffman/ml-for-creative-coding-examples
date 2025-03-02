# Code Examples for Machine Learning for Creative Coding

This repository contains sketches demonstrating the full lifecycle of a machine learning project: how to **collect data**, **train a model**, and **deploy the trained model** using [ml5.js](https://ml5js.org/) and [TensorFlow.js](https://www.tensorflow.org/js).

The examples are part of the curriculum for the **Machine Learning for Creative Coding** class at NYU's ITP/IMA program (Spring 2025).

More details: [Machine Learning for Creative Coding Repository](https://github.com/shiffman/ML-for-Creative-Coding)

## 📂 Pose Classification and Regression

This repository contains three projects that use handpose data as inputs. There are ml5.js demonstrations for both handpose classification and a regression model controlling a single value. Additionally, a TensorFlow.js version is included for direct comparison between high-level ml5.js functionality and lower-level TensorFlow.js customization.

Each example includes three parts located in `ml5js/pose-classifier`, `ml5js/pose-regression`, and `tfjs/pose-classifier`:

1. **`1-save-data/`** – Collects and saves handpose data.
2. **`2-train-model/`** – Trains a neural network on collected data.
3. **`3-deploy-model/`** – Loads the trained model and classifies gestures in real-time.

## 📂 Additional TensorFlow.js Demonstrations

These examples demonstrate direct usage of TensorFlow.js for applications extending beyond the scope of ml5.js.

### Feature Extraction and Custom Classification

These examples leverage MobileNet feature vectors for various applications:

- **`tfjs/image-similarity/`** – Finds visually similar images using MobileNet embeddings.
- **`tfjs/teachable-machine/`** – Builds a custom interactive classifier with MobileNet embeddings.

> **Note:** To run the `image-similarity` example, include an "images" directory with filenames `0.jpg`, `1.jpg`, etc. You can download sample images from the [imagenet-sample-images repository](https://github.com/EliSchwartz/imagenet-sample-images) and rename them using the provided helper script `rename-files.js` (included directly within the `tfjs/image-similarity/` directory).

### Autoencoder

- **`tfjs/autoencoder/`** – Demonstrates a basic autoencoder implementation. Requires a dataset of images (28x28 pixels) in the `data/` folder named sequentially (e.g., `square0000.png`, `square0001.png`). Use the Processing sketch included in `tfjs/autoencoder/training_generator` to generate images!
