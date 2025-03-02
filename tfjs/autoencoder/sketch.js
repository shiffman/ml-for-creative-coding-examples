// TensorFlow.js Autoencoder Example
// Demonstrates training a simple autoencoder to reconstruct 28x28 images.
// https://github.com/shiffman/ml-for-creative-coding-examples
// https://github.com/shiffman/ML-for-Creative-Coding

// Autoencoder model
let autoencoder;
// Array to store images for training
let images = [];

// Image resolution (28x28 pixels)
const IMAGE_SIZE = 28;

// Load training images
function preload() {
  for (let i = 0; i < 100; i++) {
    let filename = `data/square${String(i).padStart(4, '0')}.png`;
    images.push(loadImage(filename));
  }
  console.log('Images loaded');
}

async function setup() {
  createCanvas(IMAGE_SIZE * 10 * 2, IMAGE_SIZE * 10);
  background(200);

  // Create the autoencoder
  await createAutoencoder();
  // Train the autoencoder
  await trainAutoencoder();
  // Test the autoencoder on one image
  testAutoencoder(images[0]);
}

// Define autoencoder architecture and compile model
async function createAutoencoder() {
  autoencoder = tf.sequential();

  // Encoder layers (compressing input data)
  autoencoder.add(
    tf.layers.dense({ inputShape: [IMAGE_SIZE * IMAGE_SIZE], units: 64, activation: 'relu' })
  );
  autoencoder.add(tf.layers.dense({ units: 32, activation: 'relu' }));

  // We could add more layers to compress further!
  // autoencoder.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  // Then more to expand again
  // autoencoder.add(tf.layers.dense({ units: 32, activation: 'relu' }));

  // Decoder layers (reconstructing original data)
  autoencoder.add(tf.layers.dense({ units: 64, activation: 'relu' }));

  // Sigmoid activation for pixel values between 0 and 1
  autoencoder.add(tf.layers.dense({ units: IMAGE_SIZE * IMAGE_SIZE, activation: 'sigmoid' }));

  // Compile model with Adam optimizer and mean squared error loss
  autoencoder.compile({
    optimizer: tf.train.adam(0.005),
    loss: 'meanSquaredError',
  });
}

async function trainAutoencoder() {
  // Prepare input data
  let inputs = [];

  // Convert each image to grayscale pixel array
  for (let img of images) {
    img.loadPixels();
    let imgData = [];
    for (let i = 0; i < img.pixels.length; i += 4) {
      imgData.push(img.pixels[i] / 255); // Normalize pixels
    }
    inputs.push(imgData);
  }

  // Convert image data to tensor
  const xs = tf.tensor2d(inputs);

  console.log('Training...');
  // Train autoencoder using the same data (!!!!) for input and output
  await autoencoder.fit(xs, xs, {
    epochs: 50,
    batchSize: 2,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}, Loss: ${logs.loss}`);
      },
    },
  });
  console.log('Training complete.');
}

// Test the autoencoder's reconstruction
async function testAutoencoder(testImg) {
  // Convert test image to input tensor
  testImg.loadPixels();
  let inputArray = [];
  for (let i = 0; i < testImg.pixels.length; i += 4) {
    // Normalize pixels
    inputArray.push(testImg.pixels[i] / 255);
  }
  const inputTensor = tf.tensor2d([inputArray]);

  // Predict reconstructed image
  let outputTensor = autoencoder.predict(inputTensor);
  outputTensor = outputTensor.reshape([IMAGE_SIZE, IMAGE_SIZE]);

  // Convert tensor output back to pixel data array
  const outputData = await outputTensor.array();

  // Display original image
  image(testImg, 0, 0, IMAGE_SIZE * 10, IMAGE_SIZE * 10);

  // Make a p5 image from output data
  let outputImage = createImage(IMAGE_SIZE, IMAGE_SIZE);
  outputImage.loadPixels();
  for (let y = 0; y < IMAGE_SIZE; y++) {
    for (let x = IMAGE_SIZE; x < IMAGE_SIZE * 2; x++) {
      let val = outputData[y][x - IMAGE_SIZE] * 255;
      outputImage.set(x, y, color(val));
    }
  }
  outputImage.updatePixels();

  // Show the output image
  image(outputImage, IMAGE_SIZE * 10, 0, IMAGE_SIZE * 10, IMAGE_SIZE * 10);

  // Dispose of tensors to free memory (ml5.js handles this internally)
  inputTensor.dispose();
  outputTensor.dispose();
}
