let autoencoder;
let images = [];
const IMAGE_SIZE = 28;

function preload() {
  for (let i = 0; i < 100; i++) {
    let filename = `data/square${String(i).padStart(4, '0')}.png`;
    images.push(loadImage(filename));
  }
  console.log('images loaded');
}

async function setup() {
  createCanvas(IMAGE_SIZE * 10 * 2, IMAGE_SIZE * 10);
  background(200);

  await createAutoencoder();
  // training
  await trainAutoencoder();

  testAutoencoder(images[0]);
}

async function createAutoencoder() {
  autoencoder = tf.sequential();

  autoencoder.add(
    tf.layers.dense({ inputShape: [IMAGE_SIZE * IMAGE_SIZE], units: 64, activation: 'relu' })
  );
  autoencoder.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  autoencoder.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  autoencoder.add(tf.layers.dense({ units: IMAGE_SIZE * IMAGE_SIZE, activation: 'sigmoid' }));
  autoencoder.compile({
    optimizer: tf.train.adam(0.005),
    loss: 'meanSquaredError',
  });
}

async function trainAutoencoder() {
  let inputs = [];

  for (let img of images) {
    img.loadPixels();
    let imgData = [];
    for (let i = 0; i < img.pixels.length; i += 4) {
      imgData.push(img.pixels[i] / 255);
    }
    inputs.push(imgData);
  }

  const xs = tf.tensor2d(inputs);

  console.log('Training...');
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

async function testAutoencoder(testImg) {
  let inputArray = [];
  testImg.loadPixels();
  for (let i = 0; i < testImg.pixels.length; i += 4) {
    inputArray.push(testImg.pixels[i] / 255);
  }
  const inputTensor = tf.tensor2d([inputArray]);
  let outputTensor = autoencoder.predict(inputTensor);
  outputTensor = outputTensor.reshape([IMAGE_SIZE, IMAGE_SIZE]);

  const outputData = await outputTensor.array();

  image(testImg, 0, 0, IMAGE_SIZE * 10, IMAGE_SIZE * 10);

  let outputImage = createImage(IMAGE_SIZE, IMAGE_SIZE);
  outputImage.loadPixels();
  for (let y = 0; y < IMAGE_SIZE; y++) {
    for (let x = IMAGE_SIZE; x < IMAGE_SIZE * 2; x++) {
      let val = outputData[y][x - IMAGE_SIZE] * 255;
      outputImage.set(x, y, color(val));
    }
  }
  outputImage.updatePixels();
  image(outputImage, IMAGE_SIZE * 10, 0, IMAGE_SIZE * 10, IMAGE_SIZE * 10);

  inputTensor.dispose();
  outputTensor.dispose();
}
