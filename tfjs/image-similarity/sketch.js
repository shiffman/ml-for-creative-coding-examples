let images = [];
let features = [];

let similarImage = null;

// MobileNet input size
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;

function preload() {
  for (let i = 0; i < 1000; i++) {
    images[i] = loadImage(`images/${i}.jpg`);
  }
}

let finding = false;

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();
  createButton('find similar image').mousePressed(findImage);
  loadMobileNet();
}

async function loadMobileNet() {
  const URL =
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });
  console.log('model loaded');
  await extractFeatures();
}

async function extractFeatures() {
  for (let i = 0; i < images.length; i++) {
    images[i].loadPixels();
    features[i] = await tf.tidy(() => {
      let tensorFeatures = getFeatures(images[i].canvas);
      let rawArray = tensorFeatures.arraySync();
      tensorFeatures.dispose();
      return rawArray;
    });
    console.log(`Calculated features for image ${i + 1} of ${images.length}`);
  }
}

function findImage() {
  tf.tidy(() => {
    let videoFeatures = getFeatures(video.elt).arraySync();

    let highestScore = -Infinity;
    let highestIndex = -1;

    for (let i = 0; i < features.length; i++) {
      let similarity = cosineSimilarity(features[i], videoFeatures);
      if (similarity > highestScore) {
        highestScore = similarity;
        highestIndex = i;
      }
    }
    similarImage = images[highestIndex];
    console.log(`Most similar image: ${highestIndex}, Score: ${highestScore}`);
  });
}

function getFeatures(img) {
  return tf.tidy(() => {
    let tensor = tf.browser
      .fromPixels(img)
      .resizeBilinear([MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true)
      .div(255)
      .expandDims();
    let features = mobilenet.predict(tensor);
    return features.squeeze();
  });
}

function dotProduct(vecA, vecB) {
  return vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
}

function magnitude(vec) {
  return Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
}

function cosineSimilarity(vecA, vecB) {
  return dotProduct(vecA, vecB) / (magnitude(vecA) * magnitude(vecB));
}

function draw() {
  background(0);
  image(video, 0, 0, width, height);

  if (similarImage) {
    image(similarImage, 0, 0, 100, 100);
  }
}
