// TensorFlow.js Image Similarity Example
// Demonstrates finding visually similar images using MobileNet feature vectors.
// https://github.com/shiffman/ml-for-creative-coding-examples
// https://github.com/shiffman/ML-for-Creative-Coding

// Arrays for images and corresponding feature vectors
let images = [];
let features = [];

// Image found to be most similar!
let similarImage = null;

// MobileNet input size constants (handled internally in ml5.js)
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;

// Preload images
function preload() {
  for (let i = 0; i < 1000; i++) {
    images[i] = loadImage(`images/${i}.jpg`);
  }
}

function setup() {
  createCanvas(640, 480);

  // Video
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  // Button for similarity search
  createButton('find similar image').mousePressed(findImage);

  // Load MobileNet model for feature extraction
  loadMobileNet();
}

// Load MobileNet model asynchronously from TensorFlow Hub
async function loadMobileNet() {
  const URL =
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });
  console.log('MobileNet model loaded');

  // Pre-calculate and store feature vectors for all loaded images
  // This can take a while, it might be better to do this as a separate process...
  // save the features in a JSON file, and load here.
  await extractFeatures();
}

// Calculate feature vectors for all images
async function extractFeatures() {
  for (let i = 0; i < images.length; i++) {
    images[i].loadPixels();
    features[i] = await tf.tidy(() => {
      // Extract features for each image
      let tensorFeatures = getFeatures(images[i].canvas);
      // Convert to array
      let rawArray = tensorFeatures.arraySync();
      // No need for tensor anymore
      tensorFeatures.dispose();
      return rawArray;
    });
    console.log(`Features calculated for image ${i + 1} of ${images.length}`);
  }
}

// Find the most similar image!
function findImage() {
  // Use tf.tidy() to manage memory
  tf.tidy(() => {
    // Get features of current video frame
    let videoFeatures = getFeatures(video.elt).arraySync();

    // Track highest similarity and corresponding image index
    let highestScore = -Infinity;
    let highestIndex = -1;

    for (let i = 0; i < features.length; i++) {
      // Calculate similarity between stored image features and current frame
      let similarity = cosineSimilarity(features[i], videoFeatures);
      if (similarity > highestScore) {
        highestScore = similarity;
        highestIndex = i;
      }
    }

    // Set the most similar image!
    similarImage = images[highestIndex];
    console.log(`Most similar image: ${highestIndex}, Score: ${highestScore}`);
  });
}

// Extract features from an image using MobileNet
function getFeatures(img) {
  // tf.tidy() manages memory used by tensors (ml5.js does this internally)
  return tf.tidy(() => {
    // Convert image to tensor
    let tensor = tf.browser
      .fromPixels(img)
      .resizeBilinear([MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true)
      .div(255) // Normalize pixels between 0 and 1
      .expandDims(); // Add batch dimension

    // Extract features using MobileNet
    let features = mobilenet.predict(tensor);

    // Remove batch dimension
    return features.squeeze();
  });
}

// Helper functions to calculate cosine similarity
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
  // Show the video
  image(video, 0, 0, width, height);

  // Show the similar image
  if (similarImage) {
    image(similarImage, 0, 0, 100, 100);
  }
}
