// TensorFlow.js Teachable Machine Example
// Demonstrates transfer learning using MobileNet feature vectors.
// https://github.com/shiffman/ml-for-creative-coding-examples
// https://github.com/shiffman/ML-for-Creative-Coding
// Reference: https://codelabs.developers.google.com/tensorflowjs-transfer-learning-teachable-machine

// UI elements
let statusP;
let video;
let trainButton;
let dataButtons = [];

// MobileNet input size constants (ml5.js handles this internally)
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;

// Labels for classification
const LABELS = ['A', 'B'];

// Arrays for training data (ml5.neuralNetwork manages data collection internally)
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];

// Models
let mobilenet;
let teachableModel;

function setup() {
  createCanvas(640, 480);

  // Video setup
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  // Status (e.g., loading progress, training status)
  statusP = createP('Loading MobileNet v3...');

  // Button to start training
  trainButton = createButton('Train');
  trainButton.mousePressed(trainModel);

  // Buttons for each label collecting data
  for (let i = 0; i < LABELS.length; i++) {
    const btn = createButton(`Gather ${LABELS[i]}`);
    btn.attribute('index', i);
    btn.mousePressed(collectData);
    dataButtons.push(btn);
    // Initialize counts
    examplesCount[i] = 0;
  }

  // Define a custom classifier for MobileNet embeddings as inputs
  teachableModel = tf.sequential();
  teachableModel.add(tf.layers.dense({ inputShape: [1024], units: 128, activation: 'relu' }));
  teachableModel.add(tf.layers.dense({ units: LABELS.length, activation: 'softmax' }));

  // Compile the model (in ml5.js, this is automatic and abstracted)
  teachableModel.compile({
    // Use Adam optimizer
    optimizer: 'adam',
    // Use binaryCrossentropy if there are only two labels
    loss: LABELS.length === 2 ? 'binaryCrossentropy' : 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Load pre-trained MobileNet for feature extraction
  loadMobileNet();
}

function draw() {
  background(0);
  image(video, 0, 0, width, height);
}

// Load MobileNet model from TensorFlow Hub
async function loadMobileNet() {
  const URL =
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });
  statusP.html('MobileNet v3 loaded');
}

// Collect training data from the video
function collectData() {
  let index = parseInt(this.elt.getAttribute('index'));

  // Get MobileNet feature vectors
  const imageFeatures = getFeatures(video.elt);

  // Add data to training arrays
  trainingDataInputs.push(imageFeatures);
  trainingDataOutputs.push(index);
  // Increment count for this label
  examplesCount[index]++;

  // Update status display
  let statusStr = '';
  for (let i = 0; i < LABELS.length; i++) {
    statusStr += `${LABELS[i]} data count: ${examplesCount[i]}. `;
  }
  statusP.html(statusStr);
}

// Extract features from an image using MobileNet
function getFeatures(img) {
  // If you wrap a bunch of code inside tf.tidy(),
  // TensorFlow.js will automatically clean up all the memory used by tensors
  return tf.tidy(() => {
    // Convert image to tensor
    let videoFrameAsTensor = tf.browser.fromPixels(img);
    // Resize image (ml5.js does this automatically)
    let resizedTensorFrame = tf.image.resizeBilinear(
      videoFrameAsTensor,
      [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
      true
    );
    // Normalize pixel values between 0 and 1
    let normalizedTensorFrame = resizedTensorFrame.div(255);
    // Extract features using MobileNet
    let features = mobilenet.predict(normalizedTensorFrame.expandDims());
    // Remove unnecessary batch dimension so you just get one array
    return features.squeeze();
  });
}

// Train the custom classifier using collected feature vectors
async function trainModel() {
  // Shuffle training data
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

  // Convert labels to one-hot tensors
  const outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
  const oneHotOutputs = tf.oneHot(outputsAsTensor, LABELS.length);
  const inputsAsTensor = tf.stack(trainingDataInputs);

  // Train the model
  await teachableModel.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize: 8,
    epochs: 50,
    callbacks: { onEpochEnd: (epoch, logs) => console.log('Epoch', epoch, logs) },
  });

  // Dispose tensors to free memory (ml5.js handles memory management automatically)
  // See other parts of the code that use tf.tidy() to manage memory
  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();

  // Begin predictions after training
  classifyImage();
}

// Classify the current video frame
function classifyImage() {
  // If you wrap a bunch of code inside tf.tidy(),
  // TensorFlow.js will automatically clean up all the memory used by tensors
  tf.tidy(() => {
    // Get the features
    const imageFeatures = getFeatures(video.elt);
    // Make a prediction
    let prediction = teachableModel.predict(imageFeatures.expandDims()).squeeze();

    // Find the index with the highest confidence
    let highestIndex = prediction.argMax().arraySync();
    let predictionArray = prediction.arraySync();

    // Display label and confidence
    statusP.html(
      `Prediction: ${LABELS[highestIndex]} ` +
        `(${Math.floor(predictionArray[highestIndex] * 100)}% confidence)`
    );
  });

  // Let's go again!
  setTimeout(classifyImage, 10);
}
