// Reference
// https://codelabs.developers.google.com/tensorflowjs-transfer-learning-teachable-machine

// interface stuff
let statusP;
let video;
let trainButton;
let dataButtons = [];

// MobileNet input size
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;

// Two labels
const LABELS = ['A', 'B'];

// Arrays that store training data
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];

// The two models
let mobilenet;
let teachableModel;

function setup() {
  createCanvas(640, 480);

  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  // Status element
  statusP = createP('Loading MobileNet v3...');

  // "Train" button
  trainButton = createButton('Train');
  trainButton.mousePressed(trainModel);

  // Create data-collector buttons for each class
  for (let i = 0; i < LABELS.length; i++) {
    const btn = createButton(`Gather ${LABELS[i]}`);
    btn.attribute('index', i);
    btn.mousePressed(collectData);
    dataButtons.push(btn);
    // Start a zero count for each label
    examplesCount[i] = 0;
  }

  // Define our classification architecture
  teachableModel = tf.sequential();
  teachableModel.add(tf.layers.dense({ inputShape: [1024], units: 128, activation: 'relu' }));
  teachableModel.add(tf.layers.dense({ units: LABELS.length, activation: 'softmax' }));
  teachableModel.compile({
    optimizer: 'adam',
    loss: LABELS.length === 2 ? 'binaryCrossentropy' : 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Load the MobileNet feature extraction model
  loadMobileNet();
}

function draw() {
  background(0);
  image(video, 0, 0, width, height);
}

// Asynchronously load the MobileNet model from TF Hub
async function loadMobileNet() {
  const URL =
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });
  statusP.html('MobileNet v3 loaded');
}

function collectData() {
  let index = parseInt(this.elt.getAttribute('index'));

  // capture image features from the current video frame
  const imageFeatures = getFeatures(video.elt);

  trainingDataInputs.push(imageFeatures);
  trainingDataOutputs.push(index);
  examplesCount[index]++;

  let statusStr = '';
  for (let i = 0; i < LABELS.length; i++) {
    statusStr += `${LABELS[i]} data count: ${examplesCount[i]}. `;
  }
  statusP.html(statusStr);
}

function getFeatures(img) {
  return tf.tidy(() => {
    // Grab pixels from video
    let videoFrameAsTensor = tf.browser.fromPixels(img);
    // Resize to 224x224
    let resizedTensorFrame = tf.image.resizeBilinear(
      videoFrameAsTensor,
      [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
      true
    );
    // Normalize between 0 and 1
    let normalizedTensorFrame = resizedTensorFrame.div(255);
    // Get the MobileNet features
    let features = mobilenet.predict(normalizedTensorFrame.expandDims());
    // Remove batch dimension and return
    return features.squeeze();
  });
}

async function trainModel() {
  // Shuffle the data
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

  // Convert to tensors
  const outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
  const oneHotOutputs = tf.oneHot(outputsAsTensor, LABELS.length);
  const inputsAsTensor = tf.stack(trainingDataInputs);

  // Train
  let results = await teachableModel.fit(inputsAsTensor, oneHotOutputs, {
    // Shuffle per epoch
    shuffle: true,
    batchSize: 5,
    epochs: 10,
    callbacks: { onEpochEnd: (epoch, logs) => console.log('Epoch', epoch, logs) },
  });

  // dispose memory! could use tf.tidy() instead?
  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();

  // After training, begin predicting live
  classifyImage();
}

function classifyImage() {
  tf.tidy(() => {
    const imageFeatures = calculateFeaturesOnCurrentFrame();
    let prediction = teachableModel.predict(imageFeatures.expandDims()).squeeze();
    let highestIndex = prediction.argMax().arraySync();
    let predictionArray = prediction.arraySync();

    statusP.html(
      `Prediction: ${LABELS[highestIndex]} ` +
        `(${Math.floor(predictionArray[highestIndex] * 100)}% confidence)`
    );
  });

  // Keep predicting
  setTimeout(classifyImage, 10);
}
