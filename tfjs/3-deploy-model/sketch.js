// Demo deploying a trained hand pose classifier with TensorFlow.js
// https://github.com/shiffman/ML-for-Creative-Coding

let video;

// Hand pose model
let handPose;
// Detected hands
let hands = [];

// This is our model we trained!
let classifier;

let label = 'waiting...';

// No one is keeping track of the labels, so I'm just hardcoding them
// What would be a better solution?
let labels = ['paper', 'rock'];

function setup() {
  createCanvas(640, 480);
  // Set up the video
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  // load Handpose and the custom classifier
  loadHandPose();
  loadClassifier();
}

// Load the trained TensorFlow.js classifier model
async function loadClassifier() {
  console.log('loading custom classifier model');
  classifier = await tf.loadLayersModel('model/rock-paper-model.json');
  console.log('custom classifier model loaded');
}

// Load the TensorFlow.js handpose model for hand keypoint detection
async function loadHandPose() {
  console.log('loading handpose model');
  handPose = await handpose.load();
  detectHands();
  console.log('handpose model loaded');
}

// Continuously detect hands and classify gestures
async function detectHands() {
  hands = await handPose.estimateHands(video.elt);

  // Before looking at the next hand, let's classify this one!
  if (hands.length > 0) {
    let hand = hands[0];
    let inputs = flattenHandData(hand);
    await classifyHandPose(inputs);
  }
  // Begin recursive detection loop
  setTimeout(detectHands, 10);
}

function draw() {
  // Draw video
  image(video, 0, 0, width, height);

  // Draw hand keypoints
  if (hands.length > 0) {
    let hand = hands[0];
    const keypoints = hand.landmarks;
    for (let j = 0; j < keypoints.length; j++) {
      fill(0, 255, 0);
      noStroke();
      circle(keypoints[j][0], keypoints[j][1], 10);
    }
  }

  // Display classification label
  fill(0);
  noStroke();
  rect(0, height - 60, width, 60);
  fill(255);
  textSize(32);
  textAlign(CENTER, CENTER);
  text(label, width / 2, height - 30);
}

// Flatten the data into a plain array
// Manually normalize the data, must do this the same was as the data collection sketch!
function flattenHandData(hand) {
  let inputs = [];
  const keypoints = hand.landmarks;
  for (let p of keypoints) {
    // Normalize x coordinate
    inputs.push(p[0] / width);
    // Normalize y coordinate
    inputs.push(p[1] / height);
  }
  return inputs;
}

// Function to classify the detected hand pose using the trained model
async function classifyHandPose(inputArray) {
  // Convert the input array into a 2D tensor of shape [1, 42]
  const inputTensor = tf.tensor2d([inputArray]);
  // Make a prediction through the model
  const prediction = classifier.predict(inputTensor);
  // Convert to something we can look at, these will be the raw probabilities
  const outputs = await prediction.data();

  // Find the highest confidence prediction
  let labelIndex = 0;
  let bestConfidence = 0;
  for (let i = 0; i < outputs.length; i++) {
    if (outputs[i] > bestConfidence) {
      bestConfidence = outputs[i];
      labelIndex = i;
    }
  }

  console.log(labelIndex + ' : ' + bestConfidence);

  // Convert the label index to a label string
  label = labels[labelIndex];

  // Don't forget to manage the memory!!!
  inputTensor.dispose();
  prediction.dispose();
}
