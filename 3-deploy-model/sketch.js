// In-class demo training a hand pose classifier
// https://github.com/shiffman/ML-for-Creative-Coding

// Initialize variables for ML hand pose detection
let handPose;
let video;
let hands = [];

// Neural network classifier
let classifier;
let classifying = false;

// Store the latest classification result
let label = 'Waiting...';

function preload() {
  // Load the handpose model from ml5.js
  handPose = ml5.handPose();
}

function setup() {
  createCanvas(640, 480);

  // Start detecting hands in the video
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();
  handPose.detectStart(video, gotHands);

  // Configure the neural network
  // The network will perform classification (as opposed to regression)
  // Debug mode will show the training visualization
  let options = {
    task: 'classification',
    debug: true,
  };
  classifier = ml5.neuralNetwork(options);

  // Load the trained model
  const modelDetails = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin',
  };
  classifier.load(modelDetails, modelLoaded);
}

function modelLoaded() {
  console.log('Model loaded');
}

function gotHands(results) {
  hands = results;

  // Only classify if not already classifying and at least one hand detected
  if (!classifying && hands.length > 0) {
    // Prevent overlapping classifications
    classifying = true;
    classifyData();
  }
}

function classifyData() {
  // Convert handpose data into a format suitable for the neural network
  let hand = hands[0];
  let inputData = flattenData(hand);

  // Classify the data
  classifier.classify(inputData, gotResults);
}

function gotResults(results) {
  // Store and display the classification result
  label = results[0].label;
  console.log(label);

  // Reset flag so the next fresh detection can be classified
  classifying = false;
}

// Flatten handpose data into a 1D array
function flattenData(hand) {
  let inputData = [];
  let keypoints = hand.keypoints;
  for (let keypoint of keypoints) {
    inputData.push(keypoint.x);
    inputData.push(keypoint.y);
  }
  return inputData;
}

function draw() {
  // Display video feed
  image(video, 0, 0, width, height);

  // Draw keypoints
  for (let i = 0; i < hands.length; i++) {
    let hand = hands[i];
    for (let j = 0; j < hand.keypoints.length; j++) {
      let keypoint = hand.keypoints[j];
      fill(0, 255, 0);
      noStroke();
      circle(keypoint.x, keypoint.y, 10);
    }
  }

  // Black rectangle for classification label
  fill(0);
  noStroke();
  rect(0, height - 60, width, 60);

  // Display classification label
  fill(255);
  textSize(32);
  textAlign(CENTER, CENTER);
  text(label, width / 2, height - 30);
}
