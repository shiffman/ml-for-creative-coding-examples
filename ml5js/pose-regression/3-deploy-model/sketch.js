// In-class demo training a hand pose classifier
// https://github.com/shiffman/ML-for-Creative-Coding

// Initialize variables for ML hand pose detection
let handPose;
let video;
let hands = [];

// Neural network regression model
let regressionModel;
let predicting = false;
let predictedValue = 0;

// Store the latest classification result
let label = 'Waiting...';

function preload() {
  // Load the handpose model from ml5.js
  handPose = ml5.handPose({ flipped: true });
}

function setup() {
  createCanvas(640, 480);

  // Start detecting hands in the video
  video = createCapture(VIDEO, { flipped: true });
  video.size(640, 480);
  video.hide();
  handPose.detectStart(video, gotHands);

  // Configure the neural network
  // The network will perform regression
  let options = {
    task: 'regression',
    debug: true,
  };
  regressionModel = ml5.neuralNetwork(options);

  // Load the trained model
  const modelDetails = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin',
  };
  regressionModel.load(modelDetails, modelLoaded);
}

function modelLoaded() {
  console.log('Model loaded');
}

function gotHands(results) {
  hands = results;

  // Only predict if not already predicting and at least one hand detected
  if (!predicting && hands.length > 0) {
    // Prevent overlapping classifications
    predicting = true;
    predictData();
  }
}

function predictData() {
  // Convert handpose data into a format suitable for the neural network
  let hand = hands[0];
  let inputData = flattenData(hand);

  // Classify the data
  regressionModel.predict(inputData, gotResults);
}

function gotResults(results) {
  // Store the regression result, it's just one number!
  predictedValue = results[0].value;
  console.log(predictedValue);

  // Reset flag so the next fresh detection can be predicted
  predicting = false;
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
  // Draw video
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

  // Black rectangle for showing predicted value
  fill(0);
  noStroke();
  rect(0, height - 60, width, 60);

  // Show predicted value
  fill(255);
  textSize(32);
  textAlign(CENTER, CENTER);
  text(nf(predictedValue, 0, 2), width / 2, height - 30);
}
