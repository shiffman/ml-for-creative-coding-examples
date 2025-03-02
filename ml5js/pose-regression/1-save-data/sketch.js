// In class demo training a hand pose regression model
// https://github.com/shiffman/ML-for-Creative-Coding

// Initialize variables for ML hand pose detection
let handPose;
let video;
let hands = [];

// Neural network model
let regressionModel;

// Interface for training data collection
let trainingSlider;
let trainingValueP;

function preload() {
  // Load the handpose model from ml5.js
  handPose = ml5.handPose({ flipped: true });
}

function setup() {
  createCanvas(640, 480);

  // Start video
  video = createCapture(VIDEO, { flipped: true });
  video.size(640, 480);
  video.hide();

  // Start handpose
  handPose.detectStart(video, gotHands);

  // Slider for target output
  trainingSlider = createSlider(0, 1, 0.5, 0.01);
  trainingValueP = createP(`Training Value: ${trainingSlider.value()}`);
  trainingSlider.input(updateValue);

  // Button to collect data
  let collectButton = createButton('Collect Data');
  collectButton.mousePressed(collectData);

  // Button to save data
  let saveButton = createButton('Save Data');
  saveButton.mousePressed(saveData);

  // Configure the neural network
  // The network will perform classification (as opposed to regression)
  // Debug mode will show the training visualization
  let options = {
    task: 'regression',
    debug: true,
  };
  // Create a new neural network with the specified options
  regressionModel = ml5.neuralNetwork(options);
}

// Collect data
function collectData() {
  if (hands.length > 0) {
    // Get the hand pose data and flatten it for the neural network
    let hand = hands[0];
    let inputData = flattenData(hand);
    // Add training data using the slider
    let target = trainingSlider.value();
    // Just one output but still has to be in an array
    regressionModel.addData(inputData, [target]);
    console.log(`logging data for ${target}`);
  }
}

// Save Data
function saveData() {
  // Normalize the collected data to improve training performance
  regressionModel.normalizeData();
  // Save the collected training data to a file
  regressionModel.saveData();
}

// Show value in p DOM element
function updateValue() {
  trainingValueP.html(`Training Value: ${trainingSlider.value()}`);
}

// Convert the handpose data into a format suitable for the neural network
// Each hand keypoint has x,y coordinates, resulting in 42 total inputs (21 keypoints * 2 coordinates)
function flattenData(hand) {
  let inputData = [];
  let keypoints = hand.keypoints;
  for (let keypoint of keypoints) {
    // Add raw x,y coordinates to the input array
    // Note: Data normalization will be handled by the neural network
    // But could be done here dividing by the width and height of the video
    inputData.push(keypoint.x);
    inputData.push(keypoint.y);
  }
  return inputData;
}

function draw() {
  image(video, 0, 0, width, height);

  // Draw the hand keypoints
  for (let i = 0; i < hands.length; i++) {
    let hand = hands[i];
    for (let j = 0; j < hand.keypoints.length; j++) {
      let keypoint = hand.keypoints[j];
      fill(0, 255, 0);
      noStroke();
      circle(keypoint.x, keypoint.y, 10);
    }
  }
}

// Callback function that receives hand detection results
function gotHands(results) {
  hands = results;
}
