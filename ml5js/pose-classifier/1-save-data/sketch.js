// In class demo training a hand pose classifier
// https://github.com/shiffman/ML-for-Creative-Coding

// Initialize variables for ML hand pose detection
let handPose;
let video;
let hands = [];

// Neural network classifier
let classifier;

function preload() {
  // Load the handpose model from ml5.js
  handPose = ml5.handPose();
}

function setup() {
  createCanvas(640, 480);
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
  // Create a new neural network with the specified options
  classifier = ml5.neuralNetwork(options);
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

function keyPressed() {
  if (key == 'k') {
    // Normalize the collected data to improve training performance
    classifier.normalizeData();
    // Save the collected training data to a file
    classifier.saveData();
  }

  if (hands.length > 0) {
    // Get the hand pose data and flatten it for the neural network
    let hand = hands[0];
    let inputData = flattenData(hand);
    if (key == 'p') {
      // Add training data for "paper" gesture
      classifier.addData(inputData, ['paper']);
      console.log('logging paper data');
    } else if (key == 'r') {
      // Add training data for "rock" gesture
      classifier.addData(inputData, ['rock']);
      console.log('logging rock data');
    }
  }
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
