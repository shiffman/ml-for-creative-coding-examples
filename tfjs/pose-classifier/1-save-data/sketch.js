// Demo training a hand pose classifier w/o ml5!
// https://github.com/shiffman/ML-for-Creative-Coding

let video;
// Hand pose model
let handPose;
// Detected hands
let hands = [];

// Array to save all the data in!
let collectedData = [];

function setup() {
  createCanvas(640, 480);
  // Set up video
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();
  // load the handpose model
  loadHandPose();
}

// Load the handpose model from TensorFlow.js instead of ml5.js
// Requires an async function and manual detection
async function loadHandPose() {
  console.log('loading model');
  handPose = await handpose.load();
  console.log('model loaded');

  // Begin recursive detection loop
  await detectHands();
}

// No built-in looping like ml5.js
async function detectHands() {
  // Detect the hands
  hands = await handPose.estimateHands(video.elt);
  // Wait a tiny bit and go again!
  setTimeout(detectHands, 10);
}

function draw() {
  // Draw all the keypoints
  image(video, 0, 0, width, height);
  if (hands.length > 0) {
    let hand = hands[0];
    const keypoints = hand.landmarks;
    for (let j = 0; j < keypoints.length; j++) {
      fill(0, 255, 0);
      noStroke();
      circle(keypoints[j][0], keypoints[j][1], 10);
    }
  }
}

// Flatten the data into a plain array
// Manually normalize the data
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

// Collect labeled training data manually with key presses
function keyPressed() {
  if (key === 'k') {
    // Save collected hand pose data as JSON
    saveJSON({ data: collectedData }, 'rock-paper-data.json');
  }

  if (hands.length > 0) {
    const hand = hands[0];
    // Flatten the data and add to the collectedData array
    const flattened = flattenHandData(hand);
    if (key === 'r') {
      collectedData.push({ inputs: flattened, label: 'rock' });
      console.log('Collected rock data');
    } else if (key === 'p') {
      collectedData.push({ inputs: flattened, label: 'paper' });
      console.log('Collected paper data');
    }
  }
}
