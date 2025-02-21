let handPose;
let video;
let hands = [];

let classifier;

function preload() {
  handPose = ml5.handPose();
}

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();
  handPose.detectStart(video, gotHands);

  let options = {
    // inputs: 42,
    // outputs: 3,
    task: 'classification',
    debug: true,
  };
  classifier = ml5.neuralNetwork(options);
}

class Particle {
  constructor() {
    this.x = random(width);
    this.y = random(height);
    this.label = '';
  }

  addData(s) {
    this.label = s;
  }
}

// flatten handpose data into a 1D array

function flattenData(hand) {
  let inputData = [];
  let keypoints = hand.keypoints;
  for (let keypoint of keypoints) {
    // console.log(keypoint.x, keypoint.y);

    // Normalize the data myself!
    // inputData.push(keypoint.x / width); // could use map() here
    // inputData.push(keypoint.y / height);
    inputData.push(keypoint.x);
    inputData.push(keypoint.y);
  }
  return inputData;
}

function keyPressed() {
  if (key == 'k') {
    classifier.normalizeData();
    console.log(classifier);
    classifier.saveData();
  }

  if (hands.length > 0) {
    let hand = hands[0];
    let inputData = flattenData(hand);
    if (key == 'p') {
      // let flat = hands[0].keypoints.map((k) => [k.x, k.y]).flat();
      classifier.addData(inputData, ['paper']);
      console.log('logging paper data');
    } else if (key == 'r') {
      classifier.addData(inputData, ['rock']);
      console.log('logging rock data');
    }
  }
}

function draw() {
  image(video, 0, 0, width, height);

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

function gotHands(results) {
  hands = results;
}
