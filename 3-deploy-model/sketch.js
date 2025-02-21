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
  const modelDetails = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin',
  };
  classifier.load(modelDetails, modelLoaded);
}

function modelLoaded() {
  console.log('model loaded');
  classifyData();
}

function classifyData() {
  if (hands.length > 0) {
    let hand = hands[0];
    let inputData = flattenData(hand);
    classifier.classify(inputData, gotResults);
  } else {
    setTimeout(classifyData, 100);
  }
}

function gotResults(results) {
  console.log(results[0].label);
  classifyData();
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
  //console.log(hands);
}
