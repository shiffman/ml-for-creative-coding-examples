let classifier;

function setup() {
  createCanvas(100, 100);
  ml5.setBackend('webgl');
  background(0);
  let options = {
    // inputs: 42,
    // outputs: 3,
    task: 'classification',
    debug: true,
  };
  classifier = ml5.neuralNetwork(options);
  classifier.loadData('rock-paper-data-1.json', dataLoaded);
}

function dataLoaded() {
  console.log('Data loaded');
  classifier.normalizeData();
  classifier.train({ epochs: 100, learningRate: 0.001 }, finishedTraining);
}

function finishedTraining() {
  console.log('Finished training');
  classifier.save();
}
