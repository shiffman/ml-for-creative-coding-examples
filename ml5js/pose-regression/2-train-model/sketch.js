// In class demo training a hand pose classifier
// https://github.com/shiffman/ML-for-Creative-Coding

// Neural network model
let regressionModel;

function setup() {
  // No interface or visual elements
  createCanvas(100, 100);

  // ml5 neural network not compatible with webgpu
  // could also try 'cpu' for small datasets
  ml5.setBackend('webgl');

  // Configure the neural network
  // The network will perform classification (as opposed to regression)
  // Debug mode will show the training visualization
  let options = {
    // inputs: 42,
    // outputs: 1,
    task: 'regression',
    debug: true,
  };
  regressionModel = ml5.neuralNetwork(options);

  // Load the collected data from the file
  regressionModel.loadData('sample-regression-data.json', dataLoaded);
}

function dataLoaded() {
  console.log('Data loaded');
  // Normalize the collected data
  regressionModel.normalizeData();
  // Train the neural network
  // Epochs are the number of times the network will see the data
  // Learning rate is the step size for the optimizer (weight adjustments according to error)
  regressionModel.train(
    { epochs: 100, validationSplit: 0.0, learningRate: 0.001 },
    finishedTraining
  );
}

function finishedTraining() {
  console.log('Finished training');
  // Save the trained model
  regressionModel.save();
}
