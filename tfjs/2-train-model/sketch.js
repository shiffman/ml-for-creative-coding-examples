// Training a hand pose classifier with TensorFlow.js
// https://github.com/shiffman/ML-for-Creative-Coding

let rawData;

function preload() {
  // Load the data
  rawData = loadJSON('rock-paper-data.json');
}

function setup() {
  noCanvas();

  const { data } = rawData;

  // The labels are stored as string, this maps them to indices
  let labelNumbers = {};
  let labelCount = 0;
  for (let item of data) {
    if (labelNumbers[item.label] == undefined) {
      labelNumbers[item.label] = labelCount;
      labelCount++;
    }
  }

  // Prepare inputs along with numeric indices of labels
  let inputs = [];
  let labels = [];
  for (let item of data) {
    inputs.push(item.inputs);
    labels.push(labelNumbers[item.label]);
  }

  // Convert inputs to a tensor!!
  // The shape will be [N, 42] (N is the number of samples)
  let xs = tf.tensor2d(inputs);

  // Convert labels to one-hot encoding for categorical classification

  // First a 1D tensor
  let labelsTensor = tf.tensor1d(labels, 'int32');
  // Then convert to one-hot encoding, the shape will be [N, labelCount]
  let ys = tf.oneHot(labelsTensor, labelCount);

  // We have to manage memory with tensorflow.js!!
  labelsTensor.dispose();

  // Define a simple feedforward neural network
  let model = tf.sequential();

  // hidden layer (inputs are technically not their own layer)
  model.add(
    tf.layers.dense({
      units: 16,
      activation: 'relu',
      // 42 input features from hand landmarks
      inputShape: [42],
    })
  );

  // Maybe add another hidden layer?
  // model.add(
  //   tf.layers.dense({
  //     units: 16,
  //     activation: 'relu',
  //   })
  // );

  // Output layer with softmax for classification
  model.add(
    tf.layers.dense({
      units: labelCount,
      activation: 'softmax',
    })
  );

  // Compile the model with Adam optimizer and categorical cross-entropy loss
  // So many new terms!!
  let learningRate = 0.01;

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const epochs = 200;

  // Train the model
  model.fit(xs, ys, {
    epochs: epochs,
    // Shuffle the data around randomly, this is good!
    shuffle: true,
    callbacks: {
      onTrainBegin: () => console.log('Training start'),
      onEpochEnd: (epoch, logs) => console.log('Epoch', epoch, logs),
      onTrainEnd: () => {
        console.log('Finished training');
        // Save the trained model as downloadable files
        model.save('downloads://rock-paper-model');
      },
    },
  });
}
