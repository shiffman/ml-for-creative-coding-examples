const fs = require('fs');
const path = require('path');

const directory = './images';
const outputFile = 'filenames.json';

const files = fs
  .readdirSync(directory)
  .filter((file) => file.toLowerCase().endsWith('.jpeg'))
  .sort();

let mapping = {};
files.forEach((file, index) => {
  const newName = `${index}.jpg`;
  fs.renameSync(path.join(directory, file), path.join(directory, newName));
  mapping[newName] = file;
});

fs.writeFileSync(outputFile, JSON.stringify(mapping, null, 2));
