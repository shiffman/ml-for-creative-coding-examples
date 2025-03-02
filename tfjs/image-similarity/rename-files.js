// Node.js Helper Script: rename-files.js
// I used this with: https://github.com/EliSchwartz/imagenet-sample-images

// Renames all `.jpeg` files in the './images' folder to numbered filenames (0.jpg, 1.jpg, etc.)
// It also outputs a mapping file (filenames.json) to track original filenames.

// Instructions:
// 1. Place the images in an './images' directory.
// 2. Run `node rename-files.js` from the terminal.

const fs = require('fs');
const path = require('path');

const directory = './images';
const outputFile = 'filenames.json';

// Get all JPEG filenames
const files = fs
  .readdirSync(directory)
  .filter((file) => file.toLowerCase().endsWith('.jpeg'))
  .sort();

let mapping = {};

// Rename each file
files.forEach((file, index) => {
  const newName = `${index}.jpg`;
  fs.renameSync(path.join(directory, file), path.join(directory, newName));
  mapping[newName] = file;
});

// Save original filenames to JSON file
fs.writeFileSync(outputFile, JSON.stringify(mapping, null, 2));
