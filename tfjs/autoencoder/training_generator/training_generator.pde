// Generate Training Data for Autoencoder
// https://github.com/shiffman/ml-for-creative-coding-examples
// https://github.com/shiffman/ML-for-Creative-Coding

// Counter for images generated
int counter = 0;   
 // Target image size (28x28 pixels)
int W = 28;       

void setup() {
  size(280, 280); 
}

void draw() {
  background(255);

  // Draw a randomly sized square at the center
  float r = random(0, width);
  strokeWeight(16);
  rectMode(CENTER);
  square(width / 2, width / 2, r);

  // Resize to 28x28 pixels
  PImage img = get();
  img.resize(W, W);

  // Save the image with sequential numbering
  img.save("data/square" + nf(counter, 4) + ".png");

  // Increment counter
  counter++;

  // Exit after generating 1000 images
  if (counter == 1000) {
    println("Finished generating images.");
    exit();
  }
}
