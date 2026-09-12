# Chunk 3: Training on MNIST

## What I built
- New Tensor operations needed for training:
  - __sub__ (subtraction, for calculating error)
  - __pow__ (exponentiation, for squaring the error)
  - mean (averages all values down to one number, for the final loss)
- Loaded real MNIST dataset (handwritten digit images 0-9) using
  sklearn's fetch_openml, using first 2000 images for speed
- Converted labels into one-hot format (e.g. digit 3 becomes
  [0,0,0,1,0,0,0,0,0,0]) so they can be compared against the
  network's 10 output numbers
- Built a small 2-layer neural network:
  784 inputs (28x28 pixel image, flattened)
  -> matmul(W1) + b1 -> relu -> matmul(W2) + b2 -> 10 outputs
- Wrote the training loop: forward pass -> calculate loss (Mean
  Squared Error) -> reset gradients -> backward() -> update weights
  (gradient descent) -> repeat for many epochs
- Added an accuracy() function using np.argmax to convert the
  network's 10 output scores into an actual predicted digit, then
  compared against real labels

## Bugs I hit and fixed

### Bug 1: indentation error (recurring issue)
- __sub__, __pow__, mean were not properly indented as methods
  inside the Tensor class in one version of the file
- Symptom: TypeError: unsupported operand type(s) for -: 'Tensor'
  and 'Tensor' -- Python didn't recognize __sub__ as belonging to
  the class
- Lesson: always double check new methods line up with existing
  ones (__init__, __add__, etc.) at the same indentation level

### Bug 2: missing pandas dependency
- fetch_openml needs pandas installed to load the dataset with
  as_frame=False and the newer 'auto' parser
- Fixed with: pip install pandas

### Bug 3: broadcasting mismatch in backward pass (the important one)
- Forward pass: x.matmul(W1) + b1 works fine because NumPy
  automatically "broadcasts" b1 (shape (64,)) up to match x's shape
  (2000, 64) during addition
- Backward pass broke: tried to write a (2000, 64) gradient directly
  into b1's (64,) grad slot -> ValueError: non-broadcastable output
  operand
- Fix: wrote an unbroadcast() helper function that sums a gradient
  back down along whichever dimensions were broadcasted, restoring
  it to the original input's shape before accumulating into .grad
- Applied this fix inside __add__'s _backward function
- This is a real, known subtlety that real frameworks like PyTorch
  also have to handle internally -- good proof of deep understanding

## Key concepts learned

### Why the network's first output is "wrong"
- Weights start as small random numbers (np.random.randn * 0.01)
- The math itself is correct from the start -- it's the weight
  VALUES that are meaningless until trained
- "Correct answers" come from the MNIST dataset itself (human
  labeled long before this project) -- not something the network
  invents or calculates independently

### Training loop = repeated small corrections
- Compare prediction to known correct answer (loss calculation)
- backward() automatically calculates how much each weight
  contributed to the error
- Nudge every weight slightly (learning_rate * grad) in the
  direction that reduces error
- Repeat many times (epochs) -- weights gradually shift from
  random to genuinely useful

### Learning rate matters a lot
- First attempt: learning_rate = 0.01, epochs = 20
  -> loss barely moved (0.1001 -> 0.0992), way too slow
- Fixed by increasing to learning_rate = 0.5, epochs = 100
  -> loss dropped steadily from 0.1001 to 0.0511
- Learning rate controls step size: too small = slow learning,
  too large = unstable/overshooting (would need to watch for
  loss increasing or becoming nan if pushed too high)

## Final results (confirmed working)
- Accuracy before training: 10.75% (matches expected ~10% baseline
  for random guessing across 10 digit classes)
- Final settings: learning_rate = 0.5, epochs = 100
- Loss: 0.1001 -> 0.0511 (steady, consistent decrease across all
  100 epochs)
- Accuracy after training: 75.40%

## Milestone
- Chunk 3 complete: proven, end-to-end validation that the
  hand-built Tensor class and autograd engine (Chunks 1-2) can
  actually train a real neural network on a real dataset and
  achieve genuinely useful accuracy -- not just theoretical code,
  a demonstrably working system. This is the strongest evidence
  point for the whole project so far.