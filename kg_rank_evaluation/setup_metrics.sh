#!/bin/bash

# Create directory for metric-related files
mkdir -p metrics
cd metrics

# ========= MoverScore Setup =========
echo "Setting up MoverScore..."

# Clone MoverScore repository
git clone https://github.com/AIPHES/emnlp19-moverscore.git
cd emnlp19-moverscore

# Install MoverScore dependencies
pip install -r requirements.txt
pip install .

# Install Word2Vec embeddings if not already available
if [ ! -d "word2vec" ]; then
    mkdir -p word2vec
    cd word2vec
    wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    gunzip GoogleNews-vectors-negative300.bin.gz
    cd ..
fi

cd ..

# ========= BLEURT Setup =========
echo "Setting up BLEURT..."

# Clone BLEURT repository
git clone https://github.com/google-research/bleurt.git
cd bleurt

# Install BLEURT
pip install .

# Download BLEURT checkpoints
python -c "import bleurt; bleurt.checkpoint.get_bleurt_checkpoint('BLEURT-20')"

cd ..

echo "Installation complete!"
echo "Make sure to set the PYTHONPATH environment variable to include the metric directories:"
echo "export PYTHONPATH=$PYTHONPATH:$(pwd)/emnlp19-moverscore:$(pwd)/bleurt"
