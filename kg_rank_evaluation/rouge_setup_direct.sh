#!/bin/bash

# Script to set up ROUGE-1.5.5 using direct GitHub links
# This script installs ROUGE-1.5.5 in the user's local directory without sudo

echo "Setting up ROUGE-1.5.5 from provided GitHub repositories..."

# Create directories
echo "Creating directories..."
mkdir -p $HOME/local/rouge
mkdir -p $HOME/local/lib/perl5
mkdir -p $HOME/local/bin

# Set environment variables for local Perl modules
export PERL5LIB=$HOME/local/lib/perl5:$PERL5LIB
export PATH=$HOME/local/bin:$PATH

# Install cpanm (Perl module installer) locally
echo "Installing cpanm..."
curl -L https://cpanmin.us | perl - -l $HOME/local App::cpanminus

# Install required Perl modules locally
echo "Installing required Perl modules..."
$HOME/local/bin/cpanm -l $HOME/local XML::DOM DB_File

# Option 1: Clone the Unicode-compatible version directly
echo "Cloning ROUGE-1.5.5-unicode repository..."
cd $HOME/local/rouge
if git clone https://github.com/nisargjhaveri/ROUGE-1.5.5-unicode.git; then
    echo "Successfully cloned ROUGE-1.5.5-unicode repository."
    # Rename to match expected directory name
    mv ROUGE-1.5.5-unicode ROUGE-1.5.5
else
    echo "Failed to clone ROUGE-1.5.5-unicode repository."
    echo "Trying alternative method..."
    
    # Option 2: Try to download from pyrouge repository
    mkdir -p $HOME/local/rouge/ROUGE-1.5.5
    cd $HOME/local/rouge
    
    # Download through git sparse checkout (more reliable than direct download)
    git clone --no-checkout --depth 1 https://github.com/andersjo/pyrouge.git temp_pyrouge
    cd temp_pyrouge
    git sparse-checkout init --cone
    git sparse-checkout set tools/ROUGE-1.5.5
    git checkout
    
    if [ -d tools/ROUGE-1.5.5 ]; then
        echo "Successfully retrieved ROUGE-1.5.5 from pyrouge repository."
        cp -R tools/ROUGE-1.5.5/* $HOME/local/rouge/ROUGE-1.5.5/
        cd ..
        rm -rf temp_pyrouge
    else
        echo "Failed to retrieve ROUGE-1.5.5. Please download manually."
        exit 1
    fi
fi

# Change to ROUGE-1.5.5 directory
cd $HOME/local/rouge/ROUGE-1.5.5

# Make the scripts executable
echo "Setting execute permissions..."
chmod +x ROUGE-1.5.5.pl
chmod +x bin/*

# Create a settings file
echo "Creating settings file..."
echo 'export ROUGE_HOME="'$HOME'/local/rouge/ROUGE-1.5.5"' > $HOME/.rouge-settings.sh
echo 'export PERL5LIB="'$HOME'/local/lib/perl5:$PERL5LIB"' >> $HOME/.rouge-settings.sh
echo 'export PATH="'$HOME'/local/bin:$PATH"' >> $HOME/.rouge-settings.sh

# Source the settings
source $HOME/.rouge-settings.sh

# Verify installation
echo "Verifying installation..."
cd $ROUGE_HOME
./ROUGE-1.5.5.pl -h | head -n 5

echo ""
echo "ROUGE-1.5.5 has been installed to $ROUGE_HOME"
echo "Add the following to your .bashrc or .zshrc file to make the settings permanent:"
echo ""
echo 'source $HOME/.rouge-settings.sh'
echo ""
echo "Then run 'source ~/.bashrc' to apply the changes."
