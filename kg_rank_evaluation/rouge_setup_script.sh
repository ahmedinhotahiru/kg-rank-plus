#!/bin/bash

# Script to set up ROUGE-1.5.5 without requiring sudo privileges
# This script installs Perl modules and ROUGE-1.5.5 in the user's local directory

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
$HOME/local/bin/cpanm -l $HOME/local XML::DOM DB_File Encode HTML::Parser Term::ANSIColor

# Download and extract ROUGE-1.5.5
echo "Downloading ROUGE-1.5.5..."
cd $HOME/local/rouge
wget -c https://github.com/andersjo/pyrouge/raw/master/tools/ROUGE-1.5.5.tgz
tar -xzf ROUGE-1.5.5.tgz
cd ROUGE-1.5.5

# Apply Unicode patch
echo "Applying Unicode patch..."
wget -c https://raw.githubusercontent.com/AIPHES/emnlp19-moverscore/master/metrics/rouge/rouge_unicode.patch
patch -p1 < rouge_unicode.patch

# Make the script executable
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
