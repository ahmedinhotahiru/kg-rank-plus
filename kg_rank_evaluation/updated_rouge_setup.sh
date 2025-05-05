#!/bin/bash

# Updated script to set up ROUGE-1.5.5 without requiring sudo privileges
# This script installs Perl modules and ROUGE-1.5.5 in the user's local directory

echo "Setting up ROUGE-1.5.5 (Updated Version)..."

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
$HOME/local/bin/cpanm -l $HOME/local XML::DOM

# Download ROUGE-1.5.5 from a working mirror
echo "Downloading ROUGE-1.5.5..."
cd $HOME/local/rouge
wget -c https://github.com/summanlp/evaluation/raw/master/ROUGE-1.5.5.tgz

if [ ! -f ROUGE-1.5.5.tgz ]; then
    echo "Primary download failed, trying alternative source..."
    wget -c https://github.com/neural-dialogue-metrics/rouge/raw/master/ROUGE-1.5.5.tgz
fi

if [ ! -f ROUGE-1.5.5.tgz ]; then
    echo "Error: Could not download ROUGE-1.5.5.tgz from any source."
    echo "Manually download ROUGE-1.5.5.tgz and place it in $HOME/local/rouge/ then run this script again."
    exit 1
fi

# Extract ROUGE-1.5.5
echo "Extracting ROUGE-1.5.5..."
tar -xzf ROUGE-1.5.5.tgz

if [ ! -d ROUGE-1.5.5 ]; then
    echo "Error: Could not extract ROUGE-1.5.5."
    exit 1
fi

cd ROUGE-1.5.5

# Create Unicode patch if not available for download
echo "Creating Unicode patch..."
cat > rouge_unicode.patch <<'EOL'
diff -ur ROUGE-1.5.5.orig/XML/DOM/AttDef.pm ROUGE-1.5.5/XML/DOM/AttDef.pm
--- ROUGE-1.5.5.orig/XML/DOM/AttDef.pm	1999-07-14 11:32:52.000000000 -0400
+++ ROUGE-1.5.5/XML/DOM/AttDef.pm	2013-04-08 22:12:54.000000000 -0400
@@ -27,7 +27,7 @@
 
 sub new
 {
-    my ($type,$name,$strType,$default,$fixed) = @_;
+    my ($type,$name,$strType,$default,$fixed) = map {utf8::encode($_) if defined($_); $_} @_;
     
     my $self = bless {}, $type;
 
diff -ur ROUGE-1.5.5.orig/XML/DOM/AttlistDecl.pm ROUGE-1.5.5/XML/DOM/AttlistDecl.pm
--- ROUGE-1.5.5.orig/XML/DOM/AttlistDecl.pm	1999-07-14 11:32:52.000000000 -0400
+++ ROUGE-1.5.5/XML/DOM/AttlistDecl.pm	2013-04-08 22:13:14.000000000 -0400
@@ -35,7 +35,7 @@
 
 sub new
 {
-    my ($type, $name) = @_;
+    my ($type, $name) = map {utf8::encode($_) if defined($_); $_} @_;
     my $self = bless {}, $type;
 
     $self->{Name} = $name;
diff -ur ROUGE-1.5.5.orig/XML/DOM/DocumentType.pm ROUGE-1.5.5/XML/DOM/DocumentType.pm
--- ROUGE-1.5.5.orig/XML/DOM/DocumentType.pm	1999-07-14 11:32:52.000000000 -0400
+++ ROUGE-1.5.5/XML/DOM/DocumentType.pm	2013-04-08 22:13:37.000000000 -0400
@@ -34,7 +34,7 @@
 
 sub new
 {
-    my ($type, $name, $entities, $notations) = @_;
+    my ($type, $name, $entities, $notations) = map {utf8::encode($_) if defined($_); $_} @_;
 
     my $self = bless {}, $type;
 
diff -ur ROUGE-1.5.5.orig/XML/DOM/Element.pm ROUGE-1.5.5/XML/DOM/Element.pm
--- ROUGE-1.5.5.orig/XML/DOM/Element.pm	1999-07-14 11:32:52.000000000 -0400
+++ ROUGE-1.5.5/XML/DOM/Element.pm	2013-04-08 22:13:59.000000000 -0400
@@ -47,7 +47,7 @@
 
 sub new
 {
-    my ($type, $name, $doc) = @_;
+    my ($type, $name, $doc) = map {utf8::encode($_) if defined($_); $_} @_;
 
     my $self = bless XML::DOM::Node->new (ELEMENT_NODE), $type;
 
diff -ur ROUGE-1.5.5.orig/XML/DOM/ElementDecl.pm ROUGE-1.5.5/XML/DOM/ElementDecl.pm
--- ROUGE-1.5.5.orig/XML/DOM/ElementDecl.pm	1999-07-14 11:32:52.000000000 -0400
+++ ROUGE-1.5.5/XML/DOM/ElementDecl.pm	2013-04-08 22:14:21.000000000 -0400
@@ -27,7 +27,7 @@
 
 sub new
 {
-    my ($type, $name, $model) = @_;
+    my ($type, $name, $model) = map {utf8::encode($_) if defined($_); $_} @_;
     my $self = bless {}, $type;
 
     $self->{Name} = $name;
diff -ur ROUGE-1.5.5.orig/XML/DOM/Entity.pm ROUGE-1.5.5/XML/DOM/Entity.pm
--- ROUGE-1.5.5.orig/XML/DOM/Entity.pm	1999-07-14 11:32:52.000000000 -0400
+++ ROUGE-1.5.5/XML/DOM/Entity.pm	2013-04-08 22:14:49.000000000 -0400
@@ -31,7 +31,7 @@
 
 sub new
 {
-    my ($type, $notation, $public, $system, $value, $name) = @_;
+    my ($type, $notation, $public, $system, $value, $name) = map {utf8::encode($_) if defined($_); $_} @_;
 
     my $self = bless XML::DOM::Node::new (ENTITY_NODE), $type;
 
diff -ur ROUGE-1.5.5.orig/XML/DOM/EntityReference.pm ROUGE-1.5.5/XML/DOM/EntityReference.pm
--- ROUGE-1.5.5.orig/XML/DOM/EntityReference.pm	1999-07-14 11:32:52.000000000 -0400
+++ ROUGE-1.5.5/XML/DOM/EntityReference.pm	2013-04-08 22:15:11.000000000 -0400
@@ -28,7 +28,7 @@
 
 sub new
 {
-    my ($type, $name, $doc) = @_;
+    my ($type, $name, $doc) = map {utf8::encode($_) if defined($_); $_} @_;
 
     my $self = bless XML::DOM::Node::new (ENTITY_REFERENCE_NODE), $type;
 
diff -ur ROUGE-1.5.5.orig/XML/DOM/Node.pm ROUGE-1.5.5/XML/DOM/Node.pm
--- ROUGE-1.5.5.orig/XML/DOM/Node.pm	1999-07-14 11:32:52.000000000 -0400
+++ ROUGE-1.5.5/XML/DOM/Node.pm	2013-04-08 22:15:42.000000000 -0400
@@ -58,7 +58,7 @@
 
 sub appendChild
 {
-    my ($self, $newChild) = @_;
+    my ($self, $newChild) = map {utf8::encode($_) if defined($_) && ! ref($_); $_} @_;
 
     # deepCopy only if node is from another Document
     my $doc1 = $self->getOwnerDocument;
@@ -91,7 +91,7 @@
 
 sub cloneNode
 {
-    my ($self, $deep) = @_;
+    my ($self, $deep) = map {utf8::encode($_) if defined($_) && ! ref($_); $_} @_;
     my $new;
 
     # Try to reuse the class package
@@ -158,7 +158,7 @@
 
 sub insertBefore
 {
-    my ($self, $newChild, $refChild) = @_;
+    my ($self, $newChild, $refChild) = map {utf8::encode($_) if defined($_) && ! ref($_); $_} @_;
 
     my $doc1 = $self->getOwnerDocument;
     my $doc2 = $newChild->getOwnerDocument;
@@ -267,7 +267,7 @@
 
 sub normalize
 {
-    my $self = shift;
+    my $self = map {utf8::encode($_) if defined($_) && ! ref($_); $_} shift;
 
     for (my $child = $self->getFirstChild; 
 	 defined $child; 
@@ -297,7 +297,7 @@
 
 sub removeChild
 {
-    my ($self, $oldChild) = @_;
+    my ($self, $oldChild) = map {utf8::encode($_) if defined($_) && ! ref($_); $_} @_;
     
     my $parent = $oldChild->getParentNode;
     croak new XML::DOM::DOMException (NOT_FOUND_ERR,
@@ -322,7 +322,7 @@
 
 sub replaceChild
 {
-    my ($self, $newChild, $oldChild) = @_;
+    my ($self, $newChild, $oldChild) = map {utf8::encode($_) if defined($_) && ! ref($_); $_} @_;
     return $oldChild unless defined $newChild;
     
     my $doc1 = $self->getOwnerDocument;
@@ -414,7 +414,7 @@
 
 sub setNodeValue
 {
-    my ($self, $value) = @_;
+    my ($self, $value) = map {utf8::encode($_) if defined($_) && ! ref($_); $_} @_;
 
     # Default: no action
     # (Element, Entity, EntityReference, Notation overrides)
@@ -463,7 +463,7 @@
 
 sub _checkAppendOk
 {
-    my ($self, $node) = @_;
+    my ($self, $node) = map {utf8::encode($_) if defined($_) && ! ref($_); $_} @_;
 
     my $invalid;
     my $type = $node->getNodeType;
diff -ur ROUGE-1.5.5.orig/XML/DOM/Notation.pm ROUGE-1.5.5/XML/DOM/Notation.pm
--- ROUGE-1.5.5.orig/XML/DOM/Notation.pm	1999-07-14 11:32:52.000000000 -0400
+++ ROUGE-1.5.5/XML/DOM/Notation.pm	2013-04-08 22:16:09.000000000 -0400
@@ -30,7 +30,7 @@
 
 sub new
 {
-    my ($type, $name, $base, $public, $system) = @_;
+    my ($type, $name, $base, $public, $system) = map {utf8::encode($_) if defined($_); $_} @_;
 
     my $self = bless XML::DOM::Node::new (NOTATION_NODE), $type;
 
diff -ur ROUGE-1.5.5.orig/XML/DOM/ProcessingInstruction.pm ROUGE-1.5.5/XML/DOM/ProcessingInstruction.pm
--- ROUGE-1.5.5.orig/XML/DOM/ProcessingInstruction.pm	1999-07-14 11:32:52.000000000 -0400
+++ ROUGE-1.5.5/XML/DOM/ProcessingInstruction.pm	2013-04-08 22:16:50.000000000 -0400
@@ -27,7 +27,7 @@
 
 sub new
 {
-    my ($type, $target, $data) = @_;
+    my ($type, $target, $data) = map {utf8::encode($_) if defined($_); $_} @_;
 
     my $self = bless XML::DOM::Node::new (PROCESSING_INSTRUCTION_NODE), $type;
     
diff -ur ROUGE-1.5.5.orig/XML/DOM/Text.pm ROUGE-1.5.5/XML/DOM/Text.pm
--- ROUGE-1.5.5.orig/XML/DOM/Text.pm	1999-07-14 11:32:52.000000000 -0400
+++ ROUGE-1.5.5/XML/DOM/Text.pm	2013-04-08 22:17:20.000000000 -0400
@@ -39,7 +39,7 @@
 
 sub splitText
 {
-    my ($self, $offset) = @_;
+    my ($self, $offset) = map {utf8::encode($_) if defined($_) && ! ref($_); $_} @_;
     my $value = $self->getData;
 
     # REC: raises INDEX_SIZE_ERR if offset is negative or greater that
EOL

# Apply the Unicode patch
echo "Applying Unicode patch..."
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
