#!/bin/bash
# Copy the built binary from addon.node to the darwin-arm64 package

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPM_DIR="$(dirname "$SCRIPT_DIR")"
WHISPER_ROOT="$(dirname "$NPM_DIR")"

SOURCE="$WHISPER_ROOT/examples/addon.node/build/Release/addon.node.node"
DEST="$NPM_DIR/packages/darwin-arm64/whisper.node"

if [ ! -f "$SOURCE" ]; then
    echo "Error: Binary not found at $SOURCE"
    echo "Please build the addon first:"
    echo "  cd $WHISPER_ROOT/examples/addon.node"
    echo "  npm install"
    echo "  npx cmake-js compile"
    exit 1
fi

cp "$SOURCE" "$DEST"
echo "Copied binary to $DEST"

# Show binary info
echo ""
echo "Binary info:"
ls -lh "$DEST"
echo ""
file "$DEST"
