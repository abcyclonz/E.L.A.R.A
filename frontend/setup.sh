#!/bin/bash
set -e
echo "Installing dependencies..."
npm install --legacy-peer-deps
echo "Initializing shadcn components..."
npx shadcn@latest add accordion alert-dialog alert avatar badge button card checkbox collapsible input label select separator textarea toast sonner tooltip
echo "Done! Run: npm run dev"
