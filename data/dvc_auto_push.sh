#!/bin/bash

# Define file paths
FILE_PATH="data/processed/processed_data.csv"
REPO_PATH="/home/rohith-ramanan/Desktop/DA5402/Patient_Readmission_Prediction"

# Change to the repository directory
cd $REPO_PATH

# Check if the file exists
if [ -f "$FILE_PATH" ]; then
    # Get current date for version tag
    VERSION_TAG="v$(date '+%Y%m%d_%H%M%S')"
    
    # Add file to DVC
    dvc add "$FILE_PATH"
    
    # Add DVC tracking file to Git
    git add "$FILE_PATH.dvc"
    
    # Commit the changes
    git commit -m "Update processed_data.csv ($VERSION_TAG)"
    
    # Tag this version
    git tag -a "$VERSION_TAG" -m "Data version $VERSION_TAG"
    
    # Push DVC file to remote
    dvc push
    
    # Push Git changes and tags
    git push
    git push --tags

    # Delete the local file
    rm "$FILE_PATH"
    
    echo "Successfully processed and pushed new version: $VERSION_TAG"
else
    echo "No file found at $FILE_PATH, nothing to do."
fi