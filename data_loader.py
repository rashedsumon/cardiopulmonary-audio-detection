import os
import kagglehub

def load_dataset():
    """
    Automatically downloads the latest version of the cardiopulmonary sound signals
    dataset using kagglehub and returns the local directory path.
    """
    print("Checking / Downloading dataset via kagglehub...")
    try:
        # Download latest version of the target cardiopulmonary dataset
        path = kagglehub.dataset_download("maulikgajera/cardiopulmonary-sound-signals-for-deep-learning")
        
        print("\n" + "="*50)
        print("✅ SUCCESS: Dataset is ready for use!")
        print(f"📁 Local Path to dataset files: {path}")
        print("="*50 + "\n")
        
        return path
    except Exception as e:
        print("\n" + "!"*50)
        print(f"❌ ERROR: Failed to download dataset.\nDetails: {e}")
        print("💡 Tip: Ensure your Kaggle API credentials are set up if required.")
        print("!"*50 + "\n")
        return None

if __name__ == "__main__":
    # Executing the script directly will trigger the download and verify the path
    load_dataset()