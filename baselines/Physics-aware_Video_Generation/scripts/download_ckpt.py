import argparse
import os
import zipfile
import tempfile
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files

def process_checkpoint(name, local_dir, repo_id):
    """
    Downloads and extracts a single checkpoint.
    """
    zip_filename = f"{name}.zip"
    repo_path = f"Checkpoints/Physics-aware_Video_Generation/{zip_filename}"
    
    # Define the target directory for extraction: local_dir/{name}/
    extract_dir = os.path.join(local_dir, name)
    os.makedirs(extract_dir, exist_ok=True)

    tqdm.write(f"🚀 Downloading [{zip_filename}] from Hugging Face...")
    
    # Use a temporary directory for downloading to avoid creating .cache in local_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type="dataset",
            local_dir=temp_dir,
            local_dir_use_symlinks=False  # Ensure we download a real file, not a symlink
        )
        
        tqdm.write(f"📦 Unzipping to [{extract_dir}]...")
        with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
            members = zip_ref.infolist()
            
            # Iterate and extract one by one to show a progress bar
            for member in tqdm(members, desc="   📦 Extracting", unit="file", leave=False, colour="yellow"):
                zip_ref.extract(member, extract_dir)
            
        # The downloaded zip file and the .cache folder are automatically deleted 
        # when the TemporaryDirectory context exits.
        
    tqdm.write(f"✅ Checkpoint [{name}] is ready at: {extract_dir}")


def main():
    # ==========================================
    # 1. Setup Argument Parser
    # ==========================================
    parser = argparse.ArgumentParser(description="Download and unzip checkpoints from Hugging Face.")
    parser.add_argument(
        "--name", 
        type=str, 
        required=False, 
        default=None, 
        help="Specific checkpoint name (e.g., 'lora_Wan-AI'). " \
        "If not provided, all available checkpoints will be downloaded."
    )
    parser.add_argument("--local_dir", type=str, default="./checkpoints", help="Local directory to save files.")
    args = parser.parse_args()

    repo_id = "vLAR/PhysInOne"
    prefix = "Checkpoints/Physics-aware_Video_Generation/"

    # ==========================================
    # 2. Determine which checkpoints to download
    # ==========================================
    if args.name:
        names_to_download = [args.name]
    else:
        tqdm.write("🔍 No specific name provided. Fetching list of all available checkpoints...")
        try:
            all_files = list_repo_files(repo_id, repo_type="dataset")
            # Filter for zip files in the specific directory
            names_to_download = [
                os.path.basename(f).replace('.zip', '') 
                for f in all_files 
                if f.startswith(prefix) and f.endswith('.zip')
            ]
        except Exception as e:
            tqdm.write(f"❌ Failed to fetch repository files: {e}")
            return

        if not names_to_download:
            tqdm.write("⚠️ No checkpoints found in the specified repository path.")
            return
            
        tqdm.write(f"✅ Found {len(names_to_download)} checkpoint(s): {', '.join(names_to_download)}")

    # ==========================================
    # 3. Process each checkpoint
    # ==========================================
    for i, name in enumerate(names_to_download):
        tqdm.write(f"\n{'='*20} [{i+1}/{len(names_to_download)}] Processing: {name} {'='*20}")
        try:
            process_checkpoint(name, args.local_dir, repo_id)
        except Exception as e:
            # Catch errors (e.g., file not found, network issues) so the loop can continue
            tqdm.write(f"❌ Failed to process [{name}]: {e}")
            continue

    # ==========================================
    # 4. Completion Message
    # ==========================================
    tqdm.write(f"\n🎉 All requested checkpoints have been processed and saved to: {args.local_dir}")

if __name__ == "__main__":
    main()