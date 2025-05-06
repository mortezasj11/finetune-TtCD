import subprocess
import logging

def install_packages():
    """
    Install required packages with robust error handling.
    Continues installation even if some packages fail to install.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    commands = [
        ["pip", "install", "numpy<2.0", "--force-reinstall"],
        ["pip", "install", "--upgrade",
         "--extra-index-url", "https://download.pytorch.org/whl/cu121",
         "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0"],
        ["pip", "install", "omegaconf", "torchmetrics==0.10.3", "fvcore", "iopath"],
        ["pip", "install", "--extra-index-url", "https://pypi.nvidia.com", "cuml-cu11"],
        ["pip", "install", "black==22.6.0", "flake8==5.0.4", "pylint==2.15.0"],
        ["pip", "install", "mmsegmentation==0.27.0"],
    ]
    
    failed_installations = []
    
    for i, command in enumerate(commands):
        try:
            logging.info(f"Installing package set {i+1}: {' '.join(command)}")
            result = subprocess.run(
                command, 
                check=False,  # Don't raise exception on failure
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            if result.returncode != 0:
                logging.warning(f"Package installation {i+1} returned non-zero exit code: {result.returncode}")
                logging.warning(f"stdout: {result.stdout}")
                logging.warning(f"stderr: {result.stderr}")
                failed_installations.append(i+1)
                continue
                
            logging.info(f"Successfully installed package set {i+1}")
            
        except Exception as e:
            logging.error(f"Error installing package set {i+1}: {str(e)}")
            failed_installations.append(i+1)
            continue
    
    if failed_installations:
        logging.warning(f"Failed to install the following package sets: {failed_installations}")
        return False
    return True

if __name__ == "__main__":
    install_packages() 