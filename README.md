## Installation

```shell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

1. Install dependencies:

   ```shell
   conda create -n nunchaku python=3.9
   conda activate nunchaku
   pip install ninja wheel diffusers transformers accelerate sentencepiece protobuf huggingface_hub
   
   # For gradio demos
   pip install peft opencv-python gradio spaces GPUtil  
   ```


2. Install `nunchaku` package:
    Make sure you have `gcc/g++>=11`. If you don't, you can install it via Conda on Linux:

    ```shell
    conda install -c conda-forge gxx=11 gcc=11
    ```

    Then build the package from source with
    
    ```shell
    git clone https://github.com/mit-han-lab/nunchaku.git
    cd nunchaku
    git submodule init
    git submodule update
    ```
    
    If you are building wheels for distribution, use:
    
    ```shell
    python -m build --wheel --no-isolation
    ```
    