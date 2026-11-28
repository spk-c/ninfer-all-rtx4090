from huggingface_hub import HfApi
api=HfApi()
repos=[("ggml-org/Inkling-2-Mini-GGUF","0bdc580634ceac4c4d921f452777bf44228de9af"),("unsloth/Inkling-2-Mini-GGUF","8078d149286eec7193795413b9e0643da6d11fa1")]
for repo,rev in repos:
    info=api.model_info(repo,revision=rev,files_metadata=True)
    for f in info.siblings:
        if "mmproj" in f.rfilename.lower():
            print(repo,f.rfilename,f.size,f.lfs.sha256 if f.lfs else None)
