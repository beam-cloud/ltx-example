import os
import tempfile
from typing import Any, Dict

from beam import Image, Output, PythonVersion, env, schema, task_queue

if env.is_remote():
    from ltx_video.inference import InferenceConfig

# Create a new image that includes LTX-Video dependencies
image = Image(
    python_version=PythonVersion.Python312,
    python_packages=[
        "torch>=2.1.0",
        "diffusers>=0.28.2",
        "transformers>=4.47.2,<4.52.0",
        "sentencepiece>=0.1.96",
        "huggingface-hub~=0.30",
        "einops",
        "timm",
        "imageio[ffmpeg]",
        "av",
        "torchvision",
        "safetensors",
        "pyyaml",
        "pillow",
        # Install LTX-Video directly from git
        "git+https://github.com/Lightricks/LTX-Video.git#egg=ltx-video[inference]",
    ],
).add_commands(
    [
        "apt update && apt install -y ffmpeg",
    ]
)

CONFIG_PATH = "configs/ltxv-13b-0.9.8-dev.yaml"


def on_start():
    import torch
    from huggingface_hub import hf_hub_download
    from ltx_video.inference import create_ltx_video_pipeline, load_pipeline_config

    # Load default pipeline config
    pipeline_config = load_pipeline_config(CONFIG_PATH)

    # Download model if needed
    ltxv_model_name_or_path = pipeline_config["checkpoint_path"]
    if not os.path.isfile(ltxv_model_name_or_path):
        ltxv_model_path = hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename=ltxv_model_name_or_path,
            repo_type="model",
        )
    else:
        ltxv_model_path = ltxv_model_name_or_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if prompt enhancement is configured
    prompt_enhancement_words_threshold = pipeline_config.get(
        "prompt_enhancement_words_threshold", 0
    )
    enhance_prompt = prompt_enhancement_words_threshold > 0

    # Create pipeline using the internal func
    pipeline = create_ltx_video_pipeline(
        ckpt_path=ltxv_model_path,
        precision=pipeline_config["precision"],
        text_encoder_model_name_or_path=pipeline_config["text_encoder_model_name_or_path"],
        sampler=pipeline_config.get("sampler", None),
        device=device,
        enhance_prompt=enhance_prompt,
        prompt_enhancer_image_caption_model_name_or_path=pipeline_config.get(
            "prompt_enhancer_image_caption_model_name_or_path"
        )
        if enhance_prompt
        else None,
        prompt_enhancer_llm_model_name_or_path=pipeline_config.get(
            "prompt_enhancer_llm_model_name_or_path"
        )
        if enhance_prompt
        else None,
    )

    return pipeline, pipeline_config, device


# Define output schema (you can also do this for inputs if you want)
class Outputs(schema.Schema):
    video_url = schema.String()
    metadata = schema.JSON()


@task_queue(
    name="ltx-example",
    cpu=1,
    memory="16Gi",
    image=image,
    gpu="H100",
    on_start=on_start,
    outputs=Outputs,
)
def handler(
    context,
    prompt: str,
    height: int = 704,
    width: int = 1216,
    num_frames: int = 121,
    frame_rate: int = 30,
    seed: int = 171198,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    num_inference_steps: int = 30,
    guidance_scale: float = 3.0,
) -> Dict[str, Any]:
    pipeline, pipeline_config, device = context.on_start_value

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the inference config
        config = InferenceConfig(
            prompt=prompt,
            output_path=temp_dir,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            negative_prompt=negative_prompt,
            pipeline_config=CONFIG_PATH,
        )

        # Use the pre-loaded pipeline with the InferenceConfig
        # We need to temporarily monkey-patch the pipeline creation to use our cached one
        import ltx_video.inference

        original_create_pipeline = ltx_video.inference.create_ltx_video_pipeline

        def mock_create_pipeline(*args, **kwargs):
            return pipeline

        # Temporarily replace the pipeline creation function
        ltx_video.inference.create_ltx_video_pipeline = mock_create_pipeline

        try:
            # Now call infer with our config, it will use the cached pipeline
            from ltx_video.inference import infer

            infer(config=config)
        finally:
            # Restore the original function
            ltx_video.inference.create_ltx_video_pipeline = original_create_pipeline

        # Find the generated video file
        video_files = [f for f in os.listdir(temp_dir) if f.endswith(".mp4")]
        if not video_files:
            raise RuntimeError("No video file was generated")

        video_path = os.path.join(temp_dir, video_files[0])
        output = Output(path=video_path).save()

        print("Video download URL: => ", output.public_url())

        return {
            "video_url": output.public_url(),
            "metadata": {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "seed": seed,
            },
        }
