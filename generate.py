import argparse
from pathlib import Path
import mlx.core as mx
from train import Trainer
from mlx_lm.sample_utils import make_sampler, make_logits_processors
import mlx.nn as nn
import time
from generate_lite import generate_lite, beam_search
import importlib.util
mx.set_default_device(mx.gpu)

# Check if mlx_vlm is available
has_mlx_vlm = importlib.util.find_spec("mlx_vlm") is not None
def main():
    parser = argparse.ArgumentParser(description='Generate text using a trained model')
    parser.add_argument('--run', type=str, required=True,
                       help='Name of the training run to use')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt to start generation')
    parser.add_argument('--max-tokens', type=int, default=256,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--min-p', type=float, default=0.05,
                       help='Minimum probability for nucleus sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.1,
                       help='Repetition penalty factor')
    parser.add_argument('--repetition-context-size', type=int, default=20,
                       help='Context size for repetition penalty')

    # Add arguments for vision-language models
    if has_mlx_vlm:
        parser.add_argument('--image', type=str, nargs='+', default=None,
                           help='Path(s) to image(s) for vision-language models')
        parser.add_argument('--system', type=str, default=None,
                           help='System prompt for vision-language models')
        parser.add_argument('--video', type=str, default=None,
                           help='Path to video file for video understanding')
        parser.add_argument('--fps', type=float, default=1.0,
                           help='Frames per second to sample from video')
    args = parser.parse_args()

    # Load run configuration and initialize trainer
    config_path = Path('runs') / args.run / 'config.yaml'
    if not config_path.exists():
        raise ValueError(f"Config not found for run: {args.run}")

    trainer = Trainer(str(config_path), for_training=False)

    # Load the final checkpoint
    checkpoint_path = Path('runs') / args.run / 'checkpoints' / 'step_final_model.safetensors'
    if not checkpoint_path.exists():
        checkpoint_path = Path('runs') / args.run / 'checkpoints' / 'step_final.safetensors'
        if not checkpoint_path.exists():
            raise ValueError(f"Final checkpoint not found for run: {args.run}")
    checkpoint_path = str(checkpoint_path)

    trainer.model.load_weights(checkpoint_path)

    # Check if we're using a vision-language model with image/video inputs
    is_vlm = False
    if has_mlx_vlm and (args.image is not None or args.video is not None):
        # First check if model_source is specified in the config
        if hasattr(trainer.config.model, 'model_source') and trainer.config.model.model_source == "mlx_vlm":
            is_vlm = True
        else:
            # Fall back to checking the model module
            model_module = trainer.model.__class__.__module__
            is_vlm = model_module.startswith('mlx_vlm.models')

    if is_vlm:
        # Import necessary functions from mlx_vlm
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        # Load processor and config
        processor = None
        config = None

        # Get model path from config
        model_path = Path('runs') / args.run

        try:
            # Try to load processor and config
            from mlx_vlm import load
            _, processor = load(str(model_path), weights_only=True)
            config = load_config(str(model_path))
        except Exception as e:
            print(f"Warning: Could not load processor and config from run directory: {e}")
            print("Falling back to default processing...")

        if processor is not None and config is not None:
            # Process images or video
            images = None
            if args.image is not None:
                images = args.image
            elif args.video is not None:
                # Process video frames
                try:
                    from mlx_vlm.utils import extract_frames_from_video
                    images = extract_frames_from_video(args.video, fps=args.fps)
                    print(f"Extracted {len(images)} frames from video at {args.fps} FPS")
                except Exception as e:
                    print(f"Error extracting video frames: {e}")
                    return

            # Apply chat template
            system_prompt = args.system if args.system else "You are a helpful assistant."
            formatted_prompt = apply_chat_template(
                processor, config, args.prompt, 
                system=system_prompt,
                num_images=len(images) if images else 0
            )

            # Generate output with vision-language model
            output = generate(
                trainer.model,
                processor,
                formatted_prompt,
                images,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=True
            )

            print(f"Output: {output}")
    else:
        # Standard text generation for non-VLM models
        # Prepare input
        tokens = [trainer.tokenizer.BOS_TOKEN] + trainer.tokenizer.tokenize(args.prompt)

        # Setup generation parameters
        sampler = make_sampler(temp=args.temperature, min_p=args.min_p)
        logits_processors = make_logits_processors(
            repetition_penalty=args.repetition_penalty,
            repetition_context_size=args.repetition_context_size
        )

        # Generate
        mx.random.seed(int(time.time() * 1000))
        greedy_output, greedy_score = generate_lite(
                trainer.model,
                mx.array(tokens),
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
                stop_tokens=[trainer.tokenizer.EOS_TOKEN],
                logits_processors=logits_processors
        )
        print(f"Output: {trainer.tokenizer.detokenize(greedy_output)}")

    # Print result
    #print(f"Greedy (Score: {score:.3f}): {trainer.tokenizer.detokenize(output)}")

if __name__ == "__main__":
    main()
