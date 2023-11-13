import torch
import whisper
import argparse

def hf_to_whisper_states(name):
    return (name.replace("model.", "")
                .replace("layers", "blocks")
                .replace("fc1", "mlp.0")
                .replace("fc2", "mlp.2")
                .replace("final_layer_norm", "mlp_ln")
                .replace(".self_attn.q_proj", ".attn.query")
                .replace(".self_attn.k_proj", ".attn.key")
                .replace(".self_attn.v_proj", ".attn.value")
                .replace(".self_attn_layer_norm", ".attn_ln")
                .replace(".self_attn.out_proj", ".attn.out")
                .replace(".encoder_attn.q_proj", ".cross_attn.query")
                .replace(".encoder_attn.k_proj", ".cross_attn.key")
                .replace(".encoder_attn.v_proj", ".cross_attn.value")
                .replace(".encoder_attn_layer_norm", ".cross_attn_ln")
                .replace(".encoder_attn.out_proj", ".cross_attn.out")
                .replace("decoder.layer_norm.", "decoder.ln.")
                .replace("encoder.layer_norm.", "encoder.ln_post.")
                .replace("embed_tokens", "token_embedding")
                .replace("encoder.embed_positions.weight", "encoder.positional_embedding")
                .replace("decoder.embed_positions.weight", "decoder.positional_embedding")
                .replace("layer_norm", "ln_post")
                .replace("proj_out.weight", "decoder.token_embedding.weight"))

def convert_model(hf_model_path: str,
                  whisper_model_path: str,
                  model: str = "medium",
                  device: str = "cpu"):
    # Load HF Model
    hf_state_dict = torch.load(hf_model_path, map_location=torch.device(device))

    # Rename layers
    for key in list(hf_state_dict.keys())[:]:
        new_key = hf_to_whisper_states(key)
        hf_state_dict[new_key] = hf_state_dict.pop(key)

    # Init Whisper Model and replace model weights
    whisper_model = whisper.load_model(model, device=device)
    whisper_model.load_state_dict(hf_state_dict)

    # Save Whisper Model
    torch.save(whisper_model.state_dict(), whisper_model_path)

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Convert Hugging Face model to Whisper model.')
    parser.add_argument('--hf_model_path',
                        type=str,
                        required=True,
                        help='Path to the Hugging Face model file.')
    parser.add_argument('--whisper_model_path',
                        type=str,
                        required=True,
                        help='Path to save the converted Whisper model.')
    parser.add_argument('--model',
                        type=str,
                        default='medium',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Size of the Whisper model.')

    # Parse arguments
    args = parser.parse_args()

    # Run the conversion
    convert_model(hf_model_path=args.hf_model_path,
                  whisper_model_path=args.whisper_model_path,
                  model=args.model)

if __name__ == "__main__":
    main()