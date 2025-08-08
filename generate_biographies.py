from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def generate_biography(person_name: str) -> str:
    """
    Generate a biography using Qwen 1.5 from Hugging Face.
    """

    model_id = "Qwen/Qwen1.5-7B-Chat"

    # Keep the original prompt exactly the same
    prompt = f"Cho tôi tiểu sử của {person_name}"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Prepare chat-formatted text for Qwen 1.5
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Create generation pipeline
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    outputs = generator(
        prompt_text,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0]["generated_text"]

    # Remove the prompt from the generated text
    if generated.startswith(prompt_text):
        generated = generated[len(prompt_text):].strip()

    return generated

# # Example usage
# if __name__ == "__main__":
#     biography = generate_biography("Hồ Chí Minh")
#     print(biography)
