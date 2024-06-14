import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

# Initialize the translation pipeline
pipe = pipeline("translation", model="rudrashah/RLM-hinglish-translator")
tokenizer = AutoTokenizer.from_pretrained("rudrashah/RLM-hinglish-translator")
model = AutoModelForCausalLM.from_pretrained("rudrashah/RLM-hinglish-translator")

# Define a system prompt
system_prompt = {
    "role": "system",
    "content": "You are a helpful assistant. You reply with efficient answers.",
}

def translate_hinglish_to_english(hi_en_text):
    template = "Hinglish:\n{hi_en}\n\nEnglish:\n{en}"
    input_text = template.format(hi_en=hi_en_text, en="")
    
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generate the translation with adjusted max_length
    output = model.generate(inputs['input_ids'], max_length=inputs['input_ids'].shape[1] + 50)
    
    # Decode the output
    translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return translated_text

def chat_with_translation(message, summary):
    messages = [system_prompt]
    if summary:
        messages.append({"role": "system", "content": f"Summary of previous conversation: {summary}"})
    messages.append({"role": "user", "content": message})
    
    # Print the current output prompt
    print("Current Output Prompt:")
    for msg in messages:
        print(f"{msg['role']}: {msg['content']}")
    
    # Translate Hinglish to English
    translation = translate_hinglish_to_english(message)
    print(f"Translation: {translation}")
    
    return translation

def summarize_chat(history):
    summary = ""
    if history:
        all_messages = []
        for role, content in history:
            all_messages.append(f"{role}: {content}")
        summary_prompt = "Summarize the following conversation:\n" + "\n".join(all_messages)
        summary = translate_hinglish_to_english(summary_prompt)  # Assuming the translation pipeline can handle summaries too
    return summary

# Example usage
history = [("user", "tum log mere bina pizza khane gaye the?")]
summary = summarize_chat(history)
response = chat_with_translation("tum log mere bina pizza khane gaye the?", summary)
print(response)
