from transformers import T5ForConditionalGeneration, T5Tokenizer


def txt_to_txt(language):
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

    input = "I am a proud citizen of Bangladesh"

    input_ids = tokenizer(f"translate English to {language}: "+input, return_tensors="pt").input_ids  # Batch size 1

    outputs = model.generate(input_ids)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return decoded


def main():
    
    languages = [
        "German",
        "French",
        "Spanish",
        "Italian",
        "Portuguese",
        "Dutch",
        "Russian",
        "Chinese",
        "Japanese",
        "Korean",
        "Arabic",
        "Turkish",
        "Polish",
        "Swedish",
        "Norwegian",
        "Finnish",
        "Danish",
        "Greek",
        "Hebrew",
        "Hindi",
        "Czech",
        "Romanian",
        "Thai",
        "Hungarian",
        "Bulgarian",
        "Indonesian",
        "Vietnamese",
        "Catalan",
        "Malay",
        "Swahili"
    ]
    
    for language in languages:
        text = txt_to_txt(language)
        print("Converted to", language + ":", text)
        
    
if __name__ == "__main__":
    main()
