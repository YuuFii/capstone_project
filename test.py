from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Josephgflowers/distillgpt2Cinder"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Contoh penggunaan
input_text = 'Dari 387 komentar yang dianalisis, 68.2% penonton memberikan respon positif, 21.4% negatif, dan 10.3% netral. Banyak penonton menyukai video karena hal-hal seperti "visual" dan "narasi". Namun, beberapa penonton menyampaikan kritik terkait "durasi" dan "penjelasan". Beberapa kata yang sering muncul dalam komentar antara lain: keren, jelas, panjang, saran, detail.'

generated_text = generate_text(input_text)
print(generated_text)