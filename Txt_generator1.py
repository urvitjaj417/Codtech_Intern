import tkinter as tk
from tkinter import scrolledtext, messagebox
from transformers import pipeline, set_seed

# Load GPT-2 generator
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

# Create GUI app
class TextGenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Generative Text Model")
        self.root.geometry("600x400")

        tk.Label(root, text="Enter Topic or Prompt:").pack(pady=5)
        self.prompt_entry = tk.Entry(root, width=80)
        self.prompt_entry.pack(pady=5)

        self.generate_btn = tk.Button(root, text="Generate Text", command=self.generate_text)
        self.generate_btn.pack(pady=10)

        self.output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=15)
        self.output_box.pack(pady=10)

    def generate_text(self):
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            messagebox.showwarning("Missing Input", "Please enter a prompt or topic.")
            return

        try:
            self.output_box.delete('1.0', tk.END)
            self.output_box.insert(tk.END, "Generating...\n\n")
            self.root.update()

            result = generator(prompt, max_length=200, num_return_sequences=1)
            self.output_box.delete('1.0', tk.END)
            self.output_box.insert(tk.END, result[0]['generated_text'])
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Launch app
if __name__ == "__main__":
    root = tk.Tk()
    app = TextGenApp(root)
    root.mainloop()