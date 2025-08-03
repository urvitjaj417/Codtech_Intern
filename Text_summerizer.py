import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import logging
import re
import argparse
import sys
from langdetect import detect
from transformers import BartForConditionalGeneration, BartTokenizer, MarianMTModel, MarianTokenizer
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArticleSummarizer:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        logging.info("Loading BART tokenizer and model...")
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        logging.info("Model loaded successfully.")
        self.create_gui()

    def translate_to_english(self, text):
        try:
            src_lang = detect(text)
            logging.info(f"Detected language: {src_lang}")
            if src_lang == 'en':
                return text

            model_name = f"Helsinki-NLP/opus-mt-{src_lang}-en"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            translated = model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            logging.info("Translation successful.")
            return translated_text
        except Exception as e:
            logging.warning(f"Translation failed: {e}")
            return text

    def summarize_text(self, text, max_length=150, min_length=40):
        text = re.sub(r'\s+', ' ', text).strip()
        text = self.translate_to_english(text)
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

        logging.info("Generating summary...")
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logging.info("Summary generation complete.")
        return summary

    def process_text(self):
        text = self.input_text.get("1.0", tk.END)

        if len(text.strip()) < 100:
            messagebox.showwarning("Warning", "Please enter a longer text (at least 100 characters).")
            return

        self.status_label.config(text="Summarizing...")
        self.root.update()

        try:
            summary = self.summarize_text(text)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, summary)

            original_words = len(text.split())
            summary_words = len(summary.split())
            reduction = (1 - summary_words / original_words) * 100 if original_words > 0 else 0

            self.status_label.config(text=f"Done! Reduced by {reduction:.1f}% (from {original_words} to {summary_words} words)")
            logging.info(f"Summary reduced by {reduction:.1f}%")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_label.config(text="Error occurred")
            logging.error(f"Summary error: {str(e)}")

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.input_text.delete("1.0", tk.END)
                    self.input_text.insert(tk.END, content)
                    logging.info(f"Loaded file: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {str(e)}")
                logging.error(f"File load error: {str(e)}")

    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Multilingual Article Summarizer")
        self.root.geometry("900x700")

        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        middle_frame = tk.Frame(self.root)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)

        load_button = tk.Button(top_frame, text="Load File", command=self.load_file)
        load_button.pack(side=tk.LEFT, padx=5)

        summarize_button = tk.Button(top_frame, text="Summarize", command=self.process_text)
        summarize_button.pack(side=tk.LEFT, padx=5)

        left_frame = tk.Frame(middle_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(middle_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(left_frame, text="Original Text:").pack(anchor=tk.W)
        self.input_text = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(right_frame, text="Summary:").pack(anchor=tk.W)
        self.output_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.status_label = tk.Label(bottom_frame, text="Ready")
        self.status_label.pack(fill=tk.X)

        self.root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Article Summarizer")
    parser.add_argument('--file', type=str, help='Path to text file to summarize')
    parser.add_argument('--text', type=str, help='Text to summarize')
    parser.add_argument('--max_length', type=int, default=150, help='Maximum summary length')
    parser.add_argument('--min_length', type=int, default=40, help='Minimum summary length')
    parser.add_argument('--nogui', action='store_true', help='Run in terminal mode (no GUI)')
    args = parser.parse_args()

    summarizer = ArticleSummarizer()

    if args.nogui:
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        elif args.text:
            text = args.text
        else:
            logging.error("No input provided for summarization")
            print("Please provide --file or --text for summarization.")
            sys.exit(1)
        summary = summarizer.summarize_text(text, max_length=args.max_length, min_length=args.min_length)
        print("Summary:\n")
        print(summary)